"""
Spatio-temporal NVMD: per-band spatial coupling across NEM regions.

Why spatial, and why per band
-----------------------------
A decomposition is an invertible transform of its input -- it adds no
information, only conditioning.  That is why the measured accuracy margin over
classical VMD is small and shrinks as the predictor strengthens (4.9% Linear ->
1.0% LSTM), and why the training loss is nearly flat in the decomposition
parameters (0.9% spread across banks that differ enormously).  Breaking that
ceiling requires *new information*, not a better basis.

Interconnected electricity markets couple at frequency-dependent strength:
slow components (weather, demand cycles, fuel costs) are shared across regions,
while fast components (local congestion, unit outages) are region-specific
because interconnector limits bind and decouple them.  A single spatial
correlation cannot express that; one coupling matrix per band can.

Construction
------------
    X (B, R, L) -- R regions
      -> shared StructuredSpectralNVMD per region -> (B, R, K, L)
      -> per-band mixing  M'[:, :, k, :] = A_k @ M[:, :, k, :]
      -> emit the target region's K modes

A_k is initialised to the identity, so at init this is *exactly* temporal-only
NVMD, and the output is K channels either way -- the downstream predictor is
unchanged.  `--coupling 0` freezes A_k = I, giving an exact ablation rather
than a separate model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nvmd_v3 import StructuredSpectralNVMD


class PerBandSpatialCoupling(nn.Module):
    """One R x R mixing matrix per frequency band, identity-initialised."""

    def __init__(self, R: int, K: int, enabled: bool = True):
        super().__init__()
        self.R, self.K, self.enabled = R, K, enabled
        eye = torch.eye(R).unsqueeze(0).repeat(K, 1, 1)      # (K, R, R)
        self.register_buffer("eye", eye.clone())
        # Learn the *deviation* from identity so init is exactly temporal NVMD.
        self.delta = nn.Parameter(torch.zeros(K, R, R))

    def matrices(self):
        return self.eye + self.delta if self.enabled else self.eye

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """modes: (B, R, K, L) -> (B, R, K, L)"""
        if not self.enabled:
            return modes
        A = self.matrices()                                   # (K, R, R)
        # (B,R,K,L) -> (K,B,L,R) so the mix is a plain matmul over R
        m = modes.permute(2, 0, 3, 1)                         # (K, B, L, R)
        mixed = torch.einsum("kbli,kji->kblj", m, A)          # (K, B, L, R)
        return mixed.permute(1, 3, 0, 2)                      # (B, R, K, L)

    def exogenous(self, modes: torch.Tensor, target: int = 0) -> torch.Tensor:
        """Per-band sum over the *other* channels only.  (B,R,K,L) -> (B,K,L)

        Row `target` of A_k with the self term zeroed, so this carries strictly
        the information the target does not already hold.  Used by concat mode,
        where the target's own modes are passed through untouched alongside it.
        """
        if not self.enabled:
            return torch.zeros_like(modes[:, target])
        A = self.matrices()                                   # (K, R, R)
        w = A[:, target, :].clone()                           # (K, R)
        w[:, target] = 0.0
        return torch.einsum("brkl,kr->bkl", modes, w)

    def sparsity_loss(self):
        """L1 on off-diagonal coupling, so the learned topology stays readable."""
        if not self.enabled:
            return torch.zeros((), device=self.delta.device)
        off = ~torch.eye(self.R, dtype=torch.bool, device=self.delta.device)
        return self.delta[:, off].abs().mean()

    @torch.no_grad()
    def coupling_strength(self):
        """(K, R, R) absolute off-diagonal coupling, for inspection."""
        A = self.matrices().clone()
        idx = torch.arange(self.R)
        A[:, idx, idx] = 0.0
        return A.abs().cpu()


class CrossFilter(nn.Module):
    """Cross-filtering between the temporal (own) and spatial (exogenous) streams.

    Concat keeps the two streams independent and hands both to the head, which
    projects them linearly at its input.  Two things that cannot express:

    1. **Cross-band transfer.**  Coupling is strictly within-band -- exo band k
       only ever reaches own band k -- so a 6 h wind ramp can never influence the
       72 h price trend.  `band_mix` is a K x K transfer, identity-initialised.
    2. **Conditional effect.**  Wind matters more when the system is tight; that
       is a product, and a linear input projection can only add.  `film` gates the
       own stream on the exogenous state (FiLM), zero-initialised.

    Both inits make this an exact no-op at step 0, so training starts from concat
    behaviour and can only depart from it if the data pays for it -- the same
    property that let concat start from temporal-only NVMD.
    """

    def __init__(self, K: int, hidden: int = 32):
        super().__init__()
        self.K = K
        self.band_mix = nn.Parameter(torch.eye(K))            # (K, K), identity
        self.film = nn.Sequential(
            nn.Conv1d(K, hidden, 1), nn.GELU(), nn.Conv1d(hidden, 2 * K, 1),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, own: torch.Tensor, exo: torch.Tensor):
        """own, exo: (B, K, L) -> cross-band exo (B, K, L), gated own (B, K, L)"""
        exo_x = torch.einsum("kj,bjl->bkl", self.band_mix, exo)
        gamma, beta = self.film(exo_x).chunk(2, dim=1)
        return exo_x, own * (1.0 + gamma) + beta


class SpatioTemporalNVMD(nn.Module):
    def __init__(self, R: int, K: int = 8, signal_len: int = 96,
                 d_model: int = 128, adapt: float = 0.0,
                 coupling: bool = True, target: int = 0, concat: bool = False,
                 xfilter: bool = False):
        super().__init__()
        self.R, self.K, self.target = R, K, target
        self.concat = concat or xfilter
        self.xfilter = CrossFilter(K) if xfilter else None
        # One shared bank across regions: the physical bands are the same
        # everywhere, and sharing keeps A_k the only spatial parameter.
        self.decomposer = StructuredSpectralNVMD(
            K=K, signal_len=signal_len, d_model=d_model, adapt=adapt,
        )
        self.coupling = PerBandSpatialCoupling(R, K, enabled=coupling)

    def forward(self, x: torch.Tensor):
        """x: (B, R, L) -> (B, K, L), or (B, 2K, L) in concat mode; plus masks.

        Default (mix) mode *replaces* the target's modes with the mixed ones, so
        the partition of unity is lost: measured recon error 1.82 vs 0.00, with
        cross terms 2-6.5x the self term.  That corrupts the DC/trend and daily
        bands -- which costs MAE on ordinary intervals -- while the fast bands
        gain genuine spike information, which is why MAE and RMSE disagreed.

        Concat mode keeps the target's own modes untouched (they still sum
        exactly to its input window) and *appends* the purely-exogenous mix, so
        the head gets both instead of trading one for the other.
        """
        B, R, L = x.shape
        flat = x.reshape(B * R, 1, L)
        modes, masks = self.decomposer(flat)                  # (B*R, K, L)
        modes = modes.reshape(B, R, self.K, L)
        if not self.concat:
            return self.coupling(modes)[:, self.target], masks
        own = modes[:, self.target]                           # (B, K, L) lossless
        exo = self.coupling.exogenous(modes, self.target)     # (B, K, L)
        if self.xfilter is None:
            return torch.cat([own, exo], dim=1), masks        # (B, 2K, L)
        # `own` is still carried through untouched, so the lossless encoding
        # survives the cross-filter exactly as it survives plain concat.
        exo_x, fused = self.xfilter(own, exo)
        return torch.cat([own, exo_x, fused], dim=1), masks   # (B, 3K, L)


class STForecaster(nn.Module):
    """Spatio-temporal decomposer + LSTM head on the target region's modes."""

    def __init__(self, R: int, K: int = 8, signal_len: int = 96,
                 d_model: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2,
                 adapt: float = 0.0, coupling: bool = True, target: int = 0,
                 concat: bool = False, xfilter: bool = False):
        super().__init__()
        self.decomposer = SpatioTemporalNVMD(
            R=R, K=K, signal_len=signal_len, d_model=d_model,
            adapt=adapt, coupling=coupling, target=target, concat=concat,
            xfilter=xfilter,
        )
        n_in = 3 * K if xfilter else (2 * K if concat else K)
        self.lstm = nn.LSTM(n_in, lstm_hidden, lstm_layers, batch_first=True,
                            bidirectional=True,
                            dropout=0.1 if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden), nn.ReLU(),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x):
        modes, masks = self.decomposer(x)                     # (B, K, L)
        h, _ = self.lstm(modes.permute(0, 2, 1))
        return modes, masks, None, self.head(h[:, -1])


if __name__ == "__main__":
    B, R, K, L = 4, 5, 8, 96
    x = torch.randn(B, R, L)

    off = SpatioTemporalNVMD(R=R, K=K, signal_len=L, coupling=False)
    on = SpatioTemporalNVMD(R=R, K=K, signal_len=L, coupling=True)
    on.load_state_dict(off.state_dict(), strict=False)

    m_off, _ = off(x)
    m_on, _ = on(x)
    print("target modes:", tuple(m_on.shape))
    print("coupling=True at init == coupling=False:",
          torch.allclose(m_off, m_on, atol=1e-6))

    # With A_k = I the target region's modes must still sum to its own signal.
    print("recon err (coupling off):",
          (m_off.sum(1) - x[:, 0]).abs().max().item())
    print("off-diagonal coupling at init:",
          float(on.coupling.coupling_strength().max()))
