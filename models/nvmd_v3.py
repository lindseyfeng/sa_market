"""
NVMD v3 -- structured, interpretable neural mode decomposition.

What changed from v2 and why
----------------------------
v2 emitted a free (K, F) logit field and softmaxed across K.  Nothing bound a
channel to a frequency region, and the measured consequences were:

  * no DC/trend channel  (lowest learned band centre was 0.098; VMD's is 0.002)
  * band duplication     (8 of 13 channels between 0.14-0.21, eff_modes 5.8/13)
  * unstable identity    (only 24% of windows preserved channel ordering)
  * broadband modes      (bandwidths to 0.188 vs VMD's 0.014)
  * correlated errors    (summed AR error worse than VMD despite better
                          per-mode error -- errors added instead of cancelling)

v3 parameterises each mode by (centre, bandwidth), the same quantities VMD
solves for, and lets the hyper-network perturb only the *gap logits*.  Three
properties then hold by construction rather than by penalty:

  1. Centres are strictly increasing -- ordering cannot scramble, for any
     input, at any perturbation magnitude (cumsum of a softmax).
  2. Mode 1 is pinned to DC, so the trend always has a dedicated channel.
  3. Masks form a partition of unity, so sum(modes) == signal exactly.
     No residual channel is needed and none can become a junk dump.

Interpretability is then a property you can read off the model: every mode has
an explicit centre frequency and bandwidth, directly comparable to VMD's omega.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding."""

    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation, bias=False)
        num_groups = next((g for g in (8, 4, 2) if out_ch % g == 0), 1)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(F.pad(x, (self.pad_len, 0)))))


class SignalEncoder(nn.Module):
    """Multi-scale causal CNN -> one feature vector per window."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(1, 32, kernel_size=15),
            CausalConv1d(32, 64, kernel_size=7),
            CausalConv1d(64, d_model, kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.pool(self.net(x)).squeeze(-1))


class StructuredSpectralNVMD(nn.Module):
    """Decompose a 1-channel signal into K ordered, band-limited modes.

    Args:
        K:          number of modes
        signal_len: window length L
        d_model:    encoder width
        adapt:      max perturbation applied to the gap logits by the
                    hyper-network.  0.0 gives a fixed (non-adaptive) filter
                    bank; ordering holds for any value.
        bw_init:    initial bandwidth as a fraction of the mean band gap.
    """

    def __init__(self, K: int = 8, signal_len: int = 96, d_model: int = 128,
                 adapt: float = 0.5, bw_init: float = 0.6, ratio: float = 1.8):
        super().__init__()
        self.K = K
        self.L = signal_len
        self.F = signal_len // 2 + 1

        self.encoder = SignalEncoder(d_model)
        self.gap_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, K),
        )
        self.adapt = adapt

        # Gap logits -> softmax -> cumsum gives strictly increasing centres.
        # Geometric init (gap_k ~ ratio^k) puts resolution where the energy is:
        # price structure lives at low frequency.  At K=8, ratio=1.8 this lands
        # bands on 0.0066 / 0.0186 / 0.040 / 0.079, i.e. the daily (1/48) and
        # half-daily cycles -- close to what VMD independently converged to
        # (0.002 / 0.021 / 0.043 / 0.078).  Uniform init instead wastes bands
        # above 0.25 where there is almost no energy.
        self.gap_logits = nn.Parameter(
            torch.arange(K, dtype=torch.float32) * float(torch.log(torch.tensor(ratio)))
        )

        # Bandwidth scaled to each mode's local gap: narrow at low frequency,
        # wider at high.  A single global bandwidth would either smear the
        # trend or leave holes in the top octave.
        with torch.no_grad():
            c0 = self._centres_from(self.gap_logits.unsqueeze(0))[0]
            gap = torch.diff(c0, prepend=c0[:1], append=c0[-1:])
            local = 0.5 * (gap[:-1] + gap[1:])
        self.register_buffer("bw_min", 0.25 * local)
        self.log_bw = nn.Parameter(torch.log(bw_init * local))

        self.register_buffer("freqs", torch.fft.rfftfreq(signal_len))  # [0, 0.5]

    @staticmethod
    def _centres_from(logits: torch.Tensor) -> torch.Tensor:
        gaps = F.softmax(logits, dim=-1)
        cum = torch.cumsum(gaps, dim=-1)
        cum = cum - cum[..., :1]
        return cum / cum[..., -1:].clamp(min=1e-8) * 0.5

    # ------------------------------------------------------------------
    def bands(self, x: torch.Tensor = None):
        """Return (centres, bandwidths).

        centres: (B, K) if x is given else (1, K).  Strictly increasing along K,
        with centres[:, 0] == 0 (DC) and centres[:, -1] == Nyquist.
        """
        logits = self.gap_logits.unsqueeze(0)                      # (1, K)
        if x is not None and self.adapt > 0:
            delta = torch.tanh(self.gap_head(self.encoder(x)))     # (B, K)
            logits = logits + self.adapt * delta

        centres = self._centres_from(logits)                       # (B, K)
        bw = (self.bw_min + F.softplus(self.log_bw)).unsqueeze(0)  # (1, K)
        return centres, bw.expand_as(centres)

    def masks(self, x: torch.Tensor = None):
        """Gaussian band filters normalised to a partition of unity."""
        centres, bw = self.bands(x)                                # (B, K)
        f = self.freqs.view(1, 1, -1)                              # (1, 1, F)
        g = torch.exp(-0.5 * ((f - centres.unsqueeze(-1)) / bw.unsqueeze(-1)) ** 2)
        return g / g.sum(dim=1, keepdim=True).clamp(min=1e-8)      # (B, K, F)

    def forward(self, x: torch.Tensor):
        """x: (B, 1, L) -> imfs (B, K, L) with sum(imfs) == x, masks (B, K, F)."""
        L = x.shape[-1]
        m = self.masks(x)
        Xf = torch.fft.rfft(x.squeeze(1), n=L, dim=-1)              # (B, F)
        imfs = torch.fft.irfft(Xf.unsqueeze(1) * m, n=L, dim=-1)    # (B, K, L)
        return imfs, m

    # ------------------------------------------------------------------
    # Structural losses.  These shape the filter bank; the forecast loss
    # decides what the bank is *for*.
    # ------------------------------------------------------------------
    def bandwidth_loss(self, x=None):
        """VMD's own criterion: prefer narrowband modes."""
        _, bw = self.bands(x)
        return bw.mean()

    def separation_loss(self, x=None, margin: float = 1.0):
        """Require adjacent centres to sit at least `margin` bandwidths apart.

        Scale-free, so it does not fight the geometric spacing prior the way a
        fixed uniform target would.  This is what stops several channels
        collapsing onto one band -- the mechanism behind v2's correlated
        per-mode errors (median per-mode error better than VMD, summed error
        worse, because the errors added instead of cancelling).
        """
        centres, bw = self.bands(x)
        gaps = centres[:, 1:] - centres[:, :-1]
        need = margin * 0.5 * (bw[:, 1:] + bw[:, :-1])
        return F.relu(need - gaps).pow(2).mean()

    def overlap_loss(self, x=None):
        """Penalise pairwise spectral overlap between distinct bands."""
        m = self.masks(x)
        n = m / m.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gram = torch.bmm(n, n.transpose(1, 2))                      # (B, K, K)
        off = ~torch.eye(self.K, dtype=torch.bool, device=gram.device)
        return gram[:, off].pow(2).mean()

    @torch.no_grad()
    def describe(self, x=None, fs_per_step: float = 0.5):
        """Human-readable band table.  fs_per_step = hours per sample."""
        centres, bw = self.bands(x)
        c = centres.mean(0).cpu().numpy()
        b = bw.mean(0).cpu().numpy()
        rows = []
        for k in range(self.K):
            period = (1.0 / c[k] * fs_per_step) if c[k] > 1e-6 else float("inf")
            rows.append((k + 1, float(c[k]), float(b[k]), period))
        return rows


class NVMDv3Forecaster(nn.Module):
    """Decomposer + forecast head, trained end-to-end on next-step MSE.

    head='lstm'   -- expressive, but the modes get tuned to what an LSTM can
                     exploit, which biases any later cross-predictor benchmark.
    head='linear' -- a single linear map over the flattened modes.  It cannot
                     compensate for a poor decomposition, so the *modes*
                     themselves have to carry the signal.  Use this when the
                     decomposition is meant to transfer across predictors.
    """

    def __init__(self, K: int = 8, signal_len: int = 96, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 bidirectional: bool = True, adapt: float = 0.5,
                 head: str = "lstm", tail: int = 48):
        super().__init__()
        self.decomposer = StructuredSpectralNVMD(
            K=K, signal_len=signal_len, d_model=d_model, adapt=adapt,
        )
        self.head_kind = head
        # Only the last `tail` steps feed the head, matching the benchmark's
        # window so the decomposer is optimised for the same view it is scored on.
        self.tail = min(tail, signal_len)

        if head == "linear":
            self.lstm = None
            self.head = nn.Linear(K * self.tail, 1)
        else:
            self.lstm = nn.LSTM(K, lstm_hidden, lstm_layers, batch_first=True,
                                bidirectional=bidirectional,
                                dropout=0.1 if lstm_layers > 1 else 0.0)
            out_dim = lstm_hidden * (2 if bidirectional else 1)
            self.head = nn.Sequential(
                nn.Linear(out_dim, lstm_hidden), nn.ReLU(),
                nn.Linear(lstm_hidden, 1),
            )

    def forward(self, x):
        imfs, masks = self.decomposer(x)
        if self.head_kind == "linear":
            y = self.head(imfs[:, :, -self.tail:].flatten(1))
        else:
            h, _ = self.lstm(imfs.permute(0, 2, 1))
            y = self.head(h[:, -1])
        return imfs, masks, None, y


if __name__ == "__main__":
    B, L, K = 4, 96, 8
    x = torch.randn(B, 1, L)
    dec = StructuredSpectralNVMD(K=K, signal_len=L)
    imfs, m = dec(x)
    print("IMFs        :", tuple(imfs.shape))
    print("Recon error :", (imfs.sum(1) - x.squeeze(1)).abs().max().item())

    c, _ = dec.bands(x)
    print("Centres ordered for every sample:",
          bool((c[:, 1:] - c[:, :-1] > 0).all()))
    print("Mode 1 at DC:", float(c[:, 0].abs().max()))
    print("\nBand table (centre, bandwidth, period in hours):")
    for k, cc, bb, per in dec.describe():
        print(f"  mode {k}: centre={cc:.4f}  bw={bb:.4f}  period={per:8.2f} h")
