"""Joint multi-region forecaster: one bank, one head, every region predicted.

What this changes, and why it is the fix the spatial arm needed
--------------------------------------------------------------
`nvmd_st` emits one region. In concat mode its exogenous mix reads a single row
of each coupling matrix -- `A_k[target, :]` with the self term zeroed -- so of
the `K x C x C` coupling parameters only `K x (C-1)` ever receive gradient. On
the 33-channel panel that is 256 live parameters out of 8,712, measured, i.e.
**97.1% of the spatial model is dead weight that no gradient reaches.** What
remains is not a spatial model at all: it is a per-band weighted sum of the other
channels into one target. Nothing in it is asked to represent regional structure,
which is why the whole exogenous effect could collapse onto one channel
(`ramp_VIC1` at 73-83%) without the objective noticing.

Predicting every region fixes that at the root. With `|T|` targets, `|T|` rows of
every `A_k` carry gradient, and one shared bank and head must explain five
different regions at once -- the pressure that makes a coupling matrix mean
something. It also turns one test year in one region into five, which is the
caveat standing over claims 5 and 6.

The confound, and the arm that removes it
-----------------------------------------
Joint training does two things at once: it puts gradient on the coupling, and it
hands the shared trunk five times the supervision. Those are not the same claim.
`coupling=False` freezes `A_k = I` while keeping the joint objective, so

    single           -> joint (coupling off)   = the multi-task effect
    joint (off)      -> joint (coupling on)    = the spatial effect

and neither can take credit for the other. As in `nvmd_st`, the coupling learns a
deviation from the identity, so at step 0 the coupled arm *is* the uncoupled one.
"""

import torch
import torch.nn as nn

from models.nvmd_v3 import StructuredSpectralNVMD
from models.nvmd_st import PerBandSpatialCoupling, CrossFilter


class JointSpatioTemporalNVMD(nn.Module):
    """x: (B, C, L) -> (B, T, 2K, L), one own+exogenous stack per target."""

    def __init__(self, C: int, targets, K: int = 8, signal_len: int = 96,
                 d_model: int = 128, adapt: float = 0.0, coupling: bool = True,
                 coupling_init: float = 0.0, xfilter: bool = False,
                 context: int = None):
        """`xfilter` is the only part of this model a linear reader cannot copy.

        A partition-of-unity band decomposition is an invertible linear map, and
        the per-band coupling is a weighted sum, so bands plus coupling read
        linearly sit inside the span of a linear map on the raw window -- ridge
        on bands and ridge on the raw window return identical predictions to
        0.0000 $/MWh, measured. Nothing in the additive pathway can therefore
        beat a ridge; the LSTM is left to rediscover a convex optimum by SGD,
        which is why it came in 1.3% behind one.

        `CrossFilter` breaks that. Its FiLM gate multiplies the own bands by a
        function of the exogenous state, and a product is not in the span of any
        linear map on the inputs. It is also the point of decomposing at all:
        the gate says *which time scale* the exogenous state modulates, which is
        not a statement one can make without bands. Both of its parts are
        identity- or zero-initialised, so step 0 is an exact no-op.

        `context` > signal_len decomposes over a longer window and hands the
        head only the last `signal_len` steps of each band. At window 96 nothing
        above 20.1 h exists in the bank's output, so the 168 h weekly cycle --
        the period the field's own naive benchmark is built on -- is absent from
        the input rather than merely unused. Lengthening only the bank's view
        costs an FFT instead of a longer sequence for the head.
        """
        super().__init__()
        self.C, self.K = C, K
        self.ctx = context or signal_len
        self.out_len = signal_len
        self.register_buffer("targets", torch.as_tensor(list(targets), dtype=torch.long))
        self.decomposer = StructuredSpectralNVMD(
            K=K, signal_len=self.ctx, d_model=d_model, adapt=adapt,
        )
        self.coupling = PerBandSpatialCoupling(C, K, enabled=coupling,
                                               init=coupling_init)
        self.xfilter = CrossFilter(K) if xfilter else None

    def exogenous(self, modes):
        """(B, C, K, L) -> (B, T, K, L): per target, the mix of every *other*
        channel. Reading `A_k` at all target rows at once is the whole point --
        this is the line that takes the live coupling rows from 1 to |T|."""
        if not self.coupling.enabled:
            return modes.new_zeros(modes.shape[0], len(self.targets),
                                   self.K, modes.shape[-1])
        A = self.coupling.matrices()                       # (K, C, C)
        w = A[:, self.targets, :].clone()                  # (K, T, C)
        # Zero each target's own column so the mix carries only what the
        # target's own modes do not already hold; `own` is concatenated untouched.
        w[:, torch.arange(len(self.targets)), self.targets] = 0.0
        return torch.einsum("bckl,ktc->btkl", modes, w)

    def forward(self, x):
        B, C, L = x.shape
        modes, masks = self.decomposer(x.reshape(B * C, 1, L))
        modes = modes.reshape(B, C, self.K, L)
        own = modes[:, self.targets]                       # (B, T, K, L) lossless
        exo = self.exogenous(modes)                        # (B, T, K, L)
        if self.out_len < L:                               # long bank view, short head view
            own, exo = own[..., -self.out_len:], exo[..., -self.out_len:]
        if self.xfilter is None:
            return torch.cat([own, exo], dim=2), masks     # (B, T, 2K, L)
        # `own` is carried through untouched as well as gated, so the lossless
        # encoding survives the gate exactly as it survives plain concat.
        T, Lo = own.shape[1], own.shape[-1]
        o = own.reshape(B * T, self.K, Lo)
        e = exo.reshape(B * T, self.K, Lo)
        e_x, fused = self.xfilter(o, e)
        out = torch.cat([o, e_x, fused], dim=1).reshape(B, T, 3 * self.K, Lo)
        return out, masks                                  # (B, T, 3K, L)


class JointForecaster(nn.Module):
    """Shared bank and LSTM trunk, one linear head per region.

    The trunk is shared on purpose: a bank that only works by specialising per
    region would not be evidence of shared band structure. The per-region head is
    an affine readout, small enough that it cannot absorb the regional dynamics
    the trunk is supposed to carry.
    """

    def __init__(self, C: int, targets, K: int = 8, signal_len: int = 96,
                 d_model: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2,
                 adapt: float = 0.0, coupling: bool = True,
                 coupling_init: float = 0.0, head: str = "lstm",
                 decompose: bool = True, n_chan: int = None,
                 xfilter: bool = False, context: int = None):
        """`head` selects what reads the bands.

        "lstm" is the original: a bidirectional LSTM whose *last hidden state*
        is handed to an MLP. That compresses 96 steps into one vector, and it is
        the part of this model a ridge on the same window does not have to do --
        the ridge reads all 96 x 37 values linearly and beat the LSTM version by
        1.3% at h=6 (p=0.040).

        "linear" reads the whole (2K, L) band stack with one linear map, as
        DLinear does. That makes the comparison against the ridge a clean one:
        same function class, same information, only the representation differs.
        It is therefore the sharpest test of whether the decomposition is worth
        anything -- linear-on-bands against linear-on-raw-window, 96x16 features
        against 96x37. If the bands win there, they earned their place; if they
        lose, no head will rescue them.
        """
        super().__init__()
        self.T, self.K, self.L, self.kind = len(targets), K, signal_len, head
        self.decompose = decompose
        if decompose:
            self.decomposer = JointSpatioTemporalNVMD(
                C=C, targets=targets, K=K, signal_len=signal_len,
                d_model=d_model, adapt=adapt, coupling=coupling,
                coupling_init=coupling_init, xfilter=xfilter, context=context,
            )
        n_in = ((3 if xfilter else 2) * K) if decompose else (n_chan or C)
        if head == "lstm":
            self.lstm = nn.LSTM(n_in, lstm_hidden, lstm_layers, batch_first=True,
                                bidirectional=True,
                                dropout=0.1 if lstm_layers > 1 else 0.0)
            # A per-region embedding on the trunk output, so one shared trunk can
            # be read differently per region without a trunk each.
            self.region = nn.Parameter(torch.zeros(self.T, lstm_hidden * 2))
            self.head = nn.Sequential(
                nn.Linear(lstm_hidden * 2, lstm_hidden), nn.ReLU(),
                nn.Linear(lstm_hidden, 1),
            )
        elif head == "linear":
            # One weight per (feature, timestep), shared across regions, plus a
            # per-region bias. Shared so a win cannot come from five separate
            # models, which is the same reason the LSTM trunk is shared.
            self.proj = nn.Linear(n_in * signal_len, 1)
            nn.init.zeros_(self.proj.bias)
            self.region = nn.Parameter(torch.zeros(self.T))
        else:
            raise ValueError(f"unknown head {head!r}")

    def forward(self, x):
        if self.decompose:
            feat, masks = self.decomposer(x)               # (B, T, 2K, L)
        else:
            # The no-decomposition control: the same head, reading the window
            # itself. Every target gets the whole panel, which is strictly more
            # than the band arms get -- their exogenous channels arrive as a
            # per-band projection -- so this control is not handicapped.
            feat = x.unsqueeze(1).expand(-1, self.T, -1, -1)
            masks = None
        B, T, C2, L = feat.shape
        if self.kind == "linear":
            y = self.proj(feat.reshape(B * T, C2 * L)).reshape(B, T) + self.region
            return feat, masks, None, y
        h, _ = self.lstm(feat.reshape(B * T, C2, L).permute(0, 2, 1))
        z = h[:, -1].reshape(B, T, -1) + self.region       # (B, T, 2H)
        return feat, masks, None, self.head(z).squeeze(-1)  # (B, T)


if __name__ == "__main__":
    B, C, K, L = 4, 38, 8, 96
    tg = [1, 34, 35, 36, 37]
    x = torch.randn(B, C, L)

    off = JointForecaster(C, tg, K=K, signal_len=L, coupling=False)
    on = JointForecaster(C, tg, K=K, signal_len=L, coupling=True)
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    off.eval(); on.eval()        # the LSTM has dropout; train mode is not a test
    with torch.no_grad():
        a, b = off(x)[3], on(x)[3]
    print("output shape:", tuple(b.shape), "(B, T)")
    print("coupling=True at init == coupling=False:", torch.allclose(a, b, atol=1e-6))

    # Every target's own modes must still sum to its own input window.
    with torch.no_grad():
        feat, _ = on.decomposer(x)
        own = feat[:, :, :K]
        err = (own.sum(2) - x[:, tg]).abs().max().item()
    print("own-mode reconstruction error:", err)

    # The claim this file exists for: live coupling rows go from 1 to |T|.
    on.train(); on.zero_grad(); on(x)[3].sum().backward()
    g = on.decomposer.coupling.delta.grad
    rows = (g.abs().sum(dim=(0, 2)) > 0).nonzero().flatten().tolist()
    print(f"coupling rows with gradient: {rows}  (targets {tg})")
    print(f"live parameters: {(g.abs() > 0).sum().item()} / {g.numel()}  "
          f"({(g.abs() > 0).float().mean() * 100:.1f}%)")
