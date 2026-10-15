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
import torch.nn.functional as F

from models.nvmd_v3 import StructuredSpectralNVMD
from models.nvmd_st import PerBandSpatialCoupling, CrossFilter




class TemporalBandConv(nn.Module):
    """One depthwise convolution per band, dilated to that band's own period.

    Band k oscillates at roughly `centre_k` cycles per step, so whatever
    structure it carries lives on a scale of 1/centre_k steps. A single kernel
    width shared across bands is wrong at both ends: it either smears the trend
    or cannot span one cycle of the fastest band. Setting the dilation per band
    makes every kernel cover about half of its own period, which is the one
    thing a plain LSTM over 96 raw steps cannot cheaply do.

    Output is zero-initialised and GELU(0) = 0, so the block contributes a
    channel of zeros at step 0 and the head starts exactly where it would have
    without it. Dilations come from the *initial* centres, so the kernel
    geometry is fixed even though the centres are learned.
    """

    def __init__(self, K: int, centres, L: int, kernel: int = 5):
        super().__init__()
        self.K, self.kernel = K, kernel
        self.dilations = []
        for c in [float(v) for v in centres]:
            span = float(L) if c <= 1e-6 else min(1.0 / c, float(L))
            d = int(round(span / 2.0 / (kernel - 1)))
            self.dilations.append(max(1, min(d, max(1, L // (kernel - 1)))))
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel, padding=((kernel - 1) * d) // 2, dilation=d)
            for d in self.dilations])
        for c in self.convs:
            nn.init.zeros_(c.weight)
            nn.init.zeros_(c.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, L) -> (B, K, L)"""
        L = x.shape[-1]
        out = [self.convs[k](x[:, k:k + 1])[..., :L] for k in range(self.K)]
        return F.gelu(torch.cat(out, dim=1))


class BandAggregator(nn.Module):
    """Local feature extraction across bands, and a nonlinear way to pool them.

    Concatenating K band signals and handing them to an LSTM is the weakest
    possible aggregation: the head gets K channels with no indication that they
    are ordered, that adjacent ones overlap, or that their amplitudes mean
    something jointly. Two local operators fix that, on the two axes the band
    stack actually has, and neither needs a graph.

    **Frequency axis.** Bands are ordered and their masks overlap, so band k-1,
    k, k+1 are literal neighbours in frequency and a harmonic sits about one
    band away. A 3-tap convolution *along k* is the natural local operator
    there. It is initialised to [0, 1, 0], the identity.

    **Amplitude axis.** `gamma, beta = f(envelopes)` then `x_k -> x_k (1+gamma_k)
    + beta_k`. This says how much band k matters right now depends on where the
    energy sits across all bands right now -- a statement that cannot be written
    down without a decomposition, and a product, so no linear reader and no tree
    on raw time steps can copy it. The last layer is zero-initialised.

    Both are exact no-ops at step 0, and the untouched `own` stack is carried
    alongside the aggregated one, so this can only add.

    Not done here, and the next two things to try:
      * a depthwise temporal conv whose dilation is set per band to that band's
        own period, so band k looks exactly as far back as its oscillation runs
        instead of sharing one kernel width with the trend;
      * over regions, a 3-tap conv along the grid order QLD1-NSW1-VIC1-SA1 with
        TAS1 beside VIC1. The NEM is a path, so a graph convolution on it *is*
        an ordinary 1-D convolution -- there is no GNN to build. That would be a
        restriction of the current full C x C coupling, not an addition, and the
        case for restricting is that 1,440 live coupling entries are chasing a
        regional signal worth 0.4-1.9% of the variance once the common mode is
        removed.
    """

    def __init__(self, K: int, hidden: int = 16):
        super().__init__()
        self.K = K
        self.mix = nn.Conv1d(1, 1, 3, padding=1)            # along the band axis
        with torch.no_grad():
            self.mix.weight.zero_()
            self.mix.weight[0, 0, 1] = 1.0                  # identity tap
            self.mix.bias.zero_()
        self.film = nn.Sequential(
            nn.Conv1d(K, hidden, 1), nn.GELU(), nn.Conv1d(hidden, 2 * K, 1),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, own: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """own, env: (B, K, L) -> aggregated (B, K, L)."""
        B, K, L = own.shape
        y = own.permute(0, 2, 1).reshape(B * L, 1, K)       # bands as the length axis
        y = self.mix(y).reshape(B, L, K).permute(0, 2, 1)   # (B, K, L)
        gamma, beta = self.film(env).chunk(2, dim=1)
        return y * (1.0 + gamma) + beta


class JointSpatioTemporalNVMD(nn.Module):
    """x: (B, C, L) -> (B, T, 2K, L), one own+exogenous stack per target."""

    def __init__(self, C: int, targets, K: int = 8, signal_len: int = 96,
                 d_model: int = 128, adapt: float = 0.0, coupling: bool = True,
                 coupling_init: float = 0.0, xfilter: bool = False,
                 context: int = None, expand=(), gate: bool = False,
                 envelope: bool = False, bandgate: bool = False,
                 gate_additive: bool = False, xenv: bool = False,
                 tconv: bool = False, exo_rank: int = 1, mca: int = 0):
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
        self.register_buffer("expand",
                             torch.as_tensor(list(expand), dtype=torch.long))
        self.register_buffer("targets", torch.as_tensor(list(targets), dtype=torch.long))
        self.envelope, self.xenv = envelope, xenv
        self.gate_additive = gate_additive
        self.decomposer = StructuredSpectralNVMD(
            K=K, signal_len=self.ctx, d_model=d_model, adapt=adapt,
            gate=gate, envelope=envelope, gate_additive=gate_additive,
            mca=mca,
        )
        if xenv and not envelope:
            raise ValueError("+xenv needs +env: it reuses the analytic signal")
        self.tconv = None
        if tconv:
            with torch.no_grad():
                c0, _ = self.decomposer.bands(None)
            self.tconv = TemporalBandConv(K, c0[0].tolist(), signal_len)
        self.coupling = PerBandSpatialCoupling(C, K, enabled=coupling,
                                               init=coupling_init)
        # The exogenous pathway is a rank-1 projection: A_k[r, :] turns every
        # non-target channel into ONE scalar per band. At 37 channels that is
        # 36 -> 1; at 86 it is 85 -> 1, strictly more lossy, and it is why
        # feeding this architecture more channels makes it worse while the same
        # channels make a ridge better -- measured, +env goes 36.30 -> 37.01
        # from 37 to 86 channels while arx_window goes 36.37 -> 36.13.
        #
        # exo_rank > 1 adds M-1 further components per band, so the exogenous
        # capacity scales with the panel instead of staying at one. The
        # original component is untouched and the extras are seeded small
        # rather than at zero: zero-initialised extras would receive identical
        # gradients forever and never separate from each other.
        self.exo_rank = int(exo_rank)
        self.exo_extra = None
        if self.exo_rank > 1 and coupling:
            e = torch.empty(K, len(self.targets), self.exo_rank - 1, C)
            nn.init.normal_(e, 0.0, 0.01)
            self.exo_extra = nn.Parameter(e)
        self.xfilter = CrossFilter(K) if xfilter else None
        if bandgate and not envelope:
            raise ValueError("+bgate needs +env: the gate reads band envelopes")
        self.aggregator = BandAggregator(K) if bandgate else None

    def expanded(self, modes):
        """Channels handed to the head as their own full band set.

        The default collapses all 36 non-target channels into one scalar per
        band. Measured on the panel, that is the wrong thing to compress:
        after removing the cross-region common mode, regional structure is
        0.4-1.9% of the variance in every band, so the other four regional
        prices are nearly redundant with the target's own common mode -- trees
        on five regional price windows score 40.25 against 41.37 on the target's
        own, a 2.7% gain, while adding the physical drivers takes it to 38.92, a
        further 3.3%. Drivers are causes and prices are effects, so the drivers
        carry information the price common mode does not.

        So the drivers get their own bands and the rest stays collapsed. The
        collapse is not a capacity problem where it remains: a rank-8 ridge
        already captures 98% of the exogenous value a full-rank one does.
        """
        if not len(self.expand):
            return None
        return modes[:, self.expand]                       # (B, X, K, L)

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
        out = torch.einsum("bckl,ktc->btkl", modes, w)
        if self.exo_extra is None:
            return out
        e = self.exo_extra.clone()                         # (K, T, M-1, C)
        e[:, torch.arange(len(self.targets)), :, self.targets] = 0.0
        ex = torch.einsum("bckl,ktmc->btmkl", modes, e)
        B_, T_, M_, K_, L_ = ex.shape
        return torch.cat([out, ex.reshape(B_, T_, M_ * K_, L_)], dim=2)

    def forward(self, x):
        B, C, L = x.shape
        modes, masks, aux = self.decomposer.decompose_rich(x.reshape(B * C, 1, L))
        modes = modes.reshape(B, C, self.K, L)
        own = modes[:, self.targets]                       # (B, T, K, L) lossless
        exo = self.exogenous(modes)                        # (B, T, K, L)
        # Envelope and instantaneous frequency of the target's own bands. Kept
        # out of `own` so the lossless stack still reaches the head untouched
        # and the FiLM path below still sees exactly K channels.
        rich = None
        if "env" in aux:
            rich = aux["env"].reshape(B, C, 2 * self.K, L)[:, self.targets]
        # The exogenous mix is linear in the bands, so the analytic signal of
        # the mix is the mix of the analytic signals -- which means the
        # magnitude has to be taken *after* coupling, not before. These are the
        # drivers' energy and phase at each scale, the thing the per-band
        # scalar mix throws away.
        if self.xenv:
            z = aux["z"].reshape(B, C, self.K, L)
            A = self.coupling.matrices()
            w = A[:, self.targets, :].clone()
            w[:, torch.arange(len(self.targets)), self.targets] = 0.0
            zx = torch.einsum("bckl,ktc->btkl", z, w.to(z.dtype))
            xr = self.decomposer.env_feats(zx.reshape(B * len(self.targets),
                                                      self.K, L))
            rich = torch.cat(
                [rich, xr.reshape(B, len(self.targets), 2 * self.K, L)], dim=2)
        if "mca" in aux:
            mc = aux["mca"].reshape(B, C, 3, L)[:, self.targets]
            rich = mc if rich is None else torch.cat([rich, mc], dim=2)
        if "gated" in aux:
            g = aux["gated"].reshape(B, C, self.K, L)[:, self.targets]
            rich = g if rich is None else torch.cat([rich, g], dim=2)
        if self.out_len < L:                               # long bank view, short head view
            own, exo = own[..., -self.out_len:], exo[..., -self.out_len:]
            if rich is not None:
                rich = rich[..., -self.out_len:]
        ex = self.expanded(modes)
        if ex is not None:
            if self.out_len < L:
                ex = ex[..., -self.out_len:]
            # Same expanded block for every target: it is panel-wide context,
            # not something the target index should change.
            ex = ex.unsqueeze(1).expand(-1, own.shape[1], -1, -1, -1)
            ex = ex.reshape(B, own.shape[1], -1, own.shape[-1])
        if self.tconv is not None:
            B_, T_, _, L_ = own.shape
            tc = self.tconv(own.reshape(B_ * T_, self.K, L_)
                            ).reshape(B_, T_, self.K, L_)
            rich = tc if rich is None else torch.cat([rich, tc], dim=2)
        agg = None
        if self.aggregator is not None:
            B_, T_, _, L_ = own.shape
            env = rich[:, :, :self.K]                      # (B, T, K, L)
            agg = self.aggregator(own.reshape(B_ * T_, self.K, L_),
                                  env.reshape(B_ * T_, self.K, L_))
            agg = agg.reshape(B_, T_, self.K, L_)
        if self.xfilter is None:
            parts = ([own, exo] + ([ex] if ex is not None else [])
                     + ([rich] if rich is not None else [])
                     + ([agg] if agg is not None else []))
            return torch.cat(parts, dim=2), masks          # (B, T, (2+X)K, L)
        # `own` is carried through untouched as well as gated, so the lossless
        # encoding survives the gate exactly as it survives plain concat.
        T, Lo = own.shape[1], own.shape[-1]
        o = own.reshape(B * T, self.K, Lo)
        e = exo.reshape(B * T, self.K, Lo)
        e_x, fused = self.xfilter(o, e)
        out = torch.cat([o, e_x, fused], dim=1).reshape(B, T, 3 * self.K, Lo)
        if ex is not None:
            out = torch.cat([out, ex], dim=2)
        if rich is not None:
            out = torch.cat([out, rich], dim=2)
        if agg is not None:
            out = torch.cat([out, agg], dim=2)
        return out, masks                                  # (B, T, (3+X)K, L)


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
                 xfilter: bool = False, context: int = None, expand=(),
                 reduce_to: int = 0, gate: bool = False,
                 envelope: bool = False, bandgate: bool = False,
                 gate_additive: bool = False, xenv: bool = False,
                 tconv: bool = False, horizons: int = 1, quantiles=(),
                 pool: str = "last", exo_rank: int = 1, mca: int = 0):
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
        self.h_index = 0          # set by the runner when horizons > 1
        self.decompose = decompose
        if decompose:
            self.decomposer = JointSpatioTemporalNVMD(
                C=C, targets=targets, K=K, signal_len=signal_len,
                d_model=d_model, adapt=adapt, coupling=coupling,
                coupling_init=coupling_init, xfilter=xfilter, context=context,
                expand=expand, gate=gate, envelope=envelope,
                bandgate=bandgate, gate_additive=gate_additive, xenv=xenv,
                tconv=tconv, exo_rank=exo_rank, mca=mca,
            )
        self.reduce = None
        if reduce_to:
            self.reduce = nn.Conv1d(n_chan or C, reduce_to, 1)
            self.decompose = False
        n_in = reduce_to if reduce_to else (
            (((3 if xfilter else 2) + len(expand) + (2 if envelope else 0)
              + (1 if bandgate else 0) + (1 if gate_additive else 0)
              + (2 if xenv else 0) + (1 if tconv else 0)
              + (exo_rank - 1 if coupling else 0)) * K
             + (3 if mca else 0))
            if decompose else (n_chan or C))
        # Multi-horizon and quantile output. The window list, the scored rows
        # and every regularisation term are unchanged: the trunk is the same
        # and only the final linear widens from 1 to H*Q, so with H=1 and no
        # quantiles the parameter tensors and the forward pass are identical
        # to before.
        #
        # Why widen at all. Each of the 44,535 training windows currently
        # produces five numbers, and the other eleven horizons it could be
        # supervised on are thrown away. Asking for the whole trajectory is
        # twelve times the supervision at no extra data cost, and it forces the
        # representation to carry dynamics rather than one lag. Quantiles are
        # the second half: MAE is minimised by the conditional median, and a
        # spike hour's conditional is not unimodal, so a point head is being
        # asked to summarise a mixture with one number.
        self.Q = tuple(quantiles) if quantiles else (0.5,)
        self.H = int(horizons)
        self.O = self.H * len(self.Q)
        self.q_median = min(range(len(self.Q)),
                            key=lambda i: abs(self.Q[i] - 0.5))
        # How the 96 steps are turned into one vector. This is the one place
        # the ridge has a structural advantage and the model does not: it
        # reads all 96 x C values linearly and never has to summarise them,
        # and it beat the LSTM arm by 1.3% at h=6.
        #
        #   last   the original -- the final hidden state, everything else
        #          thrown away. A bidirectional LSTM's last step is not even a
        #          summary of the whole window in both directions
        #   attn   a learned weight per timestep, so the readout can keep what
        #          it needs from anywhere in the window and still be nonlinear
        #   cat    last + mean + max, three cheap views instead of one
        self.pool = pool
        if head == "lstm":
            self.lstm = nn.LSTM(n_in, lstm_hidden, lstm_layers, batch_first=True,
                                bidirectional=True,
                                dropout=0.1 if lstm_layers > 1 else 0.0)
            if pool == "attn":
                self.att = nn.Linear(lstm_hidden * 2, 1)
                nn.init.zeros_(self.att.weight)
                nn.init.zeros_(self.att.bias)      # uniform weights at init
            wide = {"last": 1, "attn": 1, "cat": 3}[pool]
            # A per-region embedding on the trunk output, so one shared trunk can
            # be read differently per region without a trunk each.
            self.region = nn.Parameter(torch.zeros(self.T, lstm_hidden * 2 * wide))
            self.head = nn.Sequential(
                nn.Linear(lstm_hidden * 2 * wide, lstm_hidden), nn.ReLU(),
                nn.Linear(lstm_hidden, self.O),
            )
        elif head == "linear":
            # One weight per (feature, timestep), shared across regions, plus a
            # per-region bias. Shared so a win cannot come from five separate
            # models, which is the same reason the LSTM trunk is shared.
            self.proj = nn.Linear(n_in * signal_len, self.O)
            nn.init.zeros_(self.proj.bias)
            self.region = nn.Parameter(torch.zeros(self.T))
        else:
            raise ValueError(f"unknown head {head!r}")

    def forward(self, x):
        if self.reduce is not None:
            # The control that separates the two things the decomposition does.
            # It reduces 37 channels to the same width the band pathway hands
            # over, but with a plain learned projection applied per timestep --
            # so it keeps the dimension reduction and drops the non-causal
            # filtering. If most of the band pathway's 6.8% survives here, the
            # mechanism is capacity; if it does not, the mechanism is that band
            # masks carry whole-window information at every step and save the
            # LSTM from integrating 96 of them.
            feat = self.reduce(x).unsqueeze(1).expand(-1, self.T, -1, -1)
            return feat, None, None, self._read(feat)
        if self.decompose:
            feat, masks = self.decomposer(x)               # (B, T, 2K, L)
        else:
            # The no-decomposition control: the same head, reading the window
            # itself. Every target gets the whole panel, which is strictly more
            # than the band arms get -- their exogenous channels arrive as a
            # per-band projection -- so this control is not handicapped.
            feat = x.unsqueeze(1).expand(-1, self.T, -1, -1)
            masks = None
        return feat, masks, None, self._read(feat)

    def _read(self, feat):
        """(B, T, C2, L) -> (B, T) when O == 1, else (B, T, H, |Q|)."""
        B, T, C2, L = feat.shape
        if self.kind == "linear":
            o = self.proj(feat.reshape(B * T, C2 * L)).reshape(B, T, self.O)
            o = o + self.region.view(1, T, 1)
        else:
            h, _ = self.lstm(feat.reshape(B * T, C2, L).permute(0, 2, 1))
            if self.pool == "attn":
                # Zero-initialised logits make this a plain mean at step 0,
                # not the last step, so `attn` is not a no-op against `last` --
                # the comparison is between two readouts, both trained.
                w = torch.softmax(self.att(h), dim=1)       # (B*T, L, 1)
                g = (h * w).sum(1)
            elif self.pool == "cat":
                g = torch.cat([h[:, -1], h.mean(1), h.max(1).values], dim=-1)
            else:
                g = h[:, -1]
            z = g.reshape(B, T, -1) + self.region
            o = self.head(z)                               # (B, T, O)
        if self.O == 1:
            return o.squeeze(-1)                           # the original shape
        return o.view(B, T, self.H, len(self.Q))

    def point(self, o):
        """The scored prediction: target horizon, median quantile."""
        return o if o.dim() == 2 else o[:, :, self.h_index, self.q_median]


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
