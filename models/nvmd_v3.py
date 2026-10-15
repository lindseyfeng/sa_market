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

import math

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
                 adapt: float = 0.5, bw_init: float = 0.6, ratio: float = 1.8,
                 gate: bool = False, envelope: bool = False,
                 gate_additive: bool = False, mca: int = 0,
                 phase: bool = False, vmd_iters: int = 0):
        super().__init__()
        self.K = K
        self.L = signal_len
        self.F = signal_len // 2 + 1
        self.gate, self.envelope = gate, envelope
        # Measured 2026-10-15: gating in place cost 0.7-1.7% against the arm
        # that only *adds* features, and the checkpoints showed the gate had
        # very much opened (gate_c 0.34 mean, 0.98 max), so it is not that the
        # gate failed to train. The likely mechanism is that an in-place gate
        # destroys the partition of unity -- sum_k x_k stops reconstructing x
        # -- and takes information away from the head, where +env only ever
        # adds channels. `gate_additive` keeps the lossless bands and hands the
        # gated copy over as extra channels instead, which is the discipline
        # CrossFilter already follows.
        self.gate_additive = gate_additive
        # Morphological split: smooth + sparse, instead of only by frequency.
        #
        # A Gaussian band bank is localised in frequency and maximally
        # DELOCALISED in time, and the error here is dominated by time-localised
        # transients -- 115 spike rows carry 23.5% of it and negative-price
        # episodes another 27.5%. A spike is broadband in Fourier, so the bank
        # smears the one thing that matters across all K bands. Nothing
        # downstream can unsmear it, which is why every readout change so far
        # has been worth tenths of a percent.
        #
        # Electricity prices do not decompose by frequency, they decompose by
        # mechanism: p = base + cycle + scarcity premium - oversupply discount.
        # The first two are smooth and periodic and Fourier is right for them.
        # The other two are sparse, time-localised, and ASYMMETRIC -- a spike
        # runs to +15,500 $/MWh and an oversupply episode to -1,000, fifteen
        # times smaller and a different mechanism. One symmetric sinusoidal
        # basis cannot represent either.
        #
        # `mca` = how many of the lowest bands count as the smooth part. The
        # residual is soft-thresholded into a nonnegative and a nonpositive
        # sparse component with separately learned thresholds, which is
        # nonlinear and time-localised, i.e. exactly what the bank cannot do.
        # Thresholds scale with the residual's own RMS in the window, so the
        # split adapts per window rather than being a fixed level.
        # The masks are real and non-negative, so every atom this bank can
        # produce is time-symmetric -- measured at 1e-8, float precision. A
        # price spike is not symmetric: it rises in minutes and decays over
        # hours. A symmetric basis cannot represent an asymmetric transient no
        # matter how the bands are placed, and no downstream module can repair
        # that, which is the first thing about the decomposition's SHAPE that
        # has been wrong rather than merely limited.
        #
        # A learned quadratic phase response per band, M_k(f) -> M_k(f) e^{i
        # phi_k(f)}, makes the atoms asymmetric. It is applied in place, unlike
        # the magnitude gate, and that is the point: a phase rotation is an
        # allpass, exactly invertible and energy-preserving per band, so no
        # information is destroyed. The gate changed |W| and was lossy, and it
        # lost. theta = 0 is the identity.
        self.phase = phase
        if phase:
            self.theta = nn.Parameter(torch.zeros(K))
        # VMD's adaptivity, without VMD's leakage.
        #
        # What separates VMD from a filter bank is not that its modes are
        # narrowband -- ours are too -- but that its centre frequencies are
        # SOLVED FOR on each signal, by the energy-weighted spectral centroid
        #
        #     omega_k <- int w |u_k(w)|^2 dw / int |u_k(w)|^2 dw
        #
        # iterated to a fixed point. Our centres are learned parameters,
        # identical for every window, which is why the bank is a constant
        # linear map. `adapt` was meant to fix that but perturbs the gap logits
        # through an MLP -- a second black box, not VMD's rule -- and is worth
        # 0.6% (p=0.088).
        #
        # The reason this project exists is that published VMD forecasting
        # setups decompose the WHOLE series, future included, and leak. Running
        # VMD's own update inside the input window -- which ends at the
        # forecast origin -- keeps the adaptivity and cannot leak. The centroid
        # is parameter-free, differentiable and nonlinear in |X|^2, so this is
        # also the first thing here that makes the decomposition genuinely
        # nonlinear in its input rather than nonlinear in its weights.
        self.vmd_iters = int(vmd_iters)
        self.mca = int(mca)
        if self.mca:
            self.tau_p = nn.Parameter(torch.tensor(1.5))
            self.tau_n = nn.Parameter(torch.tensor(1.5))
        if gate:
            # Magnitude-dependent spectral gain. The masks are a partition of
            # unity, so sum_k x_k == x exactly and the bank hands the head a
            # re-slicing of something it already had -- which is why ridge on
            # bands equals ridge on the raw window to 0.0000 $/MWh. This breaks
            # that on purpose: the gain on a coefficient depends on how large
            # that coefficient is relative to its own band's energy, so a quiet
            # fast band is suppressed and a loud one passes. That is a learned
            # Wiener gain, it is not in the span of any linear map on x, and no
            # tree can form it either (it needs |rFFT| of the whole window).
            #
            # g = 1 + c * tanh(a * log(|W| / rms_k) + b), with c = 0 at init, so
            # step 0 is an exact no-op and the gate departs from the identity
            # only if the objective pays for it -- the same discipline as the
            # coupling and the FiLM gate.
            self.g_a = nn.Parameter(torch.ones(K))
            self.g_b = nn.Parameter(torch.zeros(K))
            self.g_c = nn.Parameter(torch.zeros(K))

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

    def _masks_from(self, centres, bw):
        f = self.freqs.view(1, 1, -1)                              # (1, 1, F)
        g = torch.exp(-0.5 * ((f - centres.unsqueeze(-1)) / bw.unsqueeze(-1)) ** 2)
        return g / g.sum(dim=1, keepdim=True).clamp(min=1e-8)      # (B, K, F)

    def vmd_centres(self, Xf, centres, bw):
        """VMD's centre-frequency update, run inside the window only.

        Xf: (B, F) complex spectrum of this window. Each iteration rebuilds the
        masks at the current centres, forms the mode spectra and moves every
        centre to its own energy-weighted centroid. Centres are clamped into
        [0, Nyquist] and re-sorted, because the centroid step has no ordering
        constraint of its own and VMD relies on its initialisation to keep the
        modes separated.
        """
        f = self.freqs.view(1, 1, -1)
        p = Xf.abs().pow(2).unsqueeze(1)                           # (B, 1, F)
        for _ in range(self.vmd_iters):
            m = self._masks_from(centres, bw)                      # (B, K, F)
            e = (m ** 2) * p                                       # |u_k(w)|^2
            centres = ((f * e).sum(-1) / e.sum(-1).clamp(min=1e-12))
            centres = centres.clamp(0.0, float(self.freqs[-1]))
            centres, _ = torch.sort(centres, dim=-1)
        return centres

    def masks(self, x: torch.Tensor = None):
        """Gaussian band filters normalised to a partition of unity."""
        centres, bw = self.bands(x)                                # (B, K)
        return self._masks_from(centres, bw)

    # Every magnitude below is sqrt(re^2 + im^2 + EPS) rather than .abs(), and
    # every rms adds EPS before the sqrt rather than clamping after it. Both
    # matter: a mask that has underflowed leaves |W| at exactly 0 (944 of
    # 8 x 49 coefficients on a random batch, measured), where d|W|/dW is 0/0,
    # and clamping the *output* of a sqrt does nothing for its backward pass.
    EPS = 1e-12

    def gain(self, W: torch.Tensor) -> torch.Tensor:
        """Per-coefficient gain in [1-|c|, 1+|c|]. W: (B, K, F) -> (B, K, F)."""
        mag = torch.sqrt(W.real ** 2 + W.imag ** 2 + self.EPS)
        rms = mag.pow(2).mean(-1, keepdim=True).add(self.EPS).sqrt()
        r = torch.log(mag / rms + 1e-6)
        a, b, c = (t.view(1, -1, 1) for t in (self.g_a, self.g_b, self.g_c))
        return 1.0 + c * torch.tanh(a * r + b)

    @staticmethod
    def analytic(W: torch.Tensor, L: int) -> torch.Tensor:
        """One-sided band coefficients -> the band's analytic signal (B, K, L).

        Zero the negative frequencies and double the strictly-positive ones, so
        `real(z) == irfft(W)` exactly while `|z|` is the band's envelope and
        `angle(z)` its instantaneous phase. Costs one ifft and gives the two
        features a narrowband signal actually has -- neither of which is in the
        span of a linear map, and neither of which an LSTM can read off a mode
        without integrating the whole window itself.
        """
        B, K, Fb = W.shape
        Z = W.new_zeros(B, K, L)
        Z[..., :Fb] = W
        hi = Fb - 1 if L % 2 == 0 else Fb        # exclude DC and, if present, Nyquist
        Z[..., 1:hi] = Z[..., 1:hi] * 2
        return torch.fft.ifft(Z, n=L, dim=-1)

    @classmethod
    def env_feats(cls, z: torch.Tensor) -> torch.Tensor:
        """Analytic signal -> (B, 2K, L): log1p envelope, instantaneous freq.

        Exposed on its own because the exogenous pathway needs the same two
        features of a *mix* of bands, and mixing is linear, so the analytic
        signal of the mix is the mix of the analytic signals -- the magnitude
        has to be taken after the mix, not before.
        """
        env = torch.sqrt(z.real ** 2 + z.imag ** 2 + cls.EPS)
        d = z[..., 1:] * z[..., :-1].conj()
        # atan2's gradient is 1/(x^2 + y^2). |d| gets down to 1e-5 on real
        # windows, which is a gradient of 1e10 straight into the bank and the
        # reason the first version of this trained to NaN in one epoch. Project
        # d onto the unit circle first: the feature is unchanged, and atan2 now
        # differentiates at radius 1.
        # The offset on the real part is not cosmetic. Several panel channels
        # are constant inside a 96-step window (the calendar ones), so every
        # band above DC is *exactly* zero for them and atan2(0, 0) has a 0/0
        # gradient -- which is what still trained to NaN after the radius fix.
        # Offsetting first sends a dead band to atan2(0, 1) = 0, the right
        # "no phase information" value, with a finite derivative.
        dr, di = d.real + 1e-8, d.imag
        den = torch.sqrt(dr ** 2 + di ** 2 + cls.EPS)
        inst = torch.atan2(di / den, dr / den) / (2.0 * math.pi)
        inst = torch.cat([inst[..., :1], inst], dim=-1)             # (B, K, L)
        return torch.cat([torch.log1p(env), inst], dim=1)           # (B, 2K, L)

    def morph(self, imfs: torch.Tensor) -> torch.Tensor:
        """(B, K, L) bands -> (B, 3, L): sparse up, sparse down, texture.

        smooth   the lowest `mca` bands, which carry the trend and the daily
                 and weekly cycles
        s_pos    soft-thresholded positive residual: the scarcity premium
        s_neg    soft-thresholded negative residual: the oversupply discount
        texture  whatever is left, so the three sum back to x - smooth and
                 nothing is thrown away

        Separate thresholds because the two tails are not reflections of each
        other: the market cap is +15,500 $/MWh and the floor is -1,000.
        """
        smooth = imfs[:, :self.mca].sum(1)                     # (B, L)
        r = imfs.sum(1) - smooth
        rms = r.pow(2).mean(-1, keepdim=True).add(self.EPS).sqrt()
        tp = F.softplus(self.tau_p) * rms
        tn = F.softplus(self.tau_n) * rms
        s_pos = F.relu(r - tp)
        s_neg = -F.relu(-r - tn)
        return torch.stack([s_pos, s_neg, r - s_pos - s_neg], dim=1)

    def decompose_rich(self, x: torch.Tensor):
        """x: (B, 1, L) -> imfs (B, K, L), masks (B, K, F), aux dict.

        `aux` carries whatever the flags asked for, all optional:
          'env'    (B, 2K, L)  log1p envelope and instantaneous frequency
          'z'      (B, K, L)   the complex analytic signal, for callers that
                               have to mix bands before taking a magnitude
          'gated'  (B, K, L)   the magnitude-gated modes, when the gate is
                               additive and the lossless modes are kept
        """
        L = x.shape[-1]
        Xf = torch.fft.rfft(x.squeeze(1), n=L, dim=-1)              # (B, F)
        centres, bw = self.bands(x)
        if self.vmd_iters:
            centres = self.vmd_centres(Xf, centres.expand(Xf.shape[0], -1),
                                       bw.expand(Xf.shape[0], -1))
            bw = bw.expand_as(centres)
        m = self._masks_from(centres, bw)                           # (B, K, F)
        W = Xf.unsqueeze(1) * m                                     # (B, K, F)
        aux = {}
        if self.phase:
            f = (self.freqs / self.freqs[-1].clamp(min=1e-8)).view(1, 1, -1)
            ph = math.pi * self.theta.view(1, -1, 1) * f ** 2
            W = W * torch.polar(torch.ones_like(ph), ph)
        if self.gate:
            Wg = W * self.gain(W)
            if self.gate_additive:
                aux["gated"] = torch.fft.irfft(Wg, n=L, dim=-1)
            else:
                W = Wg
        if not self.envelope:
            imfs = torch.fft.irfft(W, n=L, dim=-1)
            if self.mca:
                aux["mca"] = self.morph(imfs)
            return imfs, m, aux
        z = self.analytic(W, L)                                     # (B, K, L)
        aux["z"] = z
        aux["env"] = self.env_feats(z)
        if self.mca:
            aux["mca"] = self.morph(z.real)
        return z.real, m, aux

    def forward(self, x: torch.Tensor):
        """x: (B, 1, L) -> imfs (B, K, L) with sum(imfs) == x, masks (B, K, F)."""
        imfs, m, _ = self.decompose_rich(x)
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
