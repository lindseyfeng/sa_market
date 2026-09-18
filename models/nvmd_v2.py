"""
Adaptive Spectral NVMD v2 — Input-dependent neural signal decomposition.

Why this can beat hard-coded VMD:
  1. Input-adaptive: frequency masks are conditioned on signal content,
     so different windows get different decompositions.
  2. End-to-end: decomposition is optimised for forecasting quality,
     not a mathematical bandwidth criterion that ignores the downstream task.
  3. Causal: no future data leakage by construction (causal convolutions).
  4. Hard reconstruction: modes always sum exactly to the original signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding."""

    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, groups=1):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=0, dilation=dilation, groups=groups, bias=False,
        )
        num_groups = 1
        for g in (8, 4, 2):
            if out_ch % g == 0:
                num_groups = g
                break
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = F.pad(x, (self.pad_len, 0))
        return self.act(self.norm(self.conv(x)))


class SignalEncoder(nn.Module):
    """Multi-scale causal CNN that compresses a 1-D signal into a feature
    vector.  Kernel sizes decrease so early layers see broad structure and
    later layers capture fine detail."""

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
        """x: (B, 1, L) → (B, d_model)"""
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Core decomposer
# ---------------------------------------------------------------------------

class AdaptiveSpectralNVMD(nn.Module):
    """Decomposes a 1-channel signal into K modes via learned,
    *input-adaptive* spectral masks.

    Architecture
    ------------
    1. SignalEncoder → compact feature vector for the window.
    2. Hyper-network maps features → K frequency-mask logits (per sample).
    3. Static base_logits act as a learnable prior (like fixed VMD) which
       the adaptive term perturbs.
    4. Softmax over the K dimension → masks partition the spectrum.
    5. Multiply complex FFT by masks → irfft → K spectral modes.
    6. Depthwise residual refiner for time-domain corrections.
    7. Hard constraint: last mode = original − sum(first K−1 refined modes),
       guaranteeing perfect reconstruction.
    """

    def __init__(self, K: int = 13, signal_len: int = 96, d_model: int = 128):
        super().__init__()
        self.K = K
        self.L = signal_len
        self.F = signal_len // 2 + 1

        self.encoder = SignalEncoder(d_model)

        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, K * self.F),
        )

        self.base_logits = nn.Parameter(torch.randn(1, K, self.F) * 0.02)
        self.adaptive_scale = nn.Parameter(torch.tensor(0.1))
        self.log_temperature = nn.Parameter(torch.zeros(1))

        self.refiner = nn.Sequential(
            nn.Conv1d(K, K, kernel_size=7, padding=3, groups=K, bias=False),
            nn.GELU(),
            nn.Conv1d(K, K, kernel_size=5, padding=2, groups=K, bias=True),
        )
        for m in self.refiner.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def get_masks(self, x: torch.Tensor):
        B = x.size(0)
        feat = self.encoder(x)                                      # (B, d_model)
        adaptive = self.mask_head(feat).view(B, self.K, self.F)     # (B, K, F)

        scale = self.adaptive_scale.clamp(min=0.01)
        logits = self.base_logits + scale * adaptive                # (B, K, F)

        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        masks = F.softmax(logits / temp, dim=1)                     # (B, K, F)
        return masks

    def forward(self, x: torch.Tensor):
        """
        Args:   x: (B, 1, L)
        Returns:
            imfs:  (B, K, L) — modes whose sum ≡ x
            masks: (B, K, F) — frequency masks (for regularisation)
        """
        B, C, L = x.shape
        masks = self.get_masks(x)                                    # (B, K, F)

        Xf = torch.fft.rfft(x.squeeze(1), n=L, dim=-1)              # (B, F) complex
        Xf_modes = Xf.unsqueeze(1) * masks                          # (B, K, F)
        imfs_spectral = torch.fft.irfft(Xf_modes, n=L, dim=-1)      # (B, K, L)

        imfs_refined = imfs_spectral + self.refiner(imfs_spectral)

        main_sum = imfs_refined[:, :-1, :].sum(dim=1, keepdim=True)  # (B,1,L)
        residual_mode = x - main_sum                                  # (B,1,L)
        imfs = torch.cat([imfs_refined[:, :-1, :], residual_mode], dim=1)

        return imfs, masks

    # ------------------------------------------------------------------
    # Self-supervised regularisers (no VMD targets needed)
    # ------------------------------------------------------------------
    def smoothness_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Penalise sharp jumps in frequency masks."""
        diff = masks[:, :, 1:] - masks[:, :, :-1]
        return (diff ** 2).mean()

    def orthogonality_loss(self, imfs: torch.Tensor) -> torch.Tensor:
        """Off-diagonal Gram matrix → 0 encourages decorrelated modes."""
        B, K, L = imfs.shape
        centered = imfs - imfs.mean(dim=-1, keepdim=True)
        norms = centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = centered / norms
        gram = torch.bmm(normed, normed.transpose(1, 2))
        eye = torch.eye(K, device=gram.device).unsqueeze(0)
        return ((gram - eye) ** 2).mean()

    def anti_collapse_loss(self, imfs: torch.Tensor) -> torch.Tensor:
        """Prevent any mode from carrying near-zero energy."""
        energy = (imfs ** 2).mean(dim=-1)    # (B, K)
        return (-torch.log(energy + 1e-8)).mean()

    @torch.no_grad()
    def decompose(self, x: torch.Tensor):
        """Convenience wrapper for inference."""
        return self.forward(x)


# ---------------------------------------------------------------------------
# Lightweight forecaster for training the decomposer (fast on CPU)
# ---------------------------------------------------------------------------

class NVMDForecaster(nn.Module):
    """Single LSTM over all K mode channels jointly.

    Much faster than K separate MRC_BiLSTMs (~138K params vs ~22M).
    The decomposer only needs a reasonable forecasting gradient to learn
    good modes; the heavy per-mode comparison happens in benchmark.py.
    """

    def __init__(
        self,
        K: int = 13,
        signal_len: int = 96,
        d_model: int = 128,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.K = K
        self.decomposer = AdaptiveSpectralNVMD(
            K=K, signal_len=signal_len, d_model=d_model,
        )

        self.lstm = nn.LSTM(
            K, lstm_hidden, lstm_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:   x: (B, 1, L)
        Returns:
            imfs:    (B, K, L)
            masks:   (B, K, F)
            y_modes: None       (not used in light forecaster)
            y_total: (B, 1)     forecast
        """
        imfs, masks = self.decomposer(x)                       # (B,K,L)
        h, _ = self.lstm(imfs.permute(0, 2, 1))                # (B,L,2H)
        y_total = self.head(h[:, -1])                           # (B,1)
        return imfs, masks, None, y_total


# ---------------------------------------------------------------------------
# Quick shape test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, L, K = 4, 96, 13
    x = torch.randn(B, 1, L)

    decomposer = AdaptiveSpectralNVMD(K=K, signal_len=L, d_model=128)
    imfs, masks = decomposer(x)
    recon = imfs.sum(dim=1, keepdim=True)

    print("Input       :", x.shape)
    print("IMFs        :", imfs.shape)
    print("Masks       :", masks.shape)
    print("Recon error :", (recon - x).abs().max().item())

    print("\nRegularisation losses:")
    print("  smoothness   :", decomposer.smoothness_loss(masks).item())
    print("  orthogonality:", decomposer.orthogonality_loss(imfs).item())
    print("  anti-collapse:", decomposer.anti_collapse_loss(imfs).item())
