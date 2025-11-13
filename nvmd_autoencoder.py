# nvmd_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock1D(nn.Module):
    """
    Single causal conv (+optional BN) + activation.
    For s=1 it preserves length; for s=2 it downsamples by 2.
    """
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1, use_bn=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.k = k
        self.s = s
        self.d = d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=0, dilation=d, bias=not use_bn)
        self.bn   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act  = act

    def forward(self, x):
        # Left-only pad for causality
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0))
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class ResCausalBlock1D(nn.Module):
    """
    Residual block: causal conv → norm → act → causal conv → norm → add → act
    Keeps time length (stride=1) by causal left padding each conv.
    """
    def __init__(self, ch, k=7, d=1, use_bn=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.k = k
        self.d = d
        bias = not use_bn
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, stride=1, padding=0, dilation=d, bias=bias)
        self.bn1   = nn.BatchNorm1d(ch) if use_bn else nn.Identity()
        self.act   = act
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, stride=1, padding=0, dilation=d, bias=bias)
        self.bn2   = nn.BatchNorm1d(ch) if use_bn else nn.Identity()

    def forward(self, x):
        pad = (self.k - 1) * self.d
        y = F.pad(x, (pad, 0))
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y = F.pad(y, (pad, 0))
        y = self.conv2(y)
        y = self.bn2(y)

        y = y + x
        return self.act(y)


# ----------------------------
# Encoder-Decoder with residual stacks + U-Net skips
# Predicts K IMFs (channels = K)
# ----------------------------
class NVMD_Autoencoder(nn.Module):
    def __init__(self, in_ch=1, base=64, K=3, signal_len=1024, use_bn=True):
        """
        in_ch:     input channels (1 for RRP)
        base:      base channel width
        K:         number of IMFs predicted
        signal_len: for comments/debug only
        """
        super().__init__()

        # Encoder (downsample at s=2 for enc2, enc3, enc4)
        self.enc1 = nn.Sequential(
            CausalConvBlock1D(in_ch, base, k=7, s=1, d=1, use_bn=use_bn),
            ResCausalBlock1D(base, k=7, d=1, use_bn=use_bn),
        )  # -> (B, base, L)

        self.enc2 = nn.Sequential(
            CausalConvBlock1D(base, base*2, k=7, s=2, d=1, use_bn=use_bn),
            ResCausalBlock1D(base*2, k=7, d=1, use_bn=use_bn),
        )  # -> (B, 2*base, L/2)

        self.enc3 = nn.Sequential(
            CausalConvBlock1D(base*2, base*4, k=5, s=2, d=1, use_bn=use_bn),
            ResCausalBlock1D(base*4, k=5, d=1, use_bn=use_bn),
        )  # -> (B, 4*base, L/4)

        self.enc4 = nn.Sequential(
            CausalConvBlock1D(base*4, base*8, k=5, s=2, d=1, use_bn=use_bn),
            ResCausalBlock1D(base*8, k=3, d=1, use_bn=use_bn),
        )  # -> (B, 8*base, L/8)  (bottleneck in/out chans)

        # Bottleneck: small multi-dilation residual stack for more context
        self.bott = nn.Sequential(
            ResCausalBlock1D(base*8, k=3, d=1, use_bn=use_bn),
            ResCausalBlock1D(base*8, k=3, d=2, use_bn=use_bn),
            ResCausalBlock1D(base*8, k=3, d=4, use_bn=use_bn),
        )

        # Decoder (upsample x2 each stage) + residual refine, with U-Net skips
        self.up1  = nn.ConvTranspose1d(base*8, base*4, kernel_size=4, stride=2, padding=1)  # L/4
        self.dec1 = nn.Sequential(
            CausalConvBlock1D(base*4, base*4, k=5, s=1, d=1, use_bn=use_bn),
            ResCausalBlock1D(base*4, k=5, d=1, use_bn=use_bn),
        )

        self.up2  = nn.ConvTranspose1d(base*4, base*2, kernel_size=4, stride=2, padding=1)  # L/2
        self.dec2 = nn.Sequential(
            CausalConvBlock1D(base*2, base*2, k=5, s=1, d=1, use_bn=use_bn),
            ResCausalBlock1D(base*2, k=5, d=1, use_bn=use_bn),
        )

        self.up3  = nn.ConvTranspose1d(base*2, base, kernel_size=4, stride=2, padding=1)    # L
        self.dec3 = nn.Sequential(
            CausalConvBlock1D(base, base, k=7, s=1, d=1, use_bn=use_bn),
            ResCausalBlock1D(base, k=7, d=1, use_bn=use_bn),
        )

        # Shared projection + per-mode heads (your original design)
        self.proj_shared = nn.Conv1d(base, base, kernel_size=1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base, base, kernel_size=3, padding=1, groups=base),  # depthwise refine
                nn.GELU(),
                nn.Conv1d(base, 1, kernel_size=1)  # per-mode pointwise
            ) for _ in range(K)
        ])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)     # (B, base,   L)
        e2 = self.enc2(e1)    # (B, 2*base, L/2)
        e3 = self.enc3(e2)    # (B, 4*base, L/4)
        e4 = self.enc4(e3)    # (B, 8*base, L/8)

        # Bottleneck
        h  = self.bott(e4)    # (B, 8*base, L/8)

        # Decoder with U-Net skips (add, channels match by construction)
        d1 = self.up1(h)                  # (B, 4*base, L/4)
        d1 = self.dec1(d1 + e3)

        d2 = self.up2(d1)                 # (B, 2*base, L/2)
        d2 = self.dec2(d2 + e2)

        d3 = self.up3(d2)                 # (B, base, L)
        d3 = self.dec3(d3 + e1)

        # Heads
        f  = self.proj_shared(d3)         # (B, base, L)
        ys = [head(f) for head in self.heads]  # list of (B,1,L)
        imfs = torch.cat(ys, dim=1)            # (B, K, L)
        return imfs


if __name__ == "__main__":
    # quick shape test
    B, L, K = 2, 1024, 13
    x = torch.randn(B, 1, L)
    model = NVMD_Autoencoder(in_ch=1, base=64, K=K, signal_len=L)
    y = model(x)
    print("Input :", x.shape)   # (B, 1, L)
    print("Output:", y.shape)   # (B, K, L)
