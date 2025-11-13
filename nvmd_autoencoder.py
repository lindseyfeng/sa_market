# nvmd_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock1D(nn.Module):
    """
    Your original causal conv block: left padding → Conv1d → (BN) → act.
    s=1 preserves length; s=2 downsamples by 2 (causal).
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
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0))
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class ResidualCausal(nn.Module):
    """
    Residual block *composed of your CausalConvBlock1D*.
    main:  Causal(in→out, stride=s) → Causal(out→out, stride=1)
    skip:  identity if (in==out and s==1) else 1x1 conv with stride=s
    out:   act(main + skip)
    """
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1, use_bn=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.main1 = CausalConvBlock1D(in_ch, out_ch, k=k, s=s, d=d, use_bn=use_bn, act=act)
        self.main2 = CausalConvBlock1D(out_ch, out_ch, k=k, s=1, d=d, use_bn=use_bn, act=nn.Identity())
        # skip path to match shape if needed
        if (in_ch != out_ch) or (s != 1):
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=s, padding=0, bias=False)
        else:
            self.skip = nn.Identity()
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act = act

    def forward(self, x):
        y = self.main1(x)
        y = self.main2(y)
        s = self.skip(x)
        # BN after addition (common in simple ResNet variants)
        y = self.bn(y + s)
        return self.act(y)


class NVMD_Autoencoder(nn.Module):
    """
    Encoder-decoder with your causal convs, now wrapped in ResidualCausal blocks,
    plus U-Net skip additions. Per-mode heads remain the same as you wrote.
    """
    def __init__(self, in_ch=1, base=64, K=3, signal_len=1024, use_bn=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        # Encoder (downsample on enc2/3/4 via s=2)
        self.enc1 = ResidualCausal(in_ch,   base,   k=7, s=1, d=1, use_bn=use_bn, act=act)  # -> L
        self.enc2 = ResidualCausal(base,    base*2, k=7, s=2, d=1, use_bn=use_bn, act=act)  # -> L/2
        self.enc3 = ResidualCausal(base*2,  base*4, k=5, s=2, d=1, use_bn=use_bn, act=act)  # -> L/4
        self.enc4 = ResidualCausal(base*4,  base*8, k=5, s=2, d=1, use_bn=use_bn, act=act)  # -> L/8

        # Bottleneck: keep your style, just add residual flavor (no downsample)
        self.bott = nn.Sequential(
            ResidualCausal(base*8, base*8, k=3, s=1, d=1, use_bn=use_bn, act=act),
            ResidualCausal(base*8, base*8, k=3, s=1, d=1, use_bn=use_bn, act=act),
        )

        # Decoder (upsample x2 with transposed conv) + residual refine
        self.up1  = nn.ConvTranspose1d(base*8, base*4, kernel_size=4, stride=2, padding=1)  # -> L/4
        self.dec1 = ResidualCausal(base*4, base*4, k=5, s=1, d=1, use_bn=use_bn, act=act)

        self.up2  = nn.ConvTranspose1d(base*4, base*2, kernel_size=4, stride=2, padding=1)  # -> L/2
        self.dec2 = ResidualCausal(base*2, base*2, k=5, s=1, d=1, use_bn=use_bn, act=act)

        self.up3  = nn.ConvTranspose1d(base*2, base,   kernel_size=4, stride=2, padding=1)  # -> L
        self.dec3 = ResidualCausal(base,   base,   k=7, s=1, d=1, use_bn=use_bn, act=act)

        # Your per-mode heads (kept intact)
        self.proj_shared = nn.Conv1d(base, base, kernel_size=1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base, base, kernel_size=3, padding=1, groups=base),
                nn.GELU(),
                nn.Conv1d(base, 1, kernel_size=1)
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
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # (B, base, L)
        e2 = self.enc2(e1)     # (B, 2*base, L/2)
        e3 = self.enc3(e2)     # (B, 4*base, L/4)
        e4 = self.enc4(e3)     # (B, 8*base, L/8)

        # Bottleneck
        h  = self.bott(e4)     # (B, 8*base, L/8)

        # Decoder + U-Net additions (add — channels already matched)
        d1 = self.up1(h)                 # (B, 4*base, L/4)
        d1 = self.dec1(d1 + e3)

        d2 = self.up2(d1)                # (B, 2*base, L/2)
        d2 = self.dec2(d2 + e2)

        d3 = self.up3(d2)                # (B, base,   L)
        d3 = self.dec3(d3 + e1)

        # Mode heads
        f  = self.proj_shared(d3)        # (B, base, L)
        ys = [head(f) for head in self.heads]  # list of (B,1,L)
        imfs = torch.cat(ys, dim=1)            # (B, K, L)
        return imfs


if __name__ == "__main__":
    B, L, K = 2, 1024, 13
    x = torch.randn(B, 1, L)
    model = NVMD_Autoencoder(in_ch=1, base=64, K=K, signal_len=L)
    y = model(x)
    print("Input :", x.shape)   # (B, 1, 1024)
    print("Output:", y.shape)   # (B, 13, 1024)
