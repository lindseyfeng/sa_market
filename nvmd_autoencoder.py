# nvmd_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1, use_bn=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.k = k
        self.s = s
        self.d = d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=0, dilation=d, bias=not use_bn)
        self.bn   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act  = act

    def forward(self, x):
        # Left-only pad: (left, right)
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0))                  # (B, C, L + pad_left)
        x = self.conv(x)                             # (B, out_ch, L)  (stride=1)
        x = self.bn(x)
        return self.act(x)


# ----------------------------
# Encoder-Decoder (7 layers total: 3 down, bottleneck, 3 up)
# Predicts K IMFs (channels = K)
# ----------------------------
class NVMD_Autoencoder(nn.Module):
    def __init__(self, in_ch=1, base=64, K=3, signal_len=1024):
        """
        in_ch:     input channels (1 for ECG)
        base:      base channel width
        K:         number of IMFs predicted
        signal_len: used only to sanity-check shapes in comments
        """
        super().__init__()
        self.enc1 = CausalConvBlock1D(in_ch, base, k=7, s=1)           # L
        self.enc2 = CausalConvBlock1D(base, base*2, k=7, s=2)          # L/2
        self.enc3 = CausalConvBlock1D(base*2, base*4, k=5, s=2)        # L/4
        self.enc4 = CausalConvBlock1D(base*4, base*8, k=5, s=2)        # L/8  (bottleneck)

        # Bottleneck refinement
        self.bott = nn.Sequential(
            CausalConvBlock1D(base*8, base*8, k=3, s=1),
            CausalConvBlock1D(base*8, base*8, k=3, s=1),
        )

        # Decoder: upsample x2 each stage (use nearest + conv)
        self.up1 = nn.ConvTranspose1d(base*8, base*4, kernel_size=4, stride=2, padding=1)  # L/4
        self.dec1 = CausalConvBlock1D(base*4, base*4, k=5, s=1)

        self.up2 = nn.ConvTranspose1d(base*4, base*2, kernel_size=4, stride=2, padding=1)  # L/2
        self.dec2 = CausalConvBlock1D(base*2, base*2, k=5, s=1)

        self.up3 = nn.ConvTranspose1d(base*2, base, kernel_size=4, stride=2, padding=1)    # L
        self.dec3 = CausalConvBlock1D(base, base, k=7, s=1)

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
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)     # (B, base, L)
        # print(e1.shape)

        e2 = self.enc2(e1)    # (B, 2*base, L/2)
        # print(e2.shape)
        e3 = self.enc3(e2)    # (B, 4*base, L/4)
        # print(e3.shape)
        e4 = self.enc4(e3)    # (B, 8*base, L/8)
        # print(e4.shape)

        h  = self.bott(e4)
        # print(h.shape)

        d1 = self.up1(h)      # (B, 4*base, L/4)
        # print(d1.shape)
        d1 = self.dec1(d1)
        # print(d1.shape)

        d2 = self.up2(d1)     # (B, 2*base, L/2)
        # print(d2.shape)
        d2 = self.dec2(d2)
        # print(d2.shape)

        d3 = self.up3(d2)     # (B, base, L)
        # print(d3.shape)
        d3 = self.dec3(d3)
        # print(d3.shape)
        
        f  = self.proj_shared(d3)  # (B, base, L)
        ys = [h(f) for h in self.heads]    # list of (B,1,L)
        imfs = torch.cat(ys, dim=1)        # (B,K,L)
        return imfs




if __name__ == "__main__":
    # ---- Config
    B, L = 2, 8
    K = 13
    in_ch = 1      # input channels (RRP is scalar)
    base = 64      # base feature channels

    # ---- Dummy input (batch of random sequences)
    x = torch.randn(B, in_ch, L)
    print(x)

    # ---- Model
    model = NVMD_Autoencoder(in_ch=in_ch, base=base, K=K, signal_len=L)

    # ---- Forward pass
    imfs = model(x)
    print(imfs)

    print("\n===== Final outputs =====")
    print("Input shape:     ", x.shape)      # (B, 1, L)
    print("IMFs shape:      ", imfs.shape)   # (B, K, L)
