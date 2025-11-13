# multimode_nvmd.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, padding=0)
        self.act = act

    def forward(self, x):
        pad_left = self.k - 1
        x = F.pad(x, (pad_left, 0))
        return self.act(self.conv(x))


class ResCausal(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1):
        super().__init__()
        self.conv1 = CausalConv1D(in_ch, out_ch, k, s)
        self.conv2 = CausalConv1D(out_ch, out_ch, k, 1, act=nn.Identity())
        self.skip = nn.Conv1d(in_ch, out_ch, 1, s) if (s != 1 or in_ch != out_ch) else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class MultiModeNVMD(nn.Module):
    def __init__(self, K=13, base=64):
        super().__init__()
        # Encoder ↓
        self.enc1 = ResCausal(1,     base,   k=7, s=1)
        self.enc2 = ResCausal(base,  base*2, k=7, s=2)
        self.enc3 = ResCausal(base*2,base*4, k=5, s=2)
        self.enc4 = ResCausal(base*4,base*8, k=5, s=2)

        # Bottleneck
        self.bott = nn.Sequential(
            ResCausal(base*8, base*8, k=3, s=1),
            ResCausal(base*8, base*8, k=3, s=1),
        )

        # Decoder ↑
        self.up1 = nn.ConvTranspose1d(base*8, base*4, 4, 2, 1)
        self.dec1 = ResCausal(base*4, base*4, k=5, s=1)

        self.up2 = nn.ConvTranspose1d(base*4, base*2, 4, 2, 1)
        self.dec2 = ResCausal(base*2, base*2, k=5, s=1)

        self.up3 = nn.ConvTranspose1d(base*2, base,   4, 2, 1)
        self.dec3 = ResCausal(base,   base,   k=7, s=1)

        # Mode heads (K = #IMFs)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base, base, 3, padding=1, groups=base),
                nn.GELU(),
                nn.Conv1d(base, 1, 1)
            ) for _ in range(K)
        ])

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        h = self.bott(e4)

        # Decoder + skip connections
        d1 = self.dec1(self.up1(h) + e3)
        d2 = self.dec2(self.up2(d1) + e2)
        d3 = self.dec3(self.up3(d2) + e1)

        # All IMF heads
        imfs = torch.cat([head(d3) for head in self.heads], dim=1)  # (B,K,L)

        # RRP reconstruction (sum of IMFs)
        recon = imfs.sum(dim=1, keepdim=True)

        return imfs, recon
