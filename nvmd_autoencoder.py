# multimode_nvmd.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModeNVMD(nn.Module):
    """
    Encoder–decoder with K per-mode heads.
    Input : (B,1,L)  RRP window
    Output: imfs  (B,K,L)  reconstructed IMFs
            recon (B,1,L)  sum over IMFs (approx RRP)
    """
    def __init__(self, K=13, base=64):
        super().__init__()
        self.K = K
        # Encoder
        self.enc1 = ResCausal(1, base)
        self.enc2 = ResCausal(base, base*2, s=2)
        self.enc3 = ResCausal(base*2, base*4, k=5, s=2)
        self.enc4 = ResCausal(base*4, base*8, k=5, s=2)

        # Bottleneck
        self.bott = nn.Sequential(
            ResCausal(base*8, base*8, k=3),
            ResCausal(base*8, base*8, k=3),
        )

        # Decoder
        self.up1 = nn.ConvTranspose1d(base*8, base*4, 4, 2, 1)
        self.dec1 = ResCausal(base*4, base*4, k=5)

        self.up2 = nn.ConvTranspose1d(base*4, base*2, 4, 2, 1)
        self.dec2 = ResCausal(base*2, base*2, k=5)

        self.up3 = nn.ConvTranspose1d(base*2, base, 4, 2, 1)
        self.dec3 = ResCausal(base, base, k=7)

        # IMF heads (K modes)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base, base, 3, padding=1, groups=base),
                nn.GELU(),
                nn.Conv1d(base, 1, 1),
            )
            for _ in range(K)
        ])

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        h = self.bott(e4)

        # Decoder
        d1 = self.dec1(self.up1(h) + e3)
        d2 = self.dec2(self.up2(d1) + e2)
        d3 = self.dec3(self.up3(d2) + e1)

        # All IMFs
        imfs = torch.cat([head(d3) for head in self.heads], dim=1)   # (B,K,L)

        recon = imfs.sum(dim=1, keepdim=True)                        # (B,1,L)

        return imfs, recon
