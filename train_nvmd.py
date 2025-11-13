#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ===============================================================
#                DATASET (RRP + Full 13-IMF VMD)
# ===============================================================

class VMD13Dataset(Dataset):
    """
    Returns:
       x_raw:  (B,1,L)     raw RRP window
       rrp_next: (B,1)     raw next-step RRP
    """
    def __init__(self, df, seq_len=64, rrp_col="RRP"):
        super().__init__()
        self.L = seq_len
        rrp = df[rrp_col].to_numpy(dtype=np.float32)

        self.rrp = torch.tensor(rrp)
        T = len(rrp)
        self.N = T - seq_len - 1

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L
        x = self.rrp[i:i+L].unsqueeze(0)     # (1,L)
        y = self.rrp[i+L].unsqueeze(0)       # (1,)
        return x, y


# ===============================================================
#                     MULTIMODE NVMD MODEL
# ===============================================================

class CausalConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, padding=0)
        self.act = act

    def forward(self, x):
        pad = self.k - 1
        x = F.pad(x, (pad, 0))
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
        # Encoder
        self.enc1 = ResCausal(1, base)
        self.enc2 = ResCausal(base, base*2, s=2)
        self.enc3 = ResCausal(base*2, base*4, k=5, s=2)
        self.enc4 = ResCausal(base*4, base*8, k=5, s=2)

        # Bottleneck
        self.bott = nn.Sequential(
            ResCausal(base*8, base*8, k=3),
            ResCausal(base*8, base*8, k=3)
        )

        # Decoder
        self.up1 = nn.ConvTranspose1d(base*8, base*4, 4, 2, 1)
        self.dec1 = ResCausal(base*4, base*4, k=5)

        self.up2 = nn.ConvTranspose1d(base*4, base*2, 4, 2, 1)
        self.dec2 = ResCausal(base*2, base*2, k=5)

        self.up3 = nn.ConvTranspose1d(base*2, base, 4, 2, 1)
        self.dec3 = ResCausal(base, base, k=7)

        # IMF heads (K=13)
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

        # Decoder
        d1 = self.dec1(self.up1(h) + e3)
        d2 = self.dec2(self.up2(d1) + e2)
        d3 = self.dec3(self.up3(d2) + e1)

        # All IMFs
        imfs = torch.cat([head(d3) for head in self.heads], dim=1)   # (B,13,L)

        # Sum = RRP reconstruction
        recon = imfs.sum(dim=1, keepdim=True)                        # (B,1,L)

        return imfs, recon


# ===============================================================
#                 LOSSES (DECORR + BANDWIDTH)
# ===============================================================

def decorrelation_loss(imfs):
    B, K, L = imfs.shape
    loss = 0
    for i in range(K):
        for j in range(i+1, K):
            ci = imfs[:,i] - imfs[:,i].mean(dim=1, keepdim=True)
            cj = imfs[:,j] - imfs[:,j].mean(dim=1, keepdim=True)
            corr = (ci * cj).mean()
            loss += corr.abs()
    return loss / (K*(K-1)/2)


def bandwidth_loss(imfs):
    freqs = torch.fft.rfft(imfs, dim=-1).abs()
    w = torch.linspace(0,1,freqs.size(-1),device=imfs.device)
    return (freqs * w).mean()


# ===============================================================
#                     TRAIN / EVAL EPOCH
# ===============================================================

def train_epoch(model, loader, opt, device, w1, w2, w3):
    model.train()
    total = 0
    for x, rrp_next in loader:
        x = x.to(device)              # (B,1,L)
        rrp_next = rrp_next.to(device)

        opt.zero_grad()
        imfs, recon = model(x)

        # Take the last time step as prediction
        y_pred = recon[:, :, -1]      # (B,1)

        # Losses
        L_recon = F.l1_loss(y_pred, rrp_next)
        L_corr  = decorrelation_loss(imfs)
        L_band  = bandwidth_loss(imfs)

        loss = w1*L_recon + w2*L_corr + w3*L_band
        loss.backward()
        opt.step()

        total += L_recon.item() * x.size(0)

    return total / len(loader.dataset)


def eval_epoch(model, loader, device, df_imf_cols):
    model.eval()
    total_rrp = 0
    total_imf_window = 0
    total_imf_next = 0
    n = 0

    with torch.no_grad():
        for (x, rrp_next, imf_win, imf_next) in loader:
            x, rrp_next = x.to(device), rrp_next.to(device)
            imf_win = imf_win.to(device)       # (B, 13, L)
            imf_next = imf_next.to(device)     # (B,13)

            # Forward
            IMF_pred, recon = model(x)

            # 1. IMF window loss
            loss_imf_w = F.l1_loss(IMF_pred, imf_win)

            # 2. IMF next-step loss
            IMF_pred_next = IMF_pred[:,:, -1]   # (B,13)
            loss_imf_n = F.l1_loss(IMF_pred_next, imf_next)

            # 3. RRP (sum of IMFs)
            rrp_pred_next = recon[:,:, -1]      # (B,1)
            loss_rrp = F.l1_loss(rrp_pred_next, rrp_next)

            bs = x.size(0)
            n += bs
            total_imf_window += loss_imf_w.item() * bs
            total_imf_next   += loss_imf_n.item() * bs
            total_rrp        += loss_rrp.item() * bs

    return (
        total_rrp / n,
        total_imf_window / n,
        total_imf_next / n,
    )



# ===============================================================
#                           MAIN
# ===============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--test-csv",  default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=0.1)
    ap.add_argument("--w3", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="./nmvd13.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)

    tr_ds = VMD13Dataset(df_train, seq_len=args.seq_len)
    te_ds = VMD13Dataset(df_test,  seq_len=args.seq_len)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch)

    model = MultiModeNVMD(K=13, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 1e9

    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, tr_dl, opt, device, args.w1, args.w2, args.w3)
        # te is a tuple now
        rrp_mae, imf_window_mae, imf_next_mae = eval_epoch(model, te_dl, device)
        
        print(
            f"[Epoch {ep:03d}] "
            f"train RRP MAE={tr:.4f} | "
            f"test RRP MAE={rrp_mae:.4f} | "
            f"IMF-win MAE={imf_window_mae:.4f} | "
            f"IMF-next MAE={imf_next_mae:.4f}"
        )
        
        # Save checkpoint based on *RRP forecasting* only
        if rrp_mae < best:
            best = rrp_mae
            torch.save(model.state_dict(), args.out)
            print(f"  → saved best checkpoint (RRP MAE={best:.4f})")

if __name__ == "__main__":
    main()
