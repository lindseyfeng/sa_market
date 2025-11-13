#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


def default_imf_cols(df: pd.DataFrame):
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing IMF columns: {missing}")
    return cols


class VMD13Dataset(Dataset):
    def __init__(self, df, seq_len=64, rrp_col="RRP", imf_cols=None):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")

        if imf_cols is None:
            imf_cols = default_imf_cols(df)
        self.imf_cols = imf_cols

        rrp = df[rrp_col].to_numpy(dtype=np.float32)           # (T,)
        imfs = df[imf_cols].to_numpy(dtype=np.float32)         # (T, K)

        self.rrp = torch.from_numpy(rrp)                       # (T,)
        self.imfs = torch.from_numpy(imfs)                     # (T, K)

        T = len(rrp)
        self.N = T - seq_len - 1   # need i+L for next-step target

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L
        # RRP window + next
        x = self.rrp[i:i+L].unsqueeze(0)            # (1,L)
        y = self.rrp[i+L].unsqueeze(0)              # (1,)

        # IMF window + next
        imf_win  = self.imfs[i:i+L]                 # (L,K)
        imf_next = self.imfs[i+L]                   # (K,)

        # reshape IMFs to (K,L) to match model output (B,K,L)
        imf_win = imf_win.T                         # (K,L)

        return x, y, imf_win, imf_next              # DataLoader → (B,1,L), (B,1), (B,K,L), (B,K)


class CausalConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1,
                 act=nn.LeakyReLU(0.1, inplace=True)):
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
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, s)
            if (s != 1 or in_ch != out_ch) else nn.Identity()
        )
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
            ResCausal(base*8, base*8, k=3),
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

        # Sum = RRP reconstruction
        recon = imfs.sum(dim=1, keepdim=True)                        # (B,1,L)

        return imfs, recon


def decorrelation_loss(imfs):
    B, K, L = imfs.shape
    loss = 0.0
    for i in range(K):
        for j in range(i+1, K):
            ci = imfs[:, i] - imfs[:, i].mean(dim=1, keepdim=True)
            cj = imfs[:, j] - imfs[:, j].mean(dim=1, keepdim=True)
            corr = (ci * cj).mean()
            loss += corr.abs()
    return loss / (K * (K - 1) / 2)


def bandwidth_loss(imfs):
    freqs = torch.fft.rfft(imfs, dim=-1).abs()
    w = torch.linspace(0, 1, freqs.size(-1), device=imfs.device)
    return (freqs * w).mean()

def train_epoch(model, loader, opt, device, w1, w2, w3):
    """
    w1: weight for supervised losses (RRP + IMF)
    w2: weight for decorrelation loss
    w3: weight for bandwidth loss
    """
    model.train()
    total_rrp = 0.0
    n = 0

    for x, rrp_next, imf_win, imf_next in loader:
        x = x.to(device)                # (B,1,L)
        rrp_next = rrp_next.to(device)  # (B,1)
        imf_win = imf_win.to(device)    # (B,K,L)
        imf_next = imf_next.to(device)  # (B,K)

        opt.zero_grad()

        imfs_pred, recon = model(x)     # (B,K,L), (B,1,L)

        # RRP next-step prediction
        rrp_pred_next = recon[:, :, -1]           # (B,1)
        L_rrp = F.l1_loss(rrp_pred_next, rrp_next)

        # IMF window + next supervision
        L_imf_w = F.l1_loss(imfs_pred, imf_win)
        imfs_pred_next = imfs_pred[:, :, -1]      # (B,K)
        L_imf_n = F.l1_loss(imfs_pred_next, imf_next)

        # Regularizers
        L_corr = decorrelation_loss(imfs_pred)
        L_band = bandwidth_loss(imfs_pred)

        # Total loss
        supervised = L_rrp + L_imf_w + L_imf_n
        loss = w1 * supervised + w2 * L_corr + w3 * L_band

        loss.backward()
        opt.step()

        bs = x.size(0)
        n += bs
        total_rrp += L_rrp.item() * bs

    return total_rrp / max(n, 1)


def eval_epoch(model, loader, device):
    model.eval()
    total_rrp = 0.0
    total_imf_window = 0.0
    total_imf_next = 0.0
    n = 0

    with torch.no_grad():
        for x, rrp_next, imf_win, imf_next in loader:
            x = x.to(device)
            rrp_next = rrp_next.to(device)
            imf_win = imf_win.to(device)
            imf_next = imf_next.to(device)

            imfs_pred, recon = model(x)

            # IMF window loss
            L_imf_w = F.l1_loss(imfs_pred, imf_win)

            # IMF next-step loss
            imfs_pred_next = imfs_pred[:, :, -1]
            L_imf_n = F.l1_loss(imfs_pred_next, imf_next)

            # RRP next-step loss
            rrp_pred_next = recon[:, :, -1]
            L_rrp = F.l1_loss(rrp_pred_next, rrp_next)

            bs = x.size(0)
            n += bs
            total_rrp        += L_rrp.item() * bs
            total_imf_window += L_imf_w.item() * bs
            total_imf_next   += L_imf_n.item() * bs

    denom = max(n, 1)
    return (
        total_rrp / denom,
        total_imf_window / denom,
        total_imf_next / denom,
    )


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

    best = float("inf")

    for ep in range(1, args.epochs + 1):
        tr_rrp = train_epoch(model, tr_dl, opt, device, args.w1, args.w2, args.w3)
        rrp_mae, imf_window_mae, imf_next_mae = eval_epoch(model, te_dl, device)

        print(
            f"[Epoch {ep:03d}] "
            f"train RRP MAE={tr_rrp:.4f} | "
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
