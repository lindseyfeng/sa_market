#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class VMD13IMFDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        rrp_col: str = "RRP",
        K: int = 13,
    ):
        super().__init__()
        self.L = seq_len
        self.rrp_col = rrp_col
        self.K = K

        if rrp_col not in df.columns:
            raise ValueError(f"RRP column '{rrp_col}' not in dataframe")

        # Build mode column list: Mode_1..Mode_12, Residual
        mode_cols = [f"Mode_{i}" for i in range(1, K)] + ["Residual"]
        for c in mode_cols:
            if c not in df.columns:
                raise ValueError(f"Missing mode column '{c}' in dataframe")
        self.mode_cols = mode_cols

        rrp = df[rrp_col].to_numpy(dtype=np.float32)              # (T,)
        imfs = df[mode_cols].to_numpy(dtype=np.float32)           # (T, K)

        self.rrp = torch.from_numpy(rrp)                          # (T,)
        # convert to (K, T)
        self.imfs = torch.from_numpy(imfs).transpose(0, 1)        # (K, T)

        T = self.rrp.shape[0]
        self.N = max(0, T - self.L)   # last window ends at T-1

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L

        # raw RRP window (1, L)
        x_raw = self.rrp[i:i+L].unsqueeze(0)              # (1, L)

        # IMFs window (K, L)
        imfs_true = self.imfs[:, i:i+L]                   # (K, L)

        return x_raw, imfs_true



class SpectralDecomposer(nn.Module):
    def __init__(self, K: int, signal_len: int):
        super().__init__()
        self.K = K
        self.L = signal_len
        self.F = signal_len // 2 + 1

        # logits for masks: (K, F)
        self.logits = nn.Parameter(torch.zeros(K, self.F))

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert L == self.L, f"Expected signal_len={self.L}, got {L}"

        # rFFT along time dimension
        Xf = torch.fft.rfft(x, dim=-1)    # (B, 1, F), complex

        # masks: (K, F), softmax over K
        masks = F.softmax(self.logits, dim=0)  # (K, F)
        masks_exp = masks.unsqueeze(0).expand(B, -1, -1)   # (B, K, F)

        # broadcast Xf to (B, K, F)
        Xf_exp = Xf.expand(-1, self.K, -1)                  # (B, K, F)
        Xf_modes = Xf_exp * masks_exp.to(Xf.dtype)          # (B, K, F), complex

        # inverse FFT
        imfs_lin = torch.fft.irfft(Xf_modes, n=self.L, dim=-1)  # (B, K, L), real

        # reconstruction
        recon_lin = imfs_lin.sum(dim=1, keepdim=True)           # (B, 1, L)

        return imfs_lin, recon_lin

    def spectral_smoothness_loss(self):
        masks = F.softmax(self.logits, dim=0)  # (K, F)
        diff = masks[:, 1:] - masks[:, :-1]    # (K, F-1)
        return (diff ** 2).mean()
        
    def orthogonality_loss(self):
        masks = F.softmax(self.logits, dim=0)  # (K, F)
        K, Ffreq = masks.shape
        loss = 0.0
        cnt = 0
        for i in range(K):
            mi = masks[i]
            for j in range(i+1, K):
                mj = masks[j]
                num = (mi * mj).sum()
                den = mi.norm() * mj.norm() + 1e-8
                loss = loss + (num / den)
                cnt += 1
        if cnt > 0:
            loss = loss / cnt
        return loss



class ModewiseRefiner(nn.Module):
    """
    Small depthwise CNN applied per mode:

      Input:  imfs_lin (B, K, L)
      Output: imfs_refined (B, K, L)

    Uses depthwise Conv1d with groups=K so each mode has its own filter.
    """
    def __init__(self, K: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=K,
            out_channels=K,
            kernel_size=kernel_size,
            padding=padding,
            groups=K,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            in_channels=K,
            out_channels=K,
            kernel_size=kernel_size,
            padding=padding,
            groups=K,
            bias=True,
        )
        self.act = nn.GELU()

        # Init small weights so we start near identity
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, imfs_lin: torch.Tensor) -> torch.Tensor:
        """
        imfs_lin: (B, K, L)
        returns:  (B, K, L)
        """
        x = imfs_lin
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y   # residual refinement


# ============================================================
#                Hybrid Spectral + CNN Decomposer
# ============================================================

class HybridSpectralNVMD(nn.Module):
    """
    Full hybrid decomposer:

      - SpectralDecomposer produces linear IMFs (frequency-partitioned).
      - ModewiseRefiner refines each mode with small depthwise CNN.
      - Sum over refined IMFs reconstructs RRP.

    Forward:
      x_raw:        (B,1,L)
      imfs_refined: (B,K,L)
      recon_refined:(B,1,L)
      imfs_lin:     (B,K,L)
      recon_lin:    (B,1,L)
    """
    def __init__(self, K: int = 13, signal_len: int = 64):
        super().__init__()
        self.K = K
        self.L = signal_len

        self.spectral = SpectralDecomposer(K=K, signal_len=signal_len)
        self.refiner  = ModewiseRefiner(K=K)

    def forward(self, x_raw: torch.Tensor):
        imfs_lin, recon_lin = self.spectral(x_raw)               # (B,K,L), (B,1,L)
        imfs_refined = self.refiner(imfs_lin)                    # (B,K,L)
        recon_refined = imfs_refined.sum(dim=1, keepdim=True)    # (B,1,L)
        return imfs_refined, recon_refined, imfs_lin, recon_lin

def train_epoch(
    model: HybridSpectralNVMD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    w_imf: float,
    w_rrp: float,
    w_smooth: float,
    w_ortho: float,
    clip_grad: float | None = 10.0,
):
    """
    Returns:
      avg_imf_rel_rmse: scalar, average per-window relative RMSE over IMFs
      avg_imf_mae:      scalar, average element-wise MAE over IMFs (for logging)
      avg_rrp_mae:      scalar, average L1 recon error on RRP window
    """
    model.train()
    total_imf_rel = 0.0   # average relative RMSE over windows
    total_imf_mae = 0.0   # for logging only
    total_rrp = 0.0
    n = 0

    eps = 1e-8

    for x_raw, imfs_true in loader:
        x_raw = x_raw.to(device)          # (B,1,L)
        imfs_true = imfs_true.to(device)  # (B,K,L)

        optimizer.zero_grad(set_to_none=True)

        imfs_ref, recon_ref, imfs_lin, recon_lin = model(x_raw)  # (B,K,L), (B,1,L), ...

        # ------------------------------------------------------
        # IMF reconstruction: per-window relative RMSE
        # ------------------------------------------------------
        delta = imfs_ref - imfs_true           # (B,K,L)

        num = (delta ** 2).sum(dim=(1, 2))     # (B,)
        den = (imfs_true ** 2).sum(dim=(1, 2)) + eps  # (B,)
        rel_rmse = torch.sqrt(num / den)       # (B,)

        loss_imf = rel_rmse.mean()             # scalar

        # simple element-wise MAE (for logging)
        imf_mae = delta.abs().mean()

        # ------------------------------------------------------
        # RRP reconstruction loss
        # ------------------------------------------------------
        loss_rrp = F.l1_loss(recon_ref, x_raw)

        # spectral regularizers
        loss_smooth = model.spectral.spectral_smoothness_loss()
        loss_ortho  = model.spectral.orthogonality_loss()

        loss = (
            w_imf    * loss_imf
            + w_rrp  * loss_rrp
            + w_smooth * loss_smooth
            + w_ortho  * loss_ortho
        )

        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        bs = x_raw.size(0)
        n += bs
        total_imf_rel += rel_rmse.mean().item() * bs
        total_imf_mae += imf_mae.item() * bs
        total_rrp     += loss_rrp.item() * bs

    denom = max(n, 1)
    return (
        total_imf_rel / denom,   # avg relative RMSE
        total_imf_mae / denom,   # avg MAE
        total_rrp     / denom,   # avg RRP MAE
    )


def eval_epoch(model: HybridSpectralNVMD, loader: DataLoader, device: str):
    model.eval()
    total_imf_rel = 0.0
    total_imf_mae = 0.0
    total_rrp = 0.0
    n = 0
    eps = 1e-8

    with torch.no_grad():
        for x_raw, imfs_true in loader:
            x_raw = x_raw.to(device)
            imfs_true = imfs_true.to(device)

            imfs_ref, recon_ref, imfs_lin, recon_lin = model(x_raw)

            delta = imfs_ref - imfs_true  # (B,K,L)

            num = (delta ** 2).sum(dim=(1, 2))            # (B,)
            den = (imfs_true ** 2).sum(dim=(1, 2)) + eps  # (B,)
            rel_rmse = torch.sqrt(num / den)              # (B,)
            loss_imf = rel_rmse.mean()

            imf_mae = delta.abs().mean()
            loss_rrp = F.l1_loss(recon_ref, x_raw)

            bs = x_raw.size(0)
            n += bs
            total_imf_rel += rel_rmse.mean().item() * bs
            total_imf_mae += imf_mae.item() * bs
            total_rrp     += loss_rrp.item() * bs

    denom = max(n, 1)
    return (
        total_imf_rel / denom,
        total_imf_mae / denom,
        total_rrp     / denom,
    )


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train-csv", type=str,
                    default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str,
                    default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--K", type=int, default=13)

    # training
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad", type=float, default=10.0)

    # loss weights
    ap.add_argument("--w-imf", type=float, default=1.0,
                    help="weight for IMF window reconstruction loss")
    ap.add_argument("--w-rrp", type=float, default=0.1,
                    help="weight for RRP window reconstruction loss")
    ap.add_argument("--w-smooth", type=float, default=0.1,
                    help="weight for spectral smoothness of masks")
    ap.add_argument("--w-ortho", type=float, default=0.1,
                    help="weight for spectral orthogonality of masks")

    # I/O
    ap.add_argument("--out", type=str, default="./hybrid_spectral_nvmd.pt")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load dataframes
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    # datasets / loaders
    tr_ds = VMD13IMFDataset(df_tr, seq_len=args.seq_len,
                            rrp_col=args.rrp_col, K=args.K)
    va_ds = VMD13IMFDataset(df_va, seq_len=args.seq_len,
                            rrp_col=args.rrp_col, K=args.K)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    # model
    model = HybridSpectralNVMD(K=args.K, signal_len=args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # track best validation **relative RMSE** on IMFs
    best_val_imf_rel = float("inf")

    for ep in range(1, args.epochs + 1):
        # train_epoch returns: (imf_rel_rmse, imf_mae, rrp_mae)
        tr_imf_rel, tr_imf_mae, tr_rrp_mae = train_epoch(
            model,
            tr_dl,
            opt,
            device,
            w_imf=args.w_imf,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            clip_grad=args.clip_grad,
        )
        va_imf_rel, va_imf_mae, va_rrp_mae = eval_epoch(model, va_dl, device)

        print(
            f"[Epoch {ep:03d}] "
            f"train IMF relRMSE={tr_imf_rel:.4f} MAE={tr_imf_mae:.4f} | "
            f"train RRP MAE={tr_rrp_mae:.4f} || "
            f"val IMF relRMSE={va_imf_rel:.4f} MAE={va_imf_mae:.4f} | "
            f"val RRP MAE={va_rrp_mae:.4f}"
        )

        # save best on IMF *relative RMSE*
        if va_imf_rel < best_val_imf_rel:
            best_val_imf_rel = va_imf_rel
            torch.save(model.state_dict(), args.out)
            print(
                f"  → Saved new best checkpoint with val IMF relRMSE={best_val_imf_rel:.4f} "
                f"to {args.out}"
            )


if __name__ == "__main__":
    main()

