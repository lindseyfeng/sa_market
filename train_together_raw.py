#!/usr/bin/env python3
import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_nvmd import HybridSpectralNVMD
from train_transformer import ModeTime2DTransformerRRP  # or MultiModeTransformerRRP


# ============================================================
# Dataset: raw RRP → (x_raw, rrp_next)
# ============================================================

class RRPDecompDataset(Dataset):
    """
    For each window i..i+L-1:

      x_raw:   (1, L)   → raw RRP window
      rrp_next:(1,)     → raw RRP at time t+L

    We DO NOT use ground-truth IMFs here.
    """
    def __init__(self, df, seq_len=64, rrp_col="RRP"):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not found in dataframe columns.")

        rrp = df[rrp_col].to_numpy(dtype=np.float32)  # (T,)
        self.rrp = torch.from_numpy(rrp)              # (T,)

        T = self.rrp.shape[0]
        # Need t+L for target → max start index T-L-1
        self.N = max(0, T - self.L - 1)
        if self.N <= 0:
            raise ValueError(f"Not enough samples for seq_len={seq_len}, T={T}")

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        x_raw = self.rrp[i:i+L].unsqueeze(0)   # (1,L)
        rrp_next = self.rrp[i+L].unsqueeze(0)  # (1,)
        return x_raw, rrp_next


# ============================================================
# Training / Eval (end-to-end)
# ============================================================

def train_epoch(
    decomposer: HybridSpectralNVMD,
    predictor: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    w_pred: float,
    w_rrp: float,
    w_smooth: float,
    w_ortho: float,
    max_grad_norm: float = 10.0,
):
    """
    End-to-end training:
      x_raw → decomposer → IMFs → predictor → rrp_next_hat

    Loss:
      L_total = w_pred   * MSE(rrp_hat, rrp_next)
              + w_rrp    * L1(recon_ref, x_raw)
              + w_smooth * spectral_smoothness
              + w_ortho  * orthogonality
    """
    decomposer.train()
    predictor.train()

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw = x_raw.to(device)       # (B,1,L)
        rrp_next = rrp_next.to(device) # (B,1)

        opt.zero_grad(set_to_none=True)

        # Decompose
        imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)  # (B,K,L), (B,1,L), ...

        # Predict
        rrp_hat = predictor(imfs_ref)  # (B,1)

        # Losses
        loss_pred = F.mse_loss(rrp_hat, rrp_next)
        loss_rrp  = F.l1_loss(recon_ref, x_raw)

        loss_smooth = decomposer.spectral.spectral_smoothness_loss()
        loss_ortho  = decomposer.spectral.orthogonality_loss()

        loss = (
            w_pred   * loss_pred +
            w_rrp    * loss_rrp  +
            w_smooth * loss_smooth +
            w_ortho  * loss_ortho
        )

        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(decomposer.parameters()) + list(predictor.parameters()),
                max_grad_norm,
            )
        opt.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += loss_pred.item() * bs
        total_mae += F.l1_loss(rrp_hat, rrp_next).item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


def eval_epoch(
    decomposer: HybridSpectralNVMD,
    predictor: nn.Module,
    loader: DataLoader,
    device: str,
):
    decomposer.eval()
    predictor.eval()

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for x_raw, rrp_next in loader:
            x_raw = x_raw.to(device)
            rrp_next = rrp_next.to(device)

            imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)
            rrp_hat = predictor(imfs_ref)

            mse = F.mse_loss(rrp_hat, rrp_next)
            mae = F.l1_loss(rrp_hat, rrp_next)

            bs = x_raw.size(0)
            n_samples += bs
            total_mse += mse.item() * bs
            total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
# Misc
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN: Fully predictive NVMD
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str,
                    default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str,
                    default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=128)

    # Model
    ap.add_argument("--K", type=int, default=13,
                    help="Number of modes produced by decomposer")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dim-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr-dec", type=float, default=3e-4,
                    help="LR for decomposer")
    ap.add_argument("--lr-pred", type=float, default=1e-4,
                    help="LR for predictor")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)

    # Loss weights (Path 2: prediction-dominated)
    ap.add_argument("--w-pred", type=float, default=1.0,
                    help="weight for prediction loss (MSE)")
    ap.add_argument("--w-rrp", type=float, default=0.01,
                    help="weight for reconstruction L1 loss")
    ap.add_argument("--w-smooth", type=float, default=1e-3,
                    help="weight for spectral smoothness")
    ap.add_argument("--w-ortho", type=float, default=1e-3,
                    help="weight for spectral orthogonality")

    # I/O
    ap.add_argument("--out", type=str, default="./predictive_nvmd_joint.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Data -----
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = RRPDecompDataset(df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
    va_ds = RRPDecompDataset(df_va, seq_len=args.seq_len, rrp_col=args.rrp_col)

    pin = (device == "cuda")
    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # ----- Models -----
    decomposer = HybridSpectralNVMD(
        K=args.K,
        signal_len=args.seq_len,
    ).to(device)

    predictor = ModeTime2DTransformerRRP(
        K=args.K,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    # Different LRs for decomposer vs predictor (important!)
    opt = torch.optim.AdamW(
        [
            {"params": decomposer.parameters(), "lr": args.lr_dec},
            {"params": predictor.parameters(),  "lr": args.lr_pred},
        ],
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")

    print("\n====== End-to-end predictive NVMD training (Path 2) ======\n")
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = train_epoch(
            decomposer,
            predictor,
            tr_dl,
            opt,
            device,
            w_pred=args.w_pred,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )
        va_mse, va_mae = eval_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "decomposer_state": decomposer.state_dict(),
                    "predictor_state": predictor.state_dict(),
                    "args": vars(args),
                    "notes": "End-to-end predictive NVMD (no IMF supervision)",
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
