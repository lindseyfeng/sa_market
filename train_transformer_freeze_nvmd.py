#!/usr/bin/env python3
"""
Train a new Transformer predictor on top of a pretrained HybridSpectralNVMD decomposer.

- Input: raw RRP sequence windows (length L)
- Decomposer: pretrained HybridSpectralNVMD → IMFs (B, K, L)
- Predictor: new MultiModeTransformerRRP trained on IMFs to predict next-step RRP

Usage example:

    python3 train_nvmd_transformer.py \
        --train-csv VMD_modes_with_residual_2018_2021.csv \
        --val-csv   VMD_modes_with_residual_2021_2022.csv \
        --decomposer-ckpt hybrid_spectral_nvmd.pt \
        --seq-len 64 \
        --epochs 50 \
        --out nvmd_transformer_rrp.pt
"""

import argparse
import os
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_nvmd import HybridSpectralNVMD
from train_transformer import MultiModeTransformerRRP  # your definition from the other script


# ============================================================
#                     Dataset (RRP only)
# ============================================================

class RRPWindowDataset(Dataset):
    """
    Given a dataframe with an RRP column, returns:

      x_raw:    (1, L)  window of raw RRP values [t, ..., t+L-1]
      rrp_next: (1,)    RRP at time t+L

    The decomposer will turn x_raw into IMFs inside the training loop.
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 64, rrp_col: str = "RRP"):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"RRP column '{rrp_col}' not in dataframe")

        rrp_np = df[rrp_col].to_numpy(dtype=np.float32)  # (T,)
        self.rrp = torch.from_numpy(rrp_np)              # (T,)

        T = self.rrp.shape[0]
        # We need rrp[t+L] to exist, so max start index = T-L-1
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # window [i, ..., i+L-1]
        x_raw = self.rrp[i:i+L]          # (L,)
        x_raw = x_raw.unsqueeze(0)       # (1, L)  channel-first for NVMD

        # next-step RRP is at time t+L
        rrp_next = self.rrp[i + L].unsqueeze(0)  # (1,)

        return x_raw, rrp_next


# ============================================================
#                 Training / Evaluation Epochs
# ============================================================

def run_epoch(
    decomposer: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    freeze_decomposer: bool = True,
    max_grad_norm: float = 10.0,
):
    """
    If optimizer is provided → training, otherwise evaluation.

    Pipeline:
        x_raw (B,1,L) --decomposer--> imfs_ref (B,K,L)
                             |
                             v
                    predictor → rrp_next_hat (B,1)

    Loss: MSE on raw RRP.
    Metrics: MSE, MAE (on raw RRP).
    """
    is_train = optimizer is not None

    if freeze_decomposer:
        decomposer.eval()
        # Just to be safe:
        for p in decomposer.parameters():
            p.requires_grad = False
    else:
        decomposer.train(is_train)

    predictor.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw   = x_raw.to(device)      # (B,1,L)
        rrp_next = rrp_next.to(device)  # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # ---- Forward through NVMD decomposer ----
        # imfs_ref: (B,K,L)
        with torch.no_grad() if freeze_decomposer else torch.enable_grad():
            imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)

        # If decomposer is frozen, stop grads into it explicitly:
        if freeze_decomposer:
            imfs_ref = imfs_ref.detach()

        # ---- Forward through Transformer predictor ----
        rrp_next_hat = predictor(imfs_ref)   # (B,1)

        mse = F.mse_loss(rrp_next_hat, rrp_next)
        mae = F.l1_loss(rrp_next_hat, rrp_next)

        if is_train:
            mse.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
#                          Utilities
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
#                            MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=64)

    # NVMD decomposer
    ap.add_argument("--K", type=int, default=13, help="Number of modes produced by NVMD")
    ap.add_argument("--decomposer-ckpt", type=str, default="./hybrid_spectral_nvmd.pt")
    ap.add_argument("--freeze-decomposer", action="store_true",
                    help="Freeze NVMD weights (recommended).")

    # Transformer predictor hyperparams
    ap.add_argument("--d-model",    type=int, default=128)
    ap.add_argument("--n-heads",    type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dim-ff",     type=int, default=256)
    ap.add_argument("--dropout",    type=float, default=0.1)

    # Training
    ap.add_argument("--batch",          type=int,   default=256)
    ap.add_argument("--epochs",         type=int,   default=50)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--weight-decay",   type=float, default=1e-2)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--num-workers",    type=int,   default=0)
    ap.add_argument("--max-grad-norm",  type=float, default=10.0)

    # I/O
    ap.add_argument("--out", type=str, default="./nvmd_transformer_rrp.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    #  Load data
    # -----------------------------
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = RRPWindowDataset(df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
    va_ds = RRPWindowDataset(df_va, seq_len=args.seq_len, rrp_col=args.rrp_col)

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

    print(f"Train windows: {len(tr_ds)}, Val windows: {len(va_ds)}")

    # -----------------------------
    #  Load pretrained NVMD decomposer
    # -----------------------------
    decomposer = HybridSpectralNVMD(K=args.K, signal_len=args.seq_len).to(device)

    dec_ckpt = torch.load(args.decomposer_ckpt, map_location="cpu")
    # be flexible about checkpoint format
    if "model_state" in dec_ckpt:
        dec_state = dec_ckpt["model_state"]
    elif "decomposer_state" in dec_ckpt:
        dec_state = dec_ckpt["decomposer_state"]
    else:
        dec_state = dec_ckpt

    missing_d, unexpected_d = decomposer.load_state_dict(dec_state, strict=False)
    print("Loaded NVMD decomposer.")
    print("  missing:", missing_d)
    print("  unexpected:", unexpected_d)

    # -----------------------------
    #  New Transformer predictor (from scratch)
    # -----------------------------
    predictor = MultiModeTransformerRRP(
        K=args.K,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")

    # -----------------------------
    #  Training loop
    # -----------------------------
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(
            decomposer,
            predictor,
            tr_dl,
            device,
            optimizer=optimizer,
            freeze_decomposer=args.freeze_decomposer,
            max_grad_norm=args.max_grad_norm,
        )

        va_mse, va_mae = run_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
            optimizer=None,
            freeze_decomposer=args.freeze_decomposer,
            max_grad_norm=args.max_grad_norm,
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
                    "predictor_state": predictor.state_dict(),
                    "decomposer_ckpt": args.decomposer_ckpt,
                    "args": vars(args),
                    "notes": "Transformer trained on NVMD IMFs (frozen decomposer)"
                             if args.freeze_decomposer
                             else "Transformer trained on NVMD IMFs (joint gradients)",
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
