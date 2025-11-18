#!/usr/bin/env python3
"""
Usage example:

    python train.py \
        --train-csv VMD_modes_with_residual_2018_2021.csv \
        --val-csv   VMD_modes_with_residual_2021_2022.csv \
        --seq-len 256 \
        --warmup-epochs 20 \
        --joint-epochs 100 \
        --out nvmd_transformer_joint.pt
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
from MultiModeTransformerRRP import MultiModeTransformerRRP  


# ============================================================
#                     Dataset (RRP only)
# ============================================================

class RRPWindowDataset(Dataset):
    """
    Given a dataframe with an RRP column, returns:

      x_raw:    (1, L)  window of raw RRP values [t, ..., t+L-1]
      rrp_next: (1,)    RRP at time t+L
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
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    freeze_decomposer: bool = False,
    w_pred: float = 1.0,
    w_rrp: float = 0.0,
    w_smooth: float = 0.0,
    w_ortho: float = 0.0,
    max_grad_norm: float = 10.0,
):
    """
    If optimizer is provided → training, otherwise evaluation.

    Model is MultiModeTransformerRRP(use_nvmd=True).

    Forward:
        x_raw (B,1,L) → model(x_raw, return_nvmd=True) → 
            rrp_hat (B,1), recon_ref (B,1,L),
            smooth_loss (scalar), ortho_loss (scalar)

    Loss:
        w_pred   * MSE(rrp_hat, rrp_next)
      + w_rrp    * L1(recon_ref, x_raw)
      + w_smooth * smooth_loss
      + w_ortho  * ortho_loss
    """
    is_train = optimizer is not None
    model.train(is_train)

    # Optionally freeze NVMD submodule (during warmup)
    if freeze_decomposer and hasattr(model, "decomposer"):
        for p in model.decomposer.parameters():
            p.requires_grad = False
    elif hasattr(model, "decomposer"):
        for p in model.decomposer.parameters():
            p.requires_grad = True

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw    = x_raw.to(device)      # (B,1,L)
        rrp_next = rrp_next.to(device)   # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Forward through integrated NVMD+Transformer
        # (rrp_hat, recon_ref, smooth_loss, ortho_loss)
        rrp_hat, recon_ref, smooth_loss, ortho_loss = model(
            x_raw,
            return_nvmd=True,
        )

        mse = F.mse_loss(rrp_hat, rrp_next)
        mae = F.l1_loss(rrp_hat, rrp_next)

        # NVMD regularizers (only contribute when not frozen)
        if freeze_decomposer:
            # don't let these influence gradients in warmup
            loss_rrp    = torch.tensor(0.0, device=device)
            loss_smooth = torch.tensor(0.0, device=device)
            loss_ortho  = torch.tensor(0.0, device=device)
        else:
            loss_rrp    = F.l1_loss(recon_ref, x_raw)
            loss_smooth = smooth_loss
            loss_ortho  = ortho_loss

        loss = (
            w_pred   * mse
          + w_rrp    * loss_rrp
          + w_smooth * loss_smooth
          + w_ortho  * loss_ortho
        )

        if is_train:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    ap.add_argument("--seq-len",   type=int, default=256)

    # NVMD + Transformer
    ap.add_argument("--K", type=int, default=13, help="Number of modes produced by NVMD")
    ap.add_argument("--d-model",    type=int, default=128)
    ap.add_argument("--n-heads",    type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dim-ff",     type=int, default=256)
    ap.add_argument("--dropout",    type=float, default=0.1)

    # Optional: initialize decomposer from a separate NVMD checkpoint
    ap.add_argument("--decomposer-ckpt", type=str, default="",
                    help="Optional NVMD-only checkpoint to init model.decomposer")

    # Training
    ap.add_argument("--batch",          type=int,   default=256)
    ap.add_argument("--warmup-epochs",  type[int],  default=20,
                    help="Epochs with decomposer frozen (prediction-only).")
    ap.add_argument("--joint-epochs",   type[int],  default=80,
                    help="Epochs of joint training (NVMD + Transformer).")
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--weight-decay",   type=float, default=1e-2)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--num-workers",    type=int,   default=0)
    ap.add_argument("--max-grad-norm",  type=float, default=10.0)

    # Loss weights for joint stage
    ap.add_argument("--w-pred",   type=float, default=1.0)
    ap.add_argument("--w-rrp",    type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=0.01)
    ap.add_argument("--w-ortho",  type=float, default=0.01)

    # I/O
    ap.add_argument("--out", type=str, default="./nvmd_transformer_joint.pt")

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
    #  Build model (NVMD inside)
    # -----------------------------
    model = MultiModeTransformerRRP(
        K=args.K,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        use_nvmd=True,   # <--- key
    ).to(device)

    # Optionally initialize decomposer from separate NVMD ckpt
    if args.decomposer-ckpt and os.path.exists(args.decomposer_ckpt):
        print(f"Loading decomposer initialization from {args.decomposer_ckpt}")
        dec_ckpt = torch.load(args.decomposer_ckpt, map_location="cpu")
        if isinstance(dec_ckpt, dict) and "model_state" in dec_ckpt:
            dec_state = dec_ckpt["model_state"]
        else:
            dec_state = dec_ckpt
        missing, unexpected = model.decomposer.load_state_dict(dec_state, strict=False)
        print("  decomposer missing:", missing)
        print("  decomposer unexpected:", unexpected)

    best_val_mae = float("inf")

    # -----------------------------
    #  Stage 1: Warmup (freeze NVMD)
    # -----------------------------
    print("\n===== Stage 1: Warmup (Transformer only, NVMD frozen) =====\n")

    opt_warmup = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for ep in range(1, args.warmup_epochs + 1):
        tr_mse, tr_mae = run_epoch(
            model,
            tr_dl,
            device,
            optimizer=opt_warmup,
            freeze_decomposer=True,
            w_pred=1.0,
            w_rrp=0.0,
            w_smooth=0.0,
            w_ortho=0.0,
            max_grad_norm=args.max_grad_norm,
        )

        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
            freeze_decomposer=True,
            w_pred=1.0,
            w_rrp=0.0,
            w_smooth=0.0,
            w_ortho=0.0,
            max_grad_norm=args.max_grad_norm,
        )

        print(
            f"[Warmup {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "stage": "warmup",
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint (warmup) with val MAE={best_val_mae:.4f} to {args.out}")

    # -----------------------------
    #  Stage 2: Joint training
    # -----------------------------
    print("\n===== Stage 2: Joint (NVMD + Transformer) =====\n")

    opt_joint = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for ep in range(1, args.joint_epochs + 1):
        tr_mse, tr_mae = run_epoch(
            model,
            tr_dl,
            device,
            optimizer=opt_joint,
            freeze_decomposer=False,
            w_pred=args.w_pred,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )

        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
            freeze_decomposer=False,
            w_pred=args.w_pred,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )

        print(
            f"[Joint {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "stage": "joint",
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint (joint) with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
