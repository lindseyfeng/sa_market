#!/usr/bin/env python3
"""
    python train_transformer_freeze_nvmd.py \
        --train-csv VMD_modes_with_residual_2018_2021.csv \
        --val-csv   VMD_modes_with_residual_2021_2022.csv \
        --decomposer-ckpt hybrid_spectral_nvmd.pt \
        --seq-len 256 \
        --warmup-epochs 20 \
        --joint-epochs 100 \
        --decomp-grad-scale 10.0 \
        --w-decomp-rrp 0.1 \
        --w-decomp-smooth 0.01 \
        --w-decomp-ortho 0.01 \
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
from train_transformer import MultiModeTransformerRRP
from nvmd_transformer import EnhancedNVMDTransformer


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
    decomposer: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    freeze_decomposer: bool = True,
    max_grad_norm: float = 10.0,
    decomp_grad_scale: float = 1.0,
    w_decomp_rrp: float = 0.0,
    w_decomp_smooth: float = 0.0,
    w_decomp_ortho: float = 0.0,
):
    is_train = optimizer is not None

    if freeze_decomposer:
        decomposer.eval()
        for p in decomposer.parameters():
            p.requires_grad = False
    else:
        for p in decomposer.parameters():
            p.requires_grad = True
        decomposer.train(is_train)

    predictor.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw    = x_raw.to(device)      # (B,1,L)
        rrp_next = rrp_next.to(device)   # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # ---- Forward through NVMD decomposer ----
        if not is_train:
            ctx = torch.no_grad()
        else:
            ctx = torch.no_grad() if freeze_decomposer else torch.enable_grad()
        # (ctx currently unused – decomposer forward would go here if needed)

        # ---- Forward through Transformer predictor ----
        rrp_next_hat = predictor(x_raw)   # (B,1)

        # prediction metrics
        mse = F.mse_loss(rrp_next_hat, rrp_next)
        mae = F.l1_loss(rrp_next_hat, rrp_next)

        if is_train:
            # base loss is prediction loss
            loss = mse

            # only add decomposer losses when it's actually trainable
            if not freeze_decomposer:
                # spectral regularizers (unsupervised, no IMF GT)
                loss_smooth = decomposer.spectral.spectral_smoothness_loss()
                loss_ortho  = decomposer.spectral.orthogonality_loss()

                loss = (
                    loss
                    + w_decomp_smooth * loss_smooth
                    + w_decomp_ortho  * loss_ortho
                )

            loss.backward()

            # If decomposer is trainable, scale its gradients
            if not freeze_decomposer and decomp_grad_scale != 1.0:
                with torch.no_grad():
                    for p in decomposer.parameters():
                        if p.grad is not None:
                            p.grad.mul_(decomp_grad_scale)

            # Gradient clipping
            if freeze_decomposer:
                # Only predictor has grads
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
            else:
                # Both decomposer and predictor have grads
                torch.nn.utils.clip_grad_norm_(
                    list(decomposer.parameters()) + list(predictor.parameters()),
                    max_grad_norm,
                )

            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
#                     Collect eval predictions
# ============================================================

def collect_predictions(
    decomposer: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    device: str,
):
    """
    Run over the validation loader and collect:
        idx (0-based in this loader), rrp_true, rrp_pred, abs_err
    Returns a list of dicts suitable for pd.DataFrame.
    """
    decomposer.eval()
    predictor.eval()

    rows = []
    idx = 0

    with torch.no_grad():
        for x_raw, rrp_next in loader:
            x_raw = x_raw.to(device)
            rrp_next = rrp_next.to(device)        # (B,1)

            yhat = predictor(x_raw)               # (B,1)

            # Flatten to 1D for easy looping
            true_vals = rrp_next.view(-1).cpu().numpy()
            pred_vals = yhat.view(-1).cpu().numpy()

            for t, p in zip(true_vals, pred_vals):
                t = float(t)
                p = float(p)
                rows.append(
                    {
                        "idx": idx,
                        "rrp_true": t,
                        "rrp_pred": p,
                        "abs_err": abs(p - t),
                    }
                )
                idx += 1

    return rows


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

    # NVMD decomposer
    ap.add_argument("--K", type=int, default=13, help="Number of modes produced by NVMD")
    ap.add_argument("--decomposer-ckpt", type=str, default="./hybrid_spectral_nvmd.pt")

    # Transformer predictor hyperparams
    ap.add_argument("--d-model",    type=int, default=128)
    ap.add_argument("--n-heads",    type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dim-ff",     type=int, default=256)
    ap.add_argument("--dropout",    type=float, default=0.1)

    # Training
    ap.add_argument("--batch",          type=int,   default=256)
    ap.add_argument("--warmup-epochs",  type=int,   default=20,
                    help="Epochs with decomposer frozen (predictor only).")
    ap.add_argument("--joint-epochs",   type=int,   default=30,
                    help="Epochs of joint training (decomposer + predictor).")
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--weight-decay",   type=float, default=1e-2)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--num-workers",    type=int,   default=0)
    ap.add_argument("--max-grad-norm",  type=float, default=10.0)

    # Decomposer gradient scaling in joint stage
    ap.add_argument("--decomp-grad-scale", type=float, default=5.0,
                    help="Multiplier for decomposer gradients in joint stage "
                         "(>1.0 makes NVMD move more per step).")

    # Decomposer-side losses in joint stage (no IMF GT involved)
    ap.add_argument("--w-decomp-rrp", type=float, default=0.1,
                    help="Weight for L1(recon_ref, x_raw) in joint stage.")
    ap.add_argument("--w-decomp-smooth", type=float, default=0.01,
                    help="Weight for spectral_smoothness_loss() in joint stage.")
    ap.add_argument("--w-decomp-ortho", type=float, default=0.01,
                    help="Weight for orthogonality_loss() in joint stage.")

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
    predictor = EnhancedNVMDTransformer(
        decomposer=decomposer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        use_multi_scale=True,
    ).to(device)

    best_val_mae = float("inf")

    # -----------------------------
    #  Stage 1: warmup (frozen decomposer)
    # -----------------------------
    print("\n===== Stage 1: Warmup (freeze decomposer, train predictor only) =====\n")

    opt_pred = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for ep in range(1, args.warmup_epochs + 1):
        tr_mse, tr_mae = run_epoch(
            decomposer,
            predictor,
            tr_dl,
            device,
            optimizer=opt_pred,
            freeze_decomposer=True,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=1.0,  # not used when frozen
            w_decomp_rrp=0.0,
            w_decomp_smooth=0.0,
            w_decomp_ortho=0.0,
        )

        va_mse, va_mae = run_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
            optimizer=None,
            freeze_decomposer=True,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=1.0,
            w_decomp_rrp=0.0,
            w_decomp_smooth=0.0,
            w_decomp_ortho=0.0,
        )

        print(
            f"[Warmup {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae

            # save checkpoint
            torch.save(
                {
                    "stage": "warmup",
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "predictor_state": predictor.state_dict(),
                    "decomposer_state": decomposer.state_dict(),
                    "args": vars(args),
                    "notes": "Warmup: predictor on frozen NVMD IMFs",
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint (warmup) with val MAE={best_val_mae:.4f} to {args.out}")

            # save evaluation CSV for diagnostics
            pred_rows = collect_predictions(decomposer, predictor, va_dl, device)
            csv_path = args.out + ".best_val.csv"
            pd.DataFrame(pred_rows).to_csv(csv_path, index=False)
            print(f"  → Saved best-val predictions to {csv_path}")

    # -----------------------------
    #  Stage 2: joint training
    # -----------------------------
    print("\n===== Stage 2: Joint training (decomposer + predictor) =====\n")

    opt_joint = torch.optim.AdamW(
        list(decomposer.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for ep in range(1, args.joint_epochs + 1):
        tr_mse, tr_mae = run_epoch(
            decomposer,
            predictor,
            tr_dl,
            device,
            optimizer=opt_joint,
            freeze_decomposer=False,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=args.decomp_grad_scale,
            w_decomp_rrp=args.w_decomp_rrp,
            w_decomp_smooth=args.w_decomp_smooth,
            w_decomp_ortho=args.w_decomp_ortho,
        )

        va_mse, va_mae = run_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
            optimizer=None,
            freeze_decomposer=False,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=1.0,  # no grad in eval
            w_decomp_rrp=args.w_decomp_rrp,
            w_decomp_smooth=args.w_decomp_smooth,
            w_decomp_ortho=args.w_decomp_ortho,
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
                    "predictor_state": predictor.state_dict(),
                    "decomposer_state": decomposer.state_dict(),
                    "args": vars(args),
                    "notes": (
                        "Joint: predictor + decomposer trained on prediction MSE "
                        f"+ decomp priors (rrp={args.w_decomp_rrp}, "
                        f"smooth={args.w_decomp_smooth}, ortho={args.w_decomp_ortho}, "
                        f"decomp_grad_scale={args.decomp_grad_scale})"
                    ),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint (joint) with val MAE={best_val_mae:.4f} to {args.out}")

            # save diagnostic predictions
            pred_rows = collect_predictions(decomposer, predictor, va_dl, device)
            csv_path = args.out + ".best_val.csv"
            pd.DataFrame(pred_rows).to_csv(csv_path, index=False)
            print(f"  → Saved best-val predictions to {csv_path}")


if __name__ == "__main__":
    main()
