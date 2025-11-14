#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- import your model ----
from nvmd_transformer_rrp_model import NVMDTransformerRRPModel
# ^ make sure this is the file where your NVMDTransformerRRPModel lives


# ==========================================================
#                           Utils
# ==========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RRPDataset(Dataset):
    """
    Simple dataset:
      - Input:  x_raw: (1, L)   window of RRP
      - Target: y:     (1,)     next-step RRP
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 64, rrp_col: str = "RRP"):
        super().__init__()
        self.L = seq_len
        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not found in dataframe columns.")

        rrp = df[rrp_col].to_numpy(dtype=np.float32)
        self.rrp = torch.from_numpy(rrp)
        T = len(rrp)
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        x = self.rrp[i:i+L].unsqueeze(0)   # (1,L)
        y = self.rrp[i+L].unsqueeze(0)     # (1,)
        return x, y


def load_submodule_state(submodule: nn.Module, ckpt_path: str, device: str):
    """
    Robust loader:
      - ckpt can be a plain state_dict or a dict with keys like 'model_state', 'state_dict', etc.
      - Loads into the given submodule (e.g., model.decomposer or model.predictor).
    """
    if ckpt_path is None or ckpt_path == "":
        return

    print(f"Loading weights into {submodule.__class__.__name__} from: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)

    if isinstance(ck, dict):
        # Try common keys
        for key in ["model_state", "state_dict", "model_state_dict"]:
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break

    if not isinstance(ck, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} does not look like a state_dict or dict-with-state.")

    # Strip "module." if present (from DataParallel)
    cleaned = {}
    for k, v in ck.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = submodule.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning: missing keys when loading into {submodule.__class__.__name__}: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys when loading into {submodule.__class__.__name__}: {unexpected}")


def maybe_freeze(module: nn.Module, freeze: bool):
    if not freeze:
        return
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    grad_clip: float | None = 10.0,
):
    """
    One epoch.
    If optimizer is provided → training; else → eval.

    Returns:
        mse, mae  (both on RRP next-step, raw scale)
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw = x_raw.to(device)        # (B,1,L)
        rrp_next = rrp_next.to(device)  # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        y_pred = model(x_raw)           # (B,1)

        mse = F.mse_loss(y_pred, rrp_next)
        mae = F.l1_loss(y_pred, rrp_next)

        if is_train:
            mse.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ==========================================================
#                           Main
# ==========================================================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=64)

    # Model hyperparams (must match your model file)
    ap.add_argument("--K",         type=int, default=13)
    ap.add_argument("--base",      type=int, default=64)
    ap.add_argument("--d-model",   type=int, default=256)
    ap.add_argument("--n-heads",   type=int, default=8)
    ap.add_argument("--num-layers",type=int, default=6)
    ap.add_argument("--dim-ff",    type=int, default=1024)
    ap.add_argument("--dropout",   type=float, default=0.1)

    # Pretrained ckpts
    ap.add_argument("--decomposer-ckpt", type=str, default="./transformer_only_rrp.pt",
                    help="Path to pretrained decomposer (MultiModeNVMD) checkpoint.")
    ap.add_argument("--predictor-ckpt",  type=str, default="./nvmd_ae_imf_recon.pt",
                    help="Path to pretrained transformer predictor checkpoint.")

    # Freeze options
    ap.add_argument("--freeze-decomposer", action="store_true",
                    help="If set, decomposer will be frozen after loading.")
    ap.add_argument("--freeze-predictor",  action="store_true",
                    help="If set, predictor will be frozen after loading.")

    # Training
    ap.add_argument("--epochs",    type=int, default=50)
    ap.add_argument("--batch",     type=int, default=256)
    ap.add_argument("--lr",        type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    # I/O
    ap.add_argument("--out", type=str, default="./nvmd_transformer_rrp_finetuned.pt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------- Load data -----------------
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = RRPDataset(df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
    va_ds = RRPDataset(df_va, seq_len=args.seq_len, rrp_col=args.rrp_col)

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

    # --------------- Build model ----------------
    model = NVMDTransformerRRPModel(
        seq_len=args.seq_len,
        K=args.K,
        base=args.base,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        freeze_decomposer=False,   # we control freezing manually below
    ).to(device)

    # --------------- Load pretrained weights ---------------
    if args.decomposer-ckpt:
        load_submodule_state(model.decomposer, args.decomposer_ckpt, device=device)
    if args.predictor-ckpt:
        load_submodule_state(model.predictor, args.predictor_ckpt, device=device)

    # Apply freezing if requested
    maybe_freeze(model.decomposer, args.freeze_decomposer)
    maybe_freeze(model.predictor,  args.freeze_predictor)

    # --------------- Optimizer ----------------
    # Only train parameters that still require grad (so freezing works)
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        train_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")

    # --------------- Training loop ----------------
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(
            model,
            tr_dl,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
            grad_clip=None,
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
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
