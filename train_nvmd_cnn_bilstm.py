#!/usr/bin/env python3
# train_nvmd_autoencoder_raw.py
#
# Step 1 (per-mode): Train ONLY the decomposer (NVMD_Autoencoder) for a single mode.
# - Inputs:  x = raw RRP window (default) or sum of raw IMFs (if you want)
# - Targets: y = raw IMF window for ONE mode, shape (1, L)
# - Loss: Huber reconstruction in RAW scale (no normalization, no sigmoid inside model)
#
# You run this script separately for Mode_1, Mode_2, ..., Residual by changing --mode-col
# and (optionally) --outdir.

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_autoencoder import NVMD_Autoencoder


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset (per-mode, RAW)
# -------------------------
class DecompOnlyDatasetRawSingleMode(Dataset):
    """
    Train the decomposer on RAW IMF for a single mode (no normalization).

    For each window i..i+L-1:
      - x: (1, L) raw RRP window (or sum of all IMFs if x_mode='sum')
      - y: (1, L) raw IMF window for the selected mode
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mode_col: str,          # e.g. "Mode_1" or "Residual"
        target_col: str = "RRP",
        seq_len: int = 1024,
        x_mode: str = "raw",    # "raw" (RRP window) or "sum" (sum of all IMFs)
        all_decomp_cols: list[str] | None = None,  # needed only if x_mode == "sum"
    ):
        super().__init__()
        self.mode_col = mode_col
        self.L = seq_len
        self.x_mode = x_mode

        # Target IMF (single mode)
        if mode_col not in df.columns:
            raise ValueError(f"mode_col '{mode_col}' not in dataframe columns.")
        imf_np = df[mode_col].to_numpy(dtype=np.float32)     # (T,)
        self.imf = torch.from_numpy(imf_np).contiguous()     # (T,)

        # RRP series
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in dataframe columns.")
        rrp_np = df[target_col].to_numpy(dtype=np.float32)   # (T,)
        self.rrp = torch.from_numpy(rrp_np).contiguous()     # (T,)
        T = self.rrp.shape[0]

        if self.x_mode == "raw":
            self.x_series = self.rrp                    # (T,)
        elif self.x_mode == "sum":
            if all_decomp_cols is None:
                raise ValueError("all_decomp_cols must be provided when x_mode='sum'")
            imfs_np_all = df[all_decomp_cols].to_numpy(dtype=np.float32)  # (T, K_all)
            imfs_all = torch.from_numpy(imfs_np_all).transpose(0, 1).contiguous()  # (K_all, T)
            self.x_series = imfs_all.sum(dim=0)        # (T,)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum'")

        # number of sliding windows (inclusive end)
        self.N = max(0, T - self.L + 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # x: (1, L) raw input window
        x = self.x_series[i:i+L].unsqueeze(0)   # (1, L)
        # y: (1, L) raw IMF window for this mode
        y = self.imf[i:i+L].unsqueeze(0)        # (1, L)
        return x, y


# -------------------------
# Utils
# -------------------------
def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return cols


# -------------------------
# Train/Eval Epochs (RAW, per-mode)
# -------------------------
def train_or_eval_epoch_raw_single_mode(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    clip_grad: float | None = None,
):
    """
    RAW-scale training/eval for a single mode.

    loss_recon = mean HuberLoss over (B,1,L) in RAW scale.
    No cross-mode sum-consistency term here because K=1.
    """
    is_train = optimizer is not None
    model.train(is_train)

    huber = nn.HuberLoss(delta=10.0, reduction="mean")

    tot, log_total = 0, 0.0

    for x_win, y_win in loader:
        x_win = x_win.to(device)   # (B,1,L)
        y_win = y_win.to(device)   # (B,1,L)

        # forward (raw output, no sigmoid here)
        imf_pred = model(x_win)    # (B,1,L)

        # reconstruction loss in RAW scale
        loss_recon = F.l1_loss(imf_pred, y_win)  # scalar

        loss = loss_recon

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = x_win.size(0)
        tot       += bs
        log_total += loss.item() * bs

    return log_total / max(tot, 1)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--target-col", type=str, default="RRP")
    ap.add_argument("--mode-col",   type=str, default="Mode_1",
                    help="Which IMF column to train this decomposer on (e.g. Mode_1, ..., Residual)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--x-mode", type=str, choices=["raw", "sum"], default="raw",
                    help="Decomposer input: raw RRP window (default) or sum of raw IMFs")

    # Model
    ap.add_argument("--base", type=int, default=128)

    # Train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0)

    # I/O
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_autoencoder_raw_mode")
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    # All decomp columns only needed if x_mode == "sum"
    all_decomp_cols = None
    if args.x_mode == "sum":
        all_decomp_cols = build_default_13(df_tr)

    # Datasets / Loaders (per-mode)
    tr_ds = DecompOnlyDatasetRawSingleMode(
        df=df_tr,
        mode_col=args.mode_col,
        target_col=args.target_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
        all_decomp_cols=all_decomp_cols,
    )
    va_ds = DecompOnlyDatasetRawSingleMode(
        df=df_va,
        mode_col=args.mode_col,
        target_col=args.target_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
        all_decomp_cols=all_decomp_cols,
    )

    pin = (device == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)

    # Model + Optim (per-mode: K=1)
    model = NVMD_Autoencoder(
        in_ch=1,
        base=args.base,
        K=1,                      # single mode
        signal_len=args.seq_len,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss = train_or_eval_epoch_raw_single_mode(
            model, tr_dl, device,
            optimizer=opt,
            clip_grad=args.clip_grad,
        )
        va_loss = train_or_eval_epoch_raw_single_mode(
            model, va_dl, device,
            optimizer=None,
            clip_grad=None,
        )

        print(f"[Epoch {ep:03d}] "
              f"train: recon={tr_loss:.6f} | "
              f"val: recon={va_loss:.6f}")

        # Track best on validation loss
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # periodic checkpoint
        if (args.save_every > 0) and (ep % args.save_every == 0):
            ck = {
                "epoch": ep,
                "val_best": best_val,
                "model_state": model.state_dict(),
                "args": vars(args),
                "mode_col": args.mode_col,
                "notes": "RAW per-mode training for NVMD_Autoencoder",
            }
            torch.save(ck, os.path.join(args.outdir, f"{args.mode_col}_epoch_{ep:03d}.pt"))

    # Save best
    if best_state is not None:
        out_best = os.path.join(args.outdir, f"{args.mode_col}_best.pt")
        torch.save({
            "epoch": "best",
            "val_best": best_val,
            "model_state": best_state,
            "args": vars(args),
            "mode_col": args.mode_col,
            "notes": "RAW per-mode training for NVMD_Autoencoder",
        }, out_best)
        print(f"Saved best checkpoint for {args.mode_col} → {out_best}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
