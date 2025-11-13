#!/usr/bin/env python3
# train_nvmd_autoencoder_raw.py
#
# Step 1: Train ONLY the decomposer (NVMD_Autoencoder) on raw (unnormalized) IMFs.
# - Inputs: x = raw RRP window (default) or sum of raw IMFs (identical if decomposition is exact)
# - Targets: y = raw IMF windows (K, L), unnormalized
# - Loss: Huber(per-IMF) + λ * L1(sum-consistency) in RAW scale
# - No sigmoid, no normalization, no cropping

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# your decomposer class
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
# Dataset
# -------------------------
class DecompOnlyDatasetRaw(Dataset):
    """
    Train the decomposer on RAW IMFs (no normalization).
    For each window i..i+L-1:
      - x: (1, L)   raw RRP window (or sum of IMFs if x_mode='sum')
      - y: (K, L)   raw IMF windows (Mode_1..Mode_12, Residual)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],     # ["Mode_1", ..., "Mode_12", "Residual"]
        target_col: str = "RRP",
        seq_len: int = 1024,
        x_mode: str = "raw",        # "raw" (RRP window) or "sum"
    ):
        super().__init__()
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)
        self.L = seq_len
        self.x_mode = x_mode

        # arrays → tensors
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)     # (T, K)
        rrp_np  = df[target_col].to_numpy(dtype=np.float32)      # (T, )

        # Store as (K, T) and (T,)
        self.imfs = torch.from_numpy(imfs_np).transpose(0, 1).contiguous()  # (K, T)
        self.rrp  = torch.from_numpy(rrp_np).contiguous()                    # (T,)
        T = self.rrp.shape[0]

        if self.x_mode == "raw":
            self.x_series = self.rrp                      # (T,)
        elif self.x_mode == "sum":
            self.x_series = self.imfs.sum(dim=0)          # (T,), sum of raw IMFs
        else:
            raise ValueError("x_mode must be 'raw' or 'sum'")

        # Number of sliding windows (inclusive end)
        self.N = max(0, T - self.L + 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # x: (1, L) raw input
        x = self.x_series[i:i+L].unsqueeze(0)       # (1, L)
        # y: (K, L) raw IMFs
        y = self.imfs[:, i:i+L]                     # (K, L)
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
# Train/Eval Epochs (RAW scale)
# -------------------------
def train_or_eval_epoch(model, loader, device, alpha, beta,
                        imf_mins, imf_maxs,
                        optimizer=None, clip_grad=None,
                        sum_reg: float = 0.2):

    is_train = optimizer is not None
    model.train(is_train)

    decomposer_frozen = all(not p.requires_grad for p in model.decomposer.parameters())
    if decomposer_frozen:
        model.decomposer.eval()

    total, sum_loss, sum_d, sum_p = 0, 0.0, 0.0, 0.0

    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1
    )

    # IMPORTANT: per-element Huber, no reduction yet
    loss_fn_imf = torch.nn.HuberLoss(delta=10.0, reduction="none")

    for xb, imfs_true_norm, yb in loader:
        xb = xb.to(device)                         # (B,1,L)
        imfs_true_norm = imfs_true_norm.to(device) # (B,K,L)
        yb = yb.to(device)                         # (B,1)

        imfs_pred_norm, y_modes_norm = model(xb)   # (B,K,L), (B,K)

        # ---- price prediction (same as before) ----
        y_modes_raw = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
        y_pred = y_modes_raw.sum(dim=1, keepdim=True)                            # (B,1)

        # ---- PER-MODE decomposition loss in normalized space ----
        # elementwise Huber: (B,K,L)
        per_elem = loss_fn_imf(imfs_pred_norm, imfs_true_norm)   # (B,K,L)

        # average over batch & time → per-mode loss: (K,)
        per_mode_loss = per_elem.mean(dim=(0, 2))                # (K,)

        # scalar decomp loss = mean over modes (same scale as old 'mean over everything')
        loss_decomp = per_mode_loss.mean()

        # optional: per-mode sum-consistency in normalized space
        sum_pred_norm = imfs_pred_norm.sum(dim=1)                # (B,L)
        sum_true_norm = imfs_true_norm.sum(dim=1)                # (B,L)
        loss_sumcons = F.l1_loss(sum_pred_norm, sum_true_norm)

        loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons

        # ---- price loss (raw space) ----
        loss_pred = F.l1_loss(y_pred, yb)

        loss = alpha * loss_decomp_reg + beta * loss_pred

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = xb.size(0)
        total += bs
        sum_loss += loss.item() * bs
        sum_d    += loss_decomp_reg.item() * bs
        sum_p    += loss_pred.item() * bs

        # OPTIONAL: if you want to quickly inspect per-mode losses for the first batch
        # (uncomment if useful)
        # print("per-mode recon:", per_mode_loss.detach().cpu().numpy())

    return (sum_loss / max(total, 1),
            sum_d    / max(total, 1),
            sum_p    / max(total, 1))
  


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--target-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--x-mode", type=str, choices=["raw", "sum"], default="raw",
                    help="Decomposer input: raw RRP window (default) or sum of raw IMFs")

    # Model
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--K", type=int, default=13)

    # Train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sum-reg", type=float, default=0.2)
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0)

    # I/O
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_autoencoder_raw")
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    # Columns
    decomp_cols = build_default_13(df_tr)

    # Datasets / Loaders
    tr_ds = DecompOnlyDatasetRaw(
        df=df_tr,
        decomp_cols=decomp_cols,
        target_col=args.target_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
    )
    va_ds = DecompOnlyDatasetRaw(
        df=df_va,
        decomp_cols=decomp_cols,
        target_col=args.target_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
    )

    pin = (device == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)

    # Model + Optim
    model = NVMD_Autoencoder(in_ch=1, base=args.base, K=args.K, signal_len=args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_tot, tr_rec, tr_sum = train_or_eval_epoch(
            model, tr_dl, device,
            optimizer=opt,
            clip_grad=args.clip_grad,
            sum_reg=args.sum_reg
        )
        va_tot, va_rec, va_sum = train_or_eval_epoch(
            model, va_dl, device,
            optimizer=None,
            sum_reg=args.sum_reg
        )

        print(f"[Epoch {ep:03d}] "
              f"train: total={tr_tot:.6f} recon={tr_rec:.6f} sum={tr_sum:.6f} | "
              f"val: total={va_tot:.6f} recon={va_rec:.6f} sum={va_sum:.6f}")

        # Track best on validation total loss
        if va_tot < best_val:
            best_val = va_tot
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # periodic checkpoint
        if (args.save_every > 0) and (ep % args.save_every == 0):
            ck = {
                "epoch": ep,
                "val_best": best_val,
                "model_state": model.state_dict(),
                "args": vars(args),
                "decomp_cols": decomp_cols,
                "notes": "RAW training (no normalization) for NVMD_Autoencoder",
            }
            torch.save(ck, os.path.join(args.outdir, f"epoch_{ep:03d}.pt"))

    # Save best
    if best_state is not None:
        torch.save({
            "epoch": "best",
            "val_best": best_val,
            "model_state": best_state,
            "args": vars(args),
            "decomp_cols": decomp_cols,
            "notes": "RAW training (no normalization) for NVMD_Autoencoder",
        }, os.path.join(args.outdir, "best.pt"))
        print(f"Saved best checkpoint → {os.path.join(args.outdir,'best.pt')}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
