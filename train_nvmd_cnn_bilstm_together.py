#!/usr/bin/env python3
# train_nvmd_cnn_bilstm_mode.py
#
# Per-mode joint training:
#   Model: NVMD_MRC_BiLSTM (per-mode: K=1)
#   Inputs:  x_raw      = raw RRP window, shape (1, L)
#   Targets: imf_win    = raw IMF window for ONE mode, shape (1, L)
#            imf_next   = raw IMF at time t+L, shape (1,)
#   Loss:    alpha * L1(imf_pred, imf_win) + beta * L1(y_mode_pred, imf_next)
#
# Run separately for Mode_1, Mode_2, ..., Residual by changing --mode-col.

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  # per-mode joint model (K=1 inside)


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
# Dataset: per-mode, joint training
# -------------------------
class NVMDModeDataset(Dataset):
    """
    For each window i..i+L-1:

      x_raw:    raw RRP window, shape (1, L)
      imf_win:  raw IMF window for this mode, shape (1, L)
      imf_next: raw IMF at time t+L, shape (1,)

    So index i corresponds to:
      - history: t = i, ..., i+L-1
      - target:  IMF(t+L)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mode_col: str,           # e.g. "Mode_1" or "Residual"
        rrp_col: str = "RRP",
        seq_len: int = 1024,
        x_mode: str = "raw",     # "raw" or "sum"
        all_decomp_cols: list[str] | None = None,  # only needed if x_mode == "sum"
    ):
        super().__init__()
        self.L = seq_len
        self.mode_col = mode_col
        self.x_mode = x_mode

        if mode_col not in df.columns:
            raise ValueError(f"mode_col '{mode_col}' not in dataframe.")

        imf_np = df[mode_col].to_numpy(dtype=np.float32)   # (T,)
        self.imf = torch.from_numpy(imf_np).contiguous()   # (T,)

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")
        rrp_np = df[rrp_col].to_numpy(dtype=np.float32)    # (T,)
        self.rrp = torch.from_numpy(rrp_np).contiguous()   # (T,)
        T = self.rrp.shape[0]

        if x_mode == "raw":
            self.x_series = self.rrp                       # (T,)
        elif x_mode == "sum":
            if all_decomp_cols is None:
                raise ValueError("all_decomp_cols must be provided when x_mode='sum'")
            imfs_np_all = df[all_decomp_cols].to_numpy(dtype=np.float32)  # (T,K_all)
            imfs_all = torch.from_numpy(imfs_np_all).transpose(0, 1).contiguous()  # (K_all,T)
            self.x_series = imfs_all.sum(dim=0)            # (T,)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum'")

        # We need t+L for the next-step target, so max start index is T-L-1
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # Input window: RRP or sum of IMFs, times [i, ..., i+L-1]
        x_raw = self.x_series[i:i+L].unsqueeze(0)       # (1,L)

        # Reconstruction window: IMF for this mode, times [i, ..., i+L-1]
        imf_win = self.imf[i:i+L].unsqueeze(0)          # (1,L)

        # Next-step target: IMF(t+L)
        imf_next = self.imf[i+L].unsqueeze(0)           # (1,)

        return x_raw, imf_win, imf_next


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
# Train/Eval epoch: joint decomposer + predictor
# -------------------------
def train_or_eval_epoch_joint(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    alpha: float,
    beta: float,
    optimizer=None,
    clip_grad: float | None = None,
):
    """
    alpha: weight for decomposition/reconstruction loss
    beta:  weight for next-step prediction loss
    Losses are computed in RAW scale (no normalization).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    total_recon_sum = 0.0
    total_pred_sum = 0.0
    n_samples = 0

    for x_raw, imf_win, imf_next in loader:
        x_raw   = x_raw.to(device)      # (B,1,L)
        imf_win = imf_win.to(device)    # (B,1,L)
        imf_next = imf_next.to(device)  # (B,1)

        # Forward: per-mode model returns reconstructed IMF and next-step mode
        imf_pred, y_mode_pred = model(x_raw)    # (B,1,L), (B,1)

        # Reconstruction loss in raw scale
        loss_recon = F.l1_loss(imf_pred, imf_win)

        # Next-step prediction loss in raw scale
        loss_pred = F.l1_loss(y_mode_pred, imf_next)

        # Total combined loss
        loss = alpha * loss_recon + beta * loss_pred

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_loss_sum  += loss.item() * bs
        total_recon_sum += loss_recon.item() * bs
        total_pred_sum  += loss_pred.item() * bs

    denom = max(n_samples, 1)
    return (
        total_loss_sum  / denom,
        total_recon_sum / denom,
        total_pred_sum  / denom,
    )


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",    type=str, default="RRP")
    ap.add_argument("--mode-col",   type=str, default="Mode_1",
                    help="Which IMF column to train this model on (e.g. Mode_1, ..., Residual)")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--x-mode", type=str, choices=["raw", "sum"], default="raw",
                    help="Model input: raw RRP window (default) or sum of raw IMFs")

    # Model
    ap.add_argument("--base", type=int, default=256)
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)

    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=0.2, help="weight for IMF reconstruction loss")
    ap.add_argument("--beta",  type=float, default=1.0, help="weight for next-step prediction loss")
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    # I/O
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_cnn_bilstm_mode")
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

    # Datasets / Loaders
    tr_ds = NVMDModeDataset(
        df=df_tr,
        mode_col=args.mode_col,
        rrp_col=args.rrp_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
        all_decomp_cols=all_decomp_cols,
    )
    va_ds = NVMDModeDataset(
        df=df_va,
        mode_col=args.mode_col,
        rrp_col=args.rrp_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
        all_decomp_cols=all_decomp_cols,
    )

    pin = (device == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)

    # Model (per-mode joint NVMD + predictor)
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        freeze_decomposer=False,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_tot, tr_rec, tr_pred = train_or_eval_epoch_joint(
            model, tr_dl, device,
            alpha=args.alpha,
            beta=args.beta,
            optimizer=opt,
            clip_grad=args.clip_grad,
        )
        va_tot, va_rec, va_pred = train_or_eval_epoch_joint(
            model, va_dl, device,
            alpha=args.alpha,
            beta=args.beta,
            optimizer=None,
            clip_grad=None,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train: total={tr_tot:.6f} recon={tr_rec:.6f} pred={tr_pred:.6f} | "
            f"val: total={va_tot:.6f} recon={va_rec:.6f} pred={va_pred:.6f}"
        )

        # Track best on validation total
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
                "mode_col": args.mode_col,
                "notes": "Per-mode joint NVMD + CNN-BiLSTM (raw scale)",
            }
            torch.save(ck, os.path.join(args.outdir, f"{args.mode_col}_epoch_{ep:03d}.pt"))

    # Save best
    if best_state is not None:
        out_best = os.path.join(args.outdir, f"{args.mode_col}_best.pt")
        torch.save(
            {
                "epoch": "best",
                "val_best": best_val,
                "model_state": best_state,
                "args": vars(args),
                "mode_col": args.mode_col,
                "notes": "Per-mode joint NVMD + CNN-BiLSTM (raw scale)",
            },
            out_best,
        )
        print(f"Saved best checkpoint for {args.mode_col} → {out_best}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
