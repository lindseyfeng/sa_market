#!/usr/bin/env python3
# train_nvmd_cnn_bilstm_mode.py

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  # per-mode joint model (K=1 inside)
def denorm(norm_val, vmin, vmax):
    return norm_val * (vmax - vmin) + vmin




def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NVMDModeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode_col: str,           # e.g. "Mode_1" or "Residual"
        rrp_col: str = "RRP",
        seq_len: int = 1024,
        x_mode: str = "raw",     # "raw" or "sum"
        all_decomp_cols: list[str] | None = None,  # only needed if x_mode == "sum"
        # --- NEW: 可选传入 scaler，让 val 复用 train 的 ---
        x_min: float | None = None,
        x_max: float | None = None,
        imf_min: float | None = None,
        imf_max: float | None = None,
    ):
        super().__init__()
        self.L = seq_len
        self.mode_col = mode_col
        self.x_mode = x_mode

        if mode_col not in df.columns:
            raise ValueError(f"mode_col '{mode_col}' not in dataframe.")

        imf_np = df[mode_col].to_numpy(dtype=np.float32)   # (T,)
        imf_t = torch.from_numpy(imf_np).contiguous()      # (T,)

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")
        rrp_np = df[rrp_col].to_numpy(dtype=np.float32)    # (T,)
        rrp_t = torch.from_numpy(rrp_np).contiguous()      # (T,)
        T = rrp_t.shape[0]

        if x_mode == "raw":
            x_series = rrp_t                               # (T,)
        elif x_mode == "sum":
            if all_decomp_cols is None:
                raise ValueError("all_decomp_cols must be provided when x_mode='sum'")
            imfs_np_all = df[all_decomp_cols].to_numpy(dtype=np.float32)  # (T,K_all)
            imfs_all = torch.from_numpy(imfs_np_all).transpose(0, 1).contiguous()  # (K_all,T)
            x_series = imfs_all.sum(dim=0)                # (T,)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum'")

        eps = 1e-8

        if x_min is None or x_max is None:
            self.x_min = float(x_series.min().item())
            self.x_max = float(x_series.max().item())
        else:
            self.x_min = float(x_min)
            self.x_max = float(x_max)
        self.x_range = self.x_max - self.x_min if (self.x_max > self.x_min) else eps

        if imf_min is None or imf_max is None:
            self.imf_min = float(imf_t.min().item())
            self.imf_max = float(imf_t.max().item())
        else:
            self.imf_min = float(imf_min)
            self.imf_max = float(imf_max)
        self.imf_range = self.imf_max - self.imf_min if (self.imf_max > self.imf_min) else eps

        self.x_series = x_series
        self.imf = imf_t
        self.rrp = rrp_t

        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L     

        imf_win_raw  = self.imf[i:i+L]       # (L,)
        imf_next_raw = self.imf[i+L]      


        imf_win_norm  = (imf_win_raw  - self.imf_min) / self.imf_range
        imf_next_norm = (imf_next_raw - self.imf_min) / self.imf_range

        x_norm      = x_norm.unsqueeze(0)        # (1, L)
        y_norm      = y_norm.unsqueeze(0)        # (1,)
        imf_win_norm  = imf_win_norm.unsqueeze(0)   # (1, L)
        imf_next_norm = imf_next_norm.unsqueeze(0)  # (1,)

        return x_raw_seq, imf_win_norm, imf_next_norm, y_raw


# -------------------------
# Utils
# -------------------------
def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return cols


def denorm(v, vmin, vmax):
    return v * (vmax - vmin) + vmin

def train_or_eval_epoch_joint(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    alpha: float,
    beta: float,
    x_min: float,
    x_max: float,
    imf_min: float,
    imf_max: float,
    optimizer=None,
    clip_grad: float | None = None,
):
    """
    alpha: weight for RRP prediction loss (normalized)
    beta:  weight for IMF-next prediction loss (normalized)
    """

    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    total_recon_sum = 0.0
    total_pred_sum = 0.0
    n_samples = 0

    # --- real-scale metrics ---
    real_rrp_mae_sum = 0.0
    real_rrp_rmse_sum = 0.0

    for x_norm, imf_win, imf_next, y_norm in loader:
        x_norm   = x_norm.to(device)
        y_norm   = y_norm.to(device)
        imf_next = imf_next.to(device)

        # -------- forward --------
        imf_pred_norm, imf_next_hat_norm, y_pred_norm = model(x_norm)

        imf_pred = denorm(imf_pred_norm, imf_min, imf_max)
        imf_next_hat = denorm(imf_next_hat_norm,   imf_min, imf_max)

        loss_rrp  = F.l1_loss(y_pred_norm, y_norm)
        loss_imf  = F.l1_loss(imf_next_hat_norm, imf_next)

        loss = alpha * loss_rrp + beta * loss_imf

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = x_norm.size(0)
        n_samples += bs

        total_loss_sum  += loss.item() * bs
        total_recon_sum += loss_imf.item() * bs
        total_pred_sum  += loss_rrp.item() * bs

        # -------- eval mode: compute real-scale MAE/RMSE --------
        if not is_train:
            # denorm RRP


            abs_err = (y_pred_norm - y_true_real).abs()
            sq_err  = (y_pred_norm - y_true_real) ** 2

            real_rrp_mae_sum  += abs_err.mean().item() * bs
            real_rrp_rmse_sum += sq_err.mean().item() * bs

    denom = max(n_samples, 1)

    if is_train:
        return (
            total_loss_sum  / denom,
            total_recon_sum / denom,
            total_pred_sum  / denom,
        )
    else:
        real_mae  = real_rrp_mae_sum / denom
        real_rmse = (real_rrp_rmse_sum / denom) ** 0.5

        return (
            total_loss_sum  / denom,
            total_recon_sum / denom,
            total_pred_sum  / denom,
            real_mae,
            real_rmse,
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
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)

    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="weight for RRP prediction loss")
    ap.add_argument("--beta",  type=float, default=0.1,
                    help="weight for IMF-next prediction loss")
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
        x_min=tr_ds.x_min,
        x_max=tr_ds.x_max,
        imf_min=tr_ds.imf_min,
        imf_max=tr_ds.imf_max,
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
        if ep > 10:
            args.alpha, args.beta = 1.0, 0.1


        tr_tot, tr_imf, tr_rrp = train_or_eval_epoch_joint(
            model, tr_dl, device,
            alpha=args.alpha, beta=args.beta,
            x_min=tr_ds.x_min, x_max=tr_ds.x_max,
            imf_min=tr_ds.imf_min, imf_max=tr_ds.imf_max,
            optimizer=opt
        )

        va_tot, va_imf, va_rrp, va_mae_real, va_rmse_real = train_or_eval_epoch_joint(
            model, va_dl, device,
            alpha=args.alpha, beta=args.beta,
            x_min=tr_ds.x_min, x_max=tr_ds.x_max,
            imf_min=tr_ds.imf_min, imf_max=tr_ds.imf_max,
            optimizer=None
        )



        print(
            f"[Epoch {ep:03d}] "
            f"train: total={tr_tot:.6f} recon={tr_imf:.6f} pred={tr_rrp:.6f} | "
            f"val: total={va_tot:.6f} recon={va_imf:.6f} pred={va_rrp:.6f}"
            f"val real-scale MAE={va_mae_real:.4f}, RMSE={va_rmse_real:.4f}"
        )

        if va_imf < best_val:
            best_val = va_imf
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (args.save_every > 0) and (ep % args.save_every == 0):
            ck = {
                "epoch": ep,
                "val_best": best_val,
                "model_state": model.state_dict(),
                "args": vars(args),
                "mode_col": args.mode_col,
                
                "scalers": {
                    "x_min": tr_ds.x_min,
                    "x_max": tr_ds.x_max,
                    "imf_min": tr_ds.imf_min,
                    "imf_max": tr_ds.imf_max,
                },
                "notes": "Per-mode joint NVMD + CNN-BiLSTM (min-max normalized)",
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
                "scalers": {
                    "x_min": tr_ds.x_min,
                    "x_max": tr_ds.x_max,
                    "imf_min": tr_ds.imf_min,
                    "imf_max": tr_ds.imf_max,
                },
                "notes": "Per-mode joint NVMD + CNN-BiLSTM (min-max normalized)",
            },
            out_best,
        )
        print(f"Saved best checkpoint for {args.mode_col} → {out_best}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
