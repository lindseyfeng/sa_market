#!/usr/bin/env python3
# eval_nvmd_mrc_bilstm.py
# Usage:
#   python3 eval_nvmd_mrc_bilstm.py \
#     --csv VMD_modes_with_residual_2021_2022_with_EWT.csv \
#     --ckpt ./runs_nvmd_mrc_bilstm/best.pt \
#     --seq-len 8 \
#     --batch 256 \
#     --alpha 0.01 \
#     --beta 2.0 \
#     --out-csv eval_preds.csv

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  # <-- your class

# -------------------------
# Utilities / Helpers
# -------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MinMaxScalerND:
    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor, channel_axis: int = 1, eps: float = 1e-12):
        assert mins.numel() == maxs.numel()
        self.mins = mins
        self.maxs = maxs
        self.axis = channel_axis
        self.eps  = eps

    def _view_shape(self, x: torch.Tensor):
        shape = [1] * x.ndim
        shape[self.axis] = self.mins.numel()
        return shape

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        mins = self.mins.view(*self._view_shape(x))
        rng  = (self.maxs - self.mins).view(*self._view_shape(x)).clamp_min(self.eps)
        return (x - mins) / rng

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        mins = self.mins.view(*self._view_shape(x))
        rng  = (self.maxs - self.mins).view(*self._view_shape(x)).clamp_min(self.eps)
        return x * rng + mins

class Decomp13Dataset(Dataset):
    """
    13 decomposition signals: Mode_1..Mode_12 + Residual
    x_norm  = sum of per-IMF normalized signals
    y_true  = raw RRP value (scalar, next step after window)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],
        seq_len: int = 8,
        target_col: str = "RRP",
        imf_mins: np.ndarray | None = None,
        imf_maxs: np.ndarray | None = None,
    ):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)
        self.L = seq_len

        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)   # (T, K)
        self.imfs = torch.tensor(imfs_np).transpose(0,1)       # (K, T)

        target_np = df[target_col].to_numpy(dtype=np.float32)  # (T,)
        self.target = torch.tensor(target_np, dtype=torch.float32)
        T = len(self.target)

        if imf_mins is None or imf_maxs is None:
            imf_mins = self.imfs.min(dim=1).values.numpy()
            imf_maxs = self.imfs.max(dim=1).values.numpy()
        self.imf_mins = np.asarray(imf_mins)
        self.imf_maxs = np.asarray(imf_maxs)

        mins = torch.tensor(self.imf_mins).unsqueeze(1)  # (K,1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs - mins) / rngs       # (K,T)

        self.signal_norm = self.imfs_norm.sum(dim=0)     # (T,)
        self.N = max(0, T - self.L - 1)

    def __len__(self): return self.N

    def __getitem__(self, i):
        L = self.L
        x  = self.signal_norm[i:i+L].unsqueeze(0)  # (1,L)
        imf_win = self.imfs_norm[:, i:i+L]         # (K,L)
        y  = self.target[i+L]                      # scalar
        return x, imf_win, y.unsqueeze(0)          # (1,L), (K,L), (1,)

    def scalers(self):
        return dict(imf_mins=self.imf_mins, imf_maxs=self.imf_maxs)

def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 13-series setting: {missing}")
    return cols

@torch.no_grad()
@torch.no_grad()
def eval_epoch(model, loader, device, imf_mins, imf_maxs):
    model.eval()

    # only needed if you also want a recon metric
    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1
    )

    y_true_all, y_pred_all = [], []
    recon_mae_accum, n_batches = 0.0, 0

    for xb, imfs_true_norm, yb in loader:
        xb = xb.to(device)
        imfs_true_norm = imfs_true_norm.to(device)
        yb = yb.to(device)                    # (B,1)

        imfs_pred_norm, y_pred = model(xb)    # (B,K,L), (B,1)

        # collect for MAE/RMSE on the same thing you trained
        y_true_all.append(yb)
        y_pred_all.append(y_pred)

        # (optional) recon check on raw scale — will NOT match pred metrics
        imfs_pred = imf_scaler.denorm(imfs_pred_norm)
        imfs_true = imf_scaler.denorm(imfs_true_norm)
        sig_pred  = imfs_pred.sum(dim=1)          # (B,L)
        sig_true  = imfs_true.sum(dim=1)          # (B,L)
        recon_mae_accum += torch.mean(torch.abs(sig_pred - sig_true)).item()
        n_batches += 1

    y_true_all = torch.cat(y_true_all, dim=0).view(-1)
    y_pred_all = torch.cat(y_pred_all, dim=0).view(-1)

    mae  = torch.mean(torch.abs(y_pred_all - y_true_all)).item()
    rmse = torch.sqrt(torch.mean((y_pred_all - y_true_all)**2)).item()

    # optional: average reconstruction MAE across batches
    recon_mae = recon_mae_accum / max(n_batches, 1)

    return mae, rmse  # (optionally also return recon_mae)




def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--csv",  default="VMD_modes_with_residual_2021_2022.csv", help="CSV with Mode_1..Mode_12, Residual, RRP")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--target-col", type=str, default="RRP")
    # Model / 
    ap.add_argument("--ckpt", default = "./runs_nvmd_mrc_bilstm/best.pt", help="Path to checkpoint .pt from training")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Loss weights
    ap.add_argument("--alpha", type=float, default=1)
    ap.add_argument("--beta",  type=float, default=1)
    # Loader
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    # Output
    ap.add_argument("--out-csv", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    ck_args = ck.get("args", {})
    ck_scalers = ck.get("scalers", {})
    decomp_cols = ck.get("decomp_cols", None)

    # Dataframe
    df = pd.read_csv(args.csv)

    # Columns
    if decomp_cols is None:
        decomp_cols = build_default_13(df)

    # IMF scalers: prefer those saved in ckpt for consistency with training
    imf_mins = ck_scalers.get("imf_mins", None)
    imf_maxs = ck_scalers.get("imf_maxs", None)

    # Dataset / Loader
    ds = Decomp13Dataset(
        df=df,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col=args.target_col,
        imf_mins=imf_mins,
        imf_maxs=imf_maxs
    )
    sc = ds.scalers()  # actual mins/maxs used (from ckpt or computed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # Build model (match training hyperparams if present)
    K = len(decomp_cols)
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len,
        K=K,
        base=ck_args.get("base", 32),
        lstm_hidden=ck_args.get("lstm_hidden", 128),
        lstm_layers=ck_args.get("lstm_layers", 3),
        bidirectional=ck_args.get("bidirectional", True),
        freeze_decomposer=ck_args.get("freeze_decomposer", False)
    ).to(device)

    # Load weights
    state = ck["model_state"]
    model.load_state_dict(state, strict=True)

    # Evaluate
    mae, nmse = eval_epoch(
        model=model,
        loader=dl,
        device=device,
        imf_mins=sc["imf_mins"],
        imf_maxs=sc["imf_maxs"],
    )

    print("mae: ", mae)
    print("rmse: ", nmse)


if __name__ == "__main__":
    main()
