#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  # your joint model class


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MinMaxScalerND:
    """
    Simple per-channel min-max scaler for arbitrary ND tensors.
    We'll use it only to denorm per-mode scalar predictions back to raw scale.
    """
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


# -------------------------
# Dataset: ground-truth IMFs + RRP
# -------------------------
class Decomp13Dataset(Dataset):
    """
    For ablation:

    We ONLY care about:
      - imf_win_norm: (K,L) per-sample, normalized per IMF
      - y: raw RRP(t+L)

    x_series is kept for compatibility but unused (you can drop it in the future).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],   # 13 IMF columns
        seq_len: int = 9,
        target_col: str = "RRP",
        imf_mins: np.ndarray | None = None,
        imf_maxs: np.ndarray | None = None,
    ):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)   # 13
        self.L = seq_len

        # ----- raw IMFs: (T,13) -> (13,T)
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)
        self.imfs = torch.tensor(imfs_np).transpose(0, 1)   # (13,T)

        # raw target series (RRP)
        rrp_np = df[target_col].to_numpy(dtype=np.float32)
        self.rrp = torch.tensor(rrp_np, dtype=torch.float32)   # (T,)
        T = len(self.rrp)

        #  per-IMF min/max (for normalization)
        if imf_mins is None or imf_maxs is None:
            imf_mins = self.imfs.min(dim=1).values.numpy()
            imf_maxs = self.imfs.max(dim=1).values.numpy()
        self.imf_mins = np.asarray(imf_mins)
        self.imf_maxs = np.asarray(imf_maxs)

        mins = torch.tensor(self.imf_mins).unsqueeze(1)  # (13,1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs - mins) / rngs       # (13,T) in [0,1] ideally

        # We still define N like before: all windows that have L history + 1-step ahead target
        self.N = T - self.L - 1
        if self.N < 0:
            self.N = 0

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L

        # 13 normalized IMF windows (13,L)
        imf_win = self.imfs_norm[:, i:i+L]                # (13,L)

        # raw target RRP(t+L)
        y = self.rrp[i+L]                                 # scalar

        # For compatibility with old code, return dummy x: shape (1,L) (not used)
        x_dummy = torch.zeros(1, L, dtype=torch.float32)

        return x_dummy, imf_win, y.unsqueeze(0)           # (1,L), (13,L), (1,)

    def scalers(self):
        return dict(imf_mins=self.imf_mins, imf_maxs=self.imf_maxs)


def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)]
    cols.append("Residual")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 13-series setting: {missing}")
    return cols


# -------------------------
# Predictor-only train/eval
# -------------------------
def predictor_params(model):
    """Yield only predictor parameters (no decomposer)."""
    for p in model.predictors.parameters():
        if p.requires_grad:
            yield p


def train_or_eval_predictor_only(
    model,
    loader,
    device,
    imf_mins,
    imf_maxs,
    optimizer=None,
    clip_grad=None,
):
    """
    Ablation:

    - Ignore decomposer completely.
    - Use *ground-truth* normalized IMFs imfs_true_norm as input to predictors.
    - Predict per-mode next-step (normalized), denorm per-mode using imf_mins/maxs,
      sum across modes in RAW scale → y_pred (B,1).
    - Loss = L1(y_pred, y_true_raw).

    No decomposer loss, no gradients through decomposer.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total, sum_loss = 0, 0.0

    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1
    )

    for _, imfs_true_norm, yb in loader:
        imfs_true_norm = imfs_true_norm.to(device)  # (B,K,L)
        yb = yb.to(device)                          # (B,1)

        B, K, L = imfs_true_norm.shape

        # --- Predictor forward with GT IMFs ---
        y_list = []
        for k in range(K):
            # each predictor expects (B, 1, L)
            yk_norm = model.predictors[k](imfs_true_norm[:, k:k+1, :])  # (B,1)
            y_list.append(yk_norm)
        y_modes_norm = torch.cat(y_list, dim=1)  # (B,K)

        # Denorm per-mode scalars to raw IMF scale
        y_modes_raw = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
        y_pred = y_modes_raw.sum(dim=1, keepdim=True)  # (B,1) raw RRP-like

        loss = F.l1_loss(y_pred, yb)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(list(predictor_params(model)), clip_grad)
            optimizer.step()

        bs = yb.size(0)
        total += bs
        sum_loss += loss.item() * bs

    return sum_loss / max(total, 1)


@torch.no_grad()
def eval_predictor_only_metrics(model, loader, device, imf_mins, imf_maxs):
    """
    Same as train_or_eval_predictor_only but returns MAE & RMSE over all predictions.
    """
    model.eval()

    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1
    )

    y_true_all, y_pred_all = [], []

    for _, imfs_true_norm, yb in loader:
        imfs_true_norm = imfs_true_norm.to(device)  # (B,K,L)
        yb = yb.to(device)                          # (B,1)

        B, K, L = imfs_true_norm.shape

        y_list = []
        for k in range(K):
            yk_norm = model.predictors[k](imfs_true_norm[:, k:k+1, :])  # (B,1)
            y_list.append(yk_norm)
        y_modes_norm = torch.cat(y_list, dim=1)  # (B,K)

        y_modes_raw = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
        y_pred = y_modes_raw.sum(dim=1, keepdim=True)  # (B,1)

        y_true_all.append(yb)
        y_pred_all.append(y_pred)

    y_true_all = torch.cat(y_true_all, dim=0).view(-1)
    y_pred_all = torch.cat(y_pred_all, dim=0).view(-1)

    mae  = torch.mean(torch.abs(y_pred_all - y_true_all)).item()
    rmse = torch.sqrt(torch.mean((y_pred_all - y_true_all) ** 2)).item()
    return mae, rmse


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--seq-len", type=int, default=9)

    # Model
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)

    # Train
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    # Logs / save
    ap.add_argument("--outdir", type=str, default="./runs_predictor_only_ablation")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataframes
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    # Columns for 13 IMFs
    decomp_cols = build_default_13(df_tr)

    # Datasets
    trds = Decomp13Dataset(
        df=df_tr,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col="RRP"
    )
    sc = trds.scalers()

    vads = Decomp13Dataset(
        df=df_va,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col="RRP",
        imf_mins=sc["imf_mins"],
        imf_maxs=sc["imf_maxs"],
    )

    # Loaders
    pin = (device == "cuda")
    trdl = DataLoader(trds, batch_size=args.batch, shuffle=True,
                      num_workers=args.num_workers, pin_memory=pin)
    vadl = DataLoader(vads, batch_size=args.batch, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pin)

    # Model: we ignore decomposer; we only use predictors
    K = len(decomp_cols)
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len,
        K=K,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        freeze_decomposer=True,  # just to be explicit
    ).to(device)

    # Freeze decomposer params (not used anyway)
    for p in model.decomposer.parameters():
        p.requires_grad = False

    # Optimizer: only predictor params
    opt = torch.optim.Adam(predictor_params(model), lr=args.lr)

    best_rmse = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss = train_or_eval_predictor_only(
            model, trdl, device,
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
            optimizer=opt,
            clip_grad=args.clip_grad,
        )

        va_mae, va_rmse = eval_predictor_only_metrics(
            model, vadl, device,
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
        )

        print(f"[Epoch {ep:02d}] train_pred_L1={tr_loss:.6f} | "
              f"val MAE={va_mae:.4f} RMSE={va_rmse:.4f}")

        if va_rmse < best_rmse:
            best_rmse = va_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # checkpoint every epoch
        ck = {
            "epoch": ep,
            "val_best_rmse": best_rmse,
            "model_state": model.state_dict(),
            "args": vars(args),
            "scalers": sc,
            "decomp_cols": decomp_cols,
        }
        torch.save(ck, os.path.join(args.outdir, f"epoch_{ep:02d}.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({
            "epoch": "best",
            "val_best_rmse": best_rmse,
            "model_state": best_state,
            "args": vars(args),
            "scalers": sc,
            "decomp_cols": decomp_cols,
        }, os.path.join(args.outdir, "best.pt"))
        print(f"Saved best predictor-only checkpoint with RMSE={best_rmse:.4f}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
