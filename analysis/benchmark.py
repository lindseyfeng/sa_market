#!/usr/bin/env python3
"""
Predictor-agnostic decomposition benchmark.

Trains IDENTICAL predictor architectures on mode CSVs from different
decomposition methods.  If NVMD modes consistently yield better
predictions across ALL predictor types, the decomposition itself is
better — not just the predictor.

Predictors (from simplest to most complex):
  1. Linear   — single linear layer
  2. MLP      — 2-hidden-layer feedforward
  3. LSTM     — 2-layer sequence model
  4. CNN-BiLSTM — the MRC_BiLSTM from the paper baseline

Usage:
    python benchmark.py \
        --method "Global VMD (leaky)" \
            VMD_modes_with_residual_2018_2021.csv \
            VMD_modes_with_residual_2021_2022.csv \
        --method "NVMD v2" \
            nvmd_modes_2018_2021.csv \
            nvmd_modes_2021_2022.csv \
        --window 9 --epochs 30
"""

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ===================================================================
# Reproducibility
# ===================================================================

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ===================================================================
# Dataset
# ===================================================================

def detect_mode_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for i in range(1, 100):
        c = f"Mode_{i}"
        if c in df.columns:
            cols.append(c)
        else:
            break
    if "Residual" in df.columns:
        cols.append("Residual")
    return cols


class ModeWindowDataset(Dataset):
    """Window of K mode values over W steps → predict next RRP.

    All modes and targets are z-score normalised using training-set
    statistics (passed via *_mean / *_std for the val set).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode_cols: list[str],
        target_col: str,
        window: int,
        mode_means: np.ndarray | None = None,
        mode_stds: np.ndarray | None = None,
        target_mean: float | None = None,
        target_std: float | None = None,
        horizon: int = 1,
    ):
        modes = df[mode_cols].to_numpy(dtype=np.float32)
        target = df[target_col].to_numpy(dtype=np.float32)

        if mode_means is None:
            self.mode_means = modes.mean(axis=0)
            self.mode_stds = modes.std(axis=0) + 1e-8
            self.target_mean = float(target.mean())
            self.target_std = float(target.std()) + 1e-8
        else:
            self.mode_means = mode_means
            self.mode_stds = mode_stds
            self.target_mean = target_mean
            self.target_std = target_std

        modes_n = (modes - self.mode_means) / self.mode_stds
        target_n = (target - self.target_mean) / self.target_std

        self.modes = torch.from_numpy(modes_n)
        self.target = torch.from_numpy(target_n)
        self.W = window
        self.K = len(mode_cols)
        # horizon h predicts h steps past the window.  h=1 is the original
        # next-step task, which is saturated: persistence scores 14.40 there and
        # causal VMD + LSTM scores 14.43, i.e. worse than doing nothing.
        self.h = horizon
        self.N = max(0, len(target) - self.W - (horizon - 1))

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.modes[i : i + self.W].T                 # (K, W)
        y = self.target[i + self.W + self.h - 1].unsqueeze(0)   # (1,)
        return x, y


# ===================================================================
# Predictors  (all:  (B, K, W) → (B, 1))
# ===================================================================

class LinearPredictor(nn.Module):
    def __init__(self, K, W):
        super().__init__()
        self.fc = nn.Linear(K * W, 1)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MLPPredictor(nn.Module):
    def __init__(self, K, W, hidden=256):
        super().__init__()
        d = K * W
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class LSTMPredictor(nn.Module):
    def __init__(self, K, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(K, hidden, layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # (B, K, W) → (B, W, K)
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.fc(out[:, -1])


class CNNBiLSTMPredictor(nn.Module):
    """Wraps the baseline MRC_BiLSTM with multi-channel (K modes) input."""

    def __init__(self, K, W, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        from train.baseline.cnn_bilstm import MRC_BiLSTM
        self.model = MRC_BiLSTM(
            input_dim=K, seq_len=W,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            bidirectional=True,
        )

    def forward(self, x):
        return self.model(x)


# ===================================================================
# Training loop (identical for every predictor / method pair)
# ===================================================================

def train_and_eval(
    model: nn.Module,
    tr_dl: DataLoader,
    va_dl: DataLoader,
    device: str,
    target_mean: float,
    target_std: float,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 7,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5,
    )

    best_val_mae = float("inf")
    best_state = None
    stale = 0

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # ---- eval ----
        metrics = _eval(model, va_dl, device, target_mean, target_std)
        scheduler.step(metrics["mae"])

        if metrics["mae"] < best_val_mae:
            best_val_mae = metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                break

    # ---- final eval with best weights ----
    if best_state is not None:
        model.load_state_dict(best_state)
    val_m = _eval(model, va_dl, device, target_mean, target_std)
    tr_m = _eval(model, tr_dl, device, target_mean, target_std)
    return tr_m, val_m


@torch.no_grad()
def _eval(model, loader, device, target_mean, target_std):
    model.eval()
    sum_ae = sum_se = sum_ape = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        p_raw = pred * target_std + target_mean
        y_raw = y * target_std + target_mean
        sum_ae += (p_raw - y_raw).abs().sum().item()
        sum_se += ((p_raw - y_raw) ** 2).sum().item()
        denom = y_raw.abs().clamp(min=1.0)
        sum_ape += ((p_raw - y_raw).abs() / denom).sum().item()
        n += x.size(0)
    d = max(n, 1)
    return {
        "mae": sum_ae / d,
        "rmse": math.sqrt(sum_se / d),
        "mape": sum_ape / d * 100,
    }


# ===================================================================
# Decomposition quality metrics
# ===================================================================

def decomposition_quality(df, mode_cols, target_col="RRP"):
    """Compute intrinsic quality metrics of a decomposition."""
    modes = df[mode_cols].to_numpy(dtype=np.float64)
    rrp = df[target_col].to_numpy(dtype=np.float64)
    K = modes.shape[1]

    # Reconstruction error
    recon = modes.sum(axis=1)
    recon_err = np.abs(recon - rrp).mean()

    # Temporal smoothness: avg L2 norm of first difference per mode
    diffs = np.diff(modes, axis=0)
    smoothness = np.sqrt((diffs ** 2).mean(axis=0)).mean()

    # Orthogonality: avg |correlation| between all mode pairs
    corr_mat = np.corrcoef(modes.T)
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    mean_abs_corr = np.abs(corr_mat[mask]).mean()

    # Per-mode predictability: 1 − (prediction error / mode std)
    # using naive persistence forecast (next = current)
    persistence_err = np.abs(diffs).mean(axis=0)
    mode_std = modes.std(axis=0) + 1e-8
    predictability = (1.0 - persistence_err / mode_std).mean()

    return {
        "recon_err": recon_err,
        "smoothness": smoothness,
        "ortho": mean_abs_corr,
        "predictability": predictability,
    }


# ===================================================================
# Pretty printing
# ===================================================================

def print_header(title: str):
    w = 72
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def print_quality_table(quality: dict[str, dict]):
    print_header("Decomposition Quality  (lower smoothness/ortho = better,"
                 "\n  higher predictability = better)")
    fmt = "  {:<25s} {:>10s} {:>10s} {:>10s} {:>12s}"
    print(fmt.format("Method", "Smooth", "Ortho", "ReconErr", "Predict"))
    print("  " + "-" * 69)
    for name, q in quality.items():
        print(fmt.format(
            name,
            f"{q['smoothness']:.4f}",
            f"{q['ortho']:.4f}",
            f"{q['recon_err']:.4f}",
            f"{q['predictability']:.4f}",
        ))


def print_forecast_table(results: dict):
    print_header("Forecasting Performance — validation set (AUD/MWh)")

    predictors = list(next(iter(results.values())).keys())
    header = "  {:<25s}" + " {:>12s} {:>12s} {:>10s}" * len(predictors)
    sub = []
    for p in predictors:
        sub += [f"{p} MAE", f"{p} RMSE", f"{p} MAPE%"]
    print(header.format("Method", *sub))
    print("  " + "-" * (25 + 34 * len(predictors)))

    for method, preds in results.items():
        vals = []
        for p in predictors:
            m = preds[p]["val"]
            vals += [f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['mape']:.1f}"]
        row = "  {:<25s}" + " {:>12s}" * len(vals)
        print(row.format(method, *vals))


def print_leakage_table(results: dict):
    """Show train-val MAE gap as a proxy for data leakage."""
    print_header("Leakage Indicator  (Train MAE − Val MAE gap)")
    print("  Negative gap = normal generalisation loss.")
    print("  Gap close to zero or positive with global VMD "
          "suggests modes leak future info.\n")
    predictors = list(next(iter(results.values())).keys())
    header = "  {:<25s}" + " {:>14s}" * len(predictors)
    print(header.format("Method", *predictors))
    print("  " + "-" * (25 + 14 * len(predictors)))

    for method, preds in results.items():
        vals = []
        for p in predictors:
            tr_mae = preds[p]["train"]["mae"]
            va_mae = preds[p]["val"]["mae"]
            gap = tr_mae - va_mae
            vals.append(f"{gap:+.2f}")
        row = "  {:<25s}" + " {:>14s}" * len(vals)
        print(row.format(method, *vals))


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Predictor-agnostic decomposition benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--method", nargs=3, action="append",
        metavar=("NAME", "TRAIN_CSV", "VAL_CSV"),
        required=True,
        help='Decomposition method: "Name" train.csv val.csv (repeat)',
    )
    ap.add_argument("--window", type=int, default=9,
                    help="Prediction window (number of past mode steps)")
    ap.add_argument("--target-col", default="RRP")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cnn-bilstm", action="store_true",
                    help="Skip CNN-BiLSTM predictor (saves time)")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = args.window

    methods = {}
    for name, tr_csv, va_csv in args.method:
        methods[name] = (tr_csv, va_csv)

    # ---- 1. Decomposition quality metrics ----
    quality = {}
    for name, (tr_csv, va_csv) in methods.items():
        df_va = pd.read_csv(va_csv)
        mode_cols = detect_mode_cols(df_va)
        quality[name] = decomposition_quality(
            df_va, mode_cols, args.target_col
        )
    print_quality_table(quality)

    # ---- 2. Forecast comparison ----
    predictor_factories = {
        "Linear": lambda K, W: LinearPredictor(K, W),
        "MLP":    lambda K, W: MLPPredictor(K, W),
        "LSTM":   lambda K, W: LSTMPredictor(K),
    }
    if not args.skip_cnn_bilstm:
        try:
            from train.baseline.cnn_bilstm import MRC_BiLSTM  # noqa: F401
            predictor_factories["CNN-BiLSTM"] = (
                lambda K, W: CNNBiLSTMPredictor(K, W)
            )
        except ImportError:
            print("  (CNN-BiLSTM skipped — could not import MRC_BiLSTM)")

    # results[method_name][predictor_name] = {"train": {...}, "val": {...}}
    results: dict[str, dict] = defaultdict(dict)

    for mname, (tr_csv, va_csv) in methods.items():
        df_tr = pd.read_csv(tr_csv)
        df_va = pd.read_csv(va_csv)
        mode_cols = detect_mode_cols(df_tr)
        K = len(mode_cols)

        tr_ds = ModeWindowDataset(df_tr, mode_cols, args.target_col, W)
        va_ds = ModeWindowDataset(
            df_va, mode_cols, args.target_col, W,
            mode_means=tr_ds.mode_means, mode_stds=tr_ds.mode_stds,
            target_mean=tr_ds.target_mean, target_std=tr_ds.target_std,
        )
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

        for pname, factory in predictor_factories.items():
            set_seed(args.seed)
            model = factory(K, W).to(device)
            n_p = sum(p.numel() for p in model.parameters())

            t0 = time.time()
            tr_m, va_m = train_and_eval(
                model, tr_dl, va_dl, device,
                target_mean=tr_ds.target_mean,
                target_std=tr_ds.target_std,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
            )
            elapsed = time.time() - t0
            results[mname][pname] = {"train": tr_m, "val": va_m}

            print(f"  [{mname}] {pname:12s} ({n_p:>8,} params) "
                  f"val MAE={va_m['mae']:.2f}  RMSE={va_m['rmse']:.2f}  "
                  f"({elapsed:.1f}s)")

    # ---- 3. Print results ----
    print_forecast_table(results)
    print_leakage_table(results)

    # ---- 4. Summary ----
    print_header("Summary")
    method_names = list(results.keys())
    pnames = list(next(iter(results.values())).keys())
    win_count = defaultdict(int)
    for p in pnames:
        best_method = min(method_names, key=lambda m: results[m][p]["val"]["mae"])
        win_count[best_method] += 1
    total = len(pnames)
    for m, w in sorted(win_count.items(), key=lambda x: -x[1]):
        print(f"  {m}: best in {w}/{total} predictors")
    print()


if __name__ == "__main__":
    main()
