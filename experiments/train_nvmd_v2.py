#!/usr/bin/env python3
"""
End-to-end training for NVMD v2 + per-mode CNN-BiLSTM forecaster.

Key difference from train_nvmd_cnn_bilstm.py:
  - No VMD targets.  The decomposition is learned purely from the
    forecasting objective + self-supervised regularisation.
  - Regularisation weights decay over training so early epochs build
    a structurally sound decomposition, later epochs focus on forecast.
  - Reports MAE / RMSE in denormalised AUD/MWh for direct comparison
    with the VMD-based baseline.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.nvmd_v2 import NVMDForecaster


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class RRPWindowDataset(Dataset):
    """Sliding-window dataset over the RRP series.

    Normalisation is global z-score (mean/std from training set).
    Each sample: x = window of L normalised values, y = next normalised value.
    """

    def __init__(
        self,
        rrp: np.ndarray,
        seq_len: int,
        mean: float | None = None,
        std: float | None = None,
    ):
        if mean is None:
            self.mean = float(rrp.mean())
            self.std = float(rrp.std())
        else:
            self.mean = mean
            self.std = std

        rrp_norm = (rrp - self.mean) / (self.std + 1e-8)
        self.data = torch.from_numpy(rrp_norm.astype(np.float32))
        self.L = seq_len
        self.N = max(0, len(self.data) - self.L)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.data[i : i + self.L].unsqueeze(0)    # (1, L)
        y = self.data[i + self.L].unsqueeze(0)          # (1,)
        return x, y


def load_rrp(csv_path: str, rrp_col: str = "RRP",
             date_col: str = "SETTLEMENTDATE") -> np.ndarray:
    df = pd.read_csv(csv_path)
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[rrp_col])
    return df[rrp_col].to_numpy(dtype=np.float32)


# -----------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------

def reg_weight(initial: float, decay: float, epoch: int, total: int) -> float:
    """Linear decay from *initial* to *initial * decay* over training."""
    progress = epoch / max(total - 1, 1)
    return initial * (1.0 - progress * (1.0 - decay))


def train_epoch(
    model: NVMDForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    w_smooth: float,
    w_ortho: float,
    w_collapse: float,
    clip_grad: float = 5.0,
):
    model.train()
    sum_loss = sum_fc = sum_reg = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        imfs, masks, _, y_pred = model(x)

        loss_forecast = F.mse_loss(y_pred, y)

        loss_reg = (
            w_smooth * model.decomposer.smoothness_loss(masks)
            + w_ortho * model.decomposer.orthogonality_loss(imfs)
            + w_collapse * model.decomposer.anti_collapse_loss(imfs)
        )
        loss = loss_forecast + loss_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        bs = x.size(0)
        n += bs
        sum_loss += loss.item() * bs
        sum_fc += loss_forecast.item() * bs
        sum_reg += loss_reg.item() * bs

    d = max(n, 1)
    return sum_loss / d, sum_fc / d, sum_reg / d


@torch.no_grad()
def eval_epoch(
    model: NVMDForecaster,
    loader: DataLoader,
    device: str,
    mean: float,
    std: float,
):
    model.eval()
    sum_mse_norm = sum_mae_raw = sum_mse_raw = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, _, _, y_pred = model(x)

        sum_mse_norm += F.mse_loss(y_pred, y, reduction="sum").item()

        y_pred_raw = y_pred * std + mean
        y_true_raw = y * std + mean
        sum_mae_raw += F.l1_loss(y_pred_raw, y_true_raw, reduction="sum").item()
        sum_mse_raw += F.mse_loss(y_pred_raw, y_true_raw, reduction="sum").item()

        n += x.size(0)

    d = max(n, 1)
    return {
        "mse_norm": sum_mse_norm / d,
        "mae_raw": sum_mae_raw / d,
        "rmse_raw": math.sqrt(sum_mse_raw / d),
    }


@torch.no_grad()
def mode_diagnostics(model: NVMDForecaster, loader: DataLoader, device: str,
                     max_batches: int = 8):
    """Collapse check.

    A forecast-only objective can satisfy the hard reconstruction constraint
    (nvmd_v2.py:147) with mode_1 == x and the rest ~0, which makes the
    decomposition a no-op.  Report where the energy actually sits.
    """
    model.eval()
    energy_sum = None
    corr_sum, corr_n = 0.0, 0

    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        imfs, _ = model.decomposer(x)                  # (B, K, L)

        energy = (imfs ** 2).mean(dim=-1)              # (B, K)
        energy_sum = energy.sum(0) if energy_sum is None else energy_sum + energy.sum(0)

        c = imfs - imfs.mean(dim=-1, keepdim=True)
        c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gram = torch.bmm(c, c.transpose(1, 2)).abs()   # (B, K, K)
        K = gram.shape[1]
        off = ~torch.eye(K, dtype=torch.bool, device=gram.device)
        corr_sum += gram[:, off].mean().item() * x.size(0)
        corr_n += x.size(0)

    share = (energy_sum / energy_sum.sum()).cpu().numpy()
    return {
        "share": share,
        "top1": float(share.max()),
        "effective_modes": float(1.0 / (share ** 2).sum()),   # participation ratio
        "mean_abs_corr": corr_sum / max(corr_n, 1),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end NVMD v2 + CNN-BiLSTM training"
    )

    # Data
    ap.add_argument("--train-csv", default="SA_prices_combined_2018_2021.csv")
    ap.add_argument("--val-csv", default="SA_prices_combined_2021_2022.csv")
    ap.add_argument("--rrp-col", default="RRP")
    ap.add_argument("--seq-len", type=int, default=96,
                    help="window length (96 = 2 days of 30-min data)")

    # Model
    ap.add_argument("--K", type=int, default=13)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--bidirectional", action="store_true", default=True)

    # Training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stop patience (0 = disable)")
    ap.add_argument("--num-workers", type=int, default=0)

    # Regularisation (initial weights; they decay to w * reg_decay)
    ap.add_argument("--w-smooth", type=float, default=0.10)
    ap.add_argument("--w-ortho", type=float, default=0.05)
    ap.add_argument("--w-collapse", type=float, default=0.01)
    ap.add_argument("--reg-decay", type=float, default=0.1,
                    help="multiply reg weights by this at the end of training")

    # Output
    ap.add_argument("--outdir", default="./runs_nvmd_v2")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- data -----------------------------------------------------------
    rrp_tr = load_rrp(args.train_csv, args.rrp_col)
    rrp_va = load_rrp(args.val_csv, args.rrp_col)

    tr_ds = RRPWindowDataset(rrp_tr, args.seq_len)
    va_ds = RRPWindowDataset(rrp_va, args.seq_len,
                             mean=tr_ds.mean, std=tr_ds.std)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(tr_ds)} windows | Val: {len(va_ds)} windows")
    print(f"Train RRP  mean={tr_ds.mean:.2f}  std={tr_ds.std:.2f}")

    # ---- model ----------------------------------------------------------
    model = NVMDForecaster(
        K=args.K,
        signal_len=args.seq_len,
        d_model=args.d_model,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ---- training -------------------------------------------------------
    best_val_mae = float("inf")
    best_state = None
    stale = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        ws = reg_weight(args.w_smooth, args.reg_decay, ep - 1, args.epochs)
        wo = reg_weight(args.w_ortho, args.reg_decay, ep - 1, args.epochs)
        wc = reg_weight(args.w_collapse, args.reg_decay, ep - 1, args.epochs)

        tr_loss, tr_fc, tr_reg = train_epoch(
            model, tr_dl, optimizer, device,
            w_smooth=ws, w_ortho=wo, w_collapse=wc,
            clip_grad=args.clip_grad,
        )
        va = eval_epoch(model, va_dl, device, tr_ds.mean, tr_ds.std)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        dg = mode_diagnostics(model, va_dl, device)
        print(
            f"[{ep:03d}/{args.epochs}] "
            f"tr_loss={tr_loss:.6f} (fc={tr_fc:.6f} reg={tr_reg:.6f}) | "
            f"val MAE={va['mae_raw']:.2f} RMSE={va['rmse_raw']:.2f} AUD/MWh | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )
        print(
            f"      decomp: top1_share={dg['top1']:.3f} "
            f"eff_modes={dg['effective_modes']:.2f}/{args.K} "
            f"|corr|={dg['mean_abs_corr']:.3f}"
            + ("   <-- COLLAPSED, decomposition is a no-op" if dg["top1"] > 0.95 else "")
        )

        improved = va["mae_raw"] < best_val_mae
        if improved:
            best_val_mae = va["mae_raw"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
            print(f"  -> new best val MAE = {best_val_mae:.2f} AUD/MWh")
        else:
            stale += 1

        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_mae": va["mae_raw"],
            "val_rmse": va["rmse_raw"],
            "best_val_mae": best_val_mae,
            "args": vars(args),
            "norm": {"mean": tr_ds.mean, "std": tr_ds.std},
        }
        # Overwrite a single rolling checkpoint -- one per epoch is 214 MB here
        # (17.7M params + Adam state) and filled runs_nvmd_mrc_bilstm with 6.5 GB.
        torch.save(ckpt, os.path.join(args.outdir, "last.pt"))

        if args.patience and stale >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    # ---- save best ------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({
            "model_state": best_state,
            "best_val_mae": best_val_mae,
            "args": vars(args),
            "norm": {"mean": tr_ds.mean, "std": tr_ds.std},
        }, os.path.join(args.outdir, "best.pt"))
        print(f"\nSaved best model: val MAE = {best_val_mae:.2f} AUD/MWh "
              f"→ {os.path.join(args.outdir, 'best.pt')}")
    else:
        print("No best state captured.")


if __name__ == "__main__":
    main()
