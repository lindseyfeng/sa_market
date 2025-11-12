#!/usr/bin/env python3
# eval_nvmd_autoencoder_raw.py
#
# Evaluate a trained NVMD_Autoencoder on RAW (unnormalized) IMFs.
# Metrics:
#   - IMF MAE / RMSE over all (K,L) elements
#   - Sum-consistency MAE: |sum(pred IMFs) - sum(true IMFs)|
#   - Recon-vs-RRP MAE (only if x_mode='raw'): |sum(pred IMFs) - RRP window|
#
# Optionally prints per-mode MAE and saves predictions to .npz.

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_autoencoder import NVMD_Autoencoder  # your decomposer


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
# Dataset (RAW)
# -------------------------
class DecompOnlyDatasetRaw(Dataset):
    """
    For each sliding window i..i+L-1:
      x: (1, L) raw input: RRP (x_mode='raw') or sum of IMFs (x_mode='sum')
      y: (K, L) raw IMFs (Mode_1..Mode_12, Residual)
    """
    def __init__(self, df: pd.DataFrame, decomp_cols: list[str],
                 target_col: str = "RRP", seq_len: int = 1024, x_mode: str = "raw"):
        super().__init__()
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)
        self.L = seq_len
        self.x_mode = x_mode

        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)   # (T, K)
        rrp_np  = df[target_col].to_numpy(dtype=np.float32)     # (T,)

        self.imfs = torch.from_numpy(imfs_np).transpose(0, 1).contiguous()  # (K, T)
        self.rrp  = torch.from_numpy(rrp_np).contiguous()                    # (T,)
        T = self.rrp.shape[0]

        if x_mode == "raw":
            self.x_series = self.rrp
        elif x_mode == "sum":
            self.x_series = self.imfs.sum(dim=0)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum'")

        self.N = max(0, T - self.L + 1)

    def __len__(self): return self.N

    def __getitem__(self, i: int):
        L = self.L
        x = self.x_series[i:i+L].unsqueeze(0)  # (1, L)
        y = self.imfs[:, i:i+L]                # (K, L)
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


@torch.no_grad()
def eval_epoch_raw(model: nn.Module, loader: DataLoader, device: str, want_series: bool = False,
                   x_mode_for_rrp: str = "raw", per_mode: bool = True):
    """
    Returns:
      metrics dict with:
        - imf_mae, imf_rmse
        - sum_mae
        - recon_vs_rrp_mae (if x_mode_for_rrp == 'raw', else None)
        - per_mode_mae: (K,) numpy (if per_mode=True)
      plus (optionally) a dict of stacked series for saving.
    """
    model.eval()

    imf_abs_err_sum = 0.0
    imf_sq_err_sum  = 0.0
    n_elts = 0

    sum_abs_err_sum = 0.0
    n_sum_windows   = 0

    recon_rrp_abs_err_sum = 0.0
    n_rrp_windows         = 0

    # per-mode MAE accumulators
    per_mode_abs = None  # torch tensor shape (K,) on device

    # Optional series collection
    series = {
        "pred_imfs": [],   # list of (B,K,L) → stacked later
        "true_imfs": [],
        "pred_sum": [],    # (B,L)
        "true_sum": [],
        "x_input": []      # (B,1,L) if raw
    } if want_series else None

    for x_win, y_win in loader:
        # x_win: (B, 1, L) already (from dataset)
        # y_win: (B, K, L)
        x_win = x_win.to(device)
        y_win = y_win.to(device)

        # Sanity: ensure 3D for Conv1d
        if x_win.dim() == 2:
            x_win = x_win.unsqueeze(1)
        elif x_win.dim() == 4:
            x_win = x_win.squeeze(2)
        assert x_win.dim() == 3, f"Expected (B,1,L), got {tuple(x_win.shape)}"

        imfs_pred = model(x_win)    # (B, K, L) raw

        # IMF metrics
        abs_err = torch.abs(imfs_pred - y_win)     # (B,K,L)
        sq_err  = (imfs_pred - y_win) ** 2

        imf_abs_err_sum += abs_err.sum().item()
        imf_sq_err_sum  += sq_err.sum().item()
        n_elts          += abs_err.numel()

        # per-mode MAE (if requested)
        if per_mode:
            # reduce over batch and time, keep K
            pm = abs_err.sum(dim=(0, 2))  # (K,)
            per_mode_abs = pm if per_mode_abs is None else per_mode_abs + pm

        # Sum-consistency MAE over time inside each window
        pred_sum = imfs_pred.sum(dim=1)   # (B, L)
        true_sum = y_win.sum(dim=1)       # (B, L)
        sum_abs_err_sum += torch.abs(pred_sum - true_sum).mean(dim=1).sum().item()  # average over L then sum over B
        n_sum_windows   += x_win.size(0)

        # Reconstruction vs RRP window (only meaningful if x was RRP)
        if x_mode_for_rrp == "raw":
            # compare pred_sum to the raw input x_win[:,0,:]
            rrp_win = x_win[:, 0, :]                # (B, L)
            recon_rrp_abs_err_sum += torch.abs(pred_sum - rrp_win).mean(dim=1).sum().item()
            n_rrp_windows += x_win.size(0)

        if want_series:
            series["pred_imfs"].append(imfs_pred.detach().cpu())
            series["true_imfs"].append(y_win.detach().cpu())
            series["pred_sum"].append(pred_sum.detach().cpu())
            series["true_sum"].append(true_sum.detach().cpu())
            if x_mode_for_rrp == "raw":
                series["x_input"].append(x_win.detach().cpu())

    # Aggregate metrics
    imf_mae  = imf_abs_err_sum / max(n_elts, 1)
    imf_rmse = np.sqrt(imf_sq_err_sum / max(n_elts, 1))

    sum_mae  = sum_abs_err_sum / max(n_sum_windows, 1)
    recon_vs_rrp_mae = (recon_rrp_abs_err_sum / max(n_rrp_windows, 1)) if (x_mode_for_rrp == "raw") else None

    per_mode_mae = None
    if per_mode and per_mode_abs is not None:
        # divide by (B*L) total count accumulated per mode
        # We need total number of (B,L) positions processed:
        total_BL = n_elts / (per_mode_abs.numel())  # since n_elts = B*K*L, BL = n_elts/K
        per_mode_mae = (per_mode_abs / max(total_BL, 1)).detach().cpu().numpy()

    # Stack series if requested
    if want_series and series is not None:
        for k in list(series.keys()):
            if len(series[k]) > 0:
                series[k] = torch.cat(series[k], dim=0).numpy()
            else:
                series[k] = None

    metrics = dict(
        imf_mae=float(imf_mae),
        imf_rmse=float(imf_rmse),
        sum_mae=float(sum_mae),
        recon_vs_rrp_mae=(None if recon_vs_rrp_mae is None else float(recon_vs_rrp_mae)),
        per_mode_mae=per_mode_mae
    )
    return metrics, series


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--csv", default="VMD_modes_with_residual_2021_2022.csv",
                    help="CSV with Mode_1..Mode_12, Residual, RRP")
    ap.add_argument("--target-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--x-mode", type=str, choices=["raw", "sum"], default="raw",
                    help="Decomposer input used at eval: raw RRP window (default) or sum of IMFs")
    # Model
    ap.add_argument("--ckpt", default="./runs_nvmd_autoencoder_raw/best.pt",
                    help="Path to decomposer checkpoint (.pt)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Loader
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    # Output
    ap.add_argument("--dump-npz", type=str, default="",
                    help="Optional: save predictions/targets to .npz")
    ap.add_argument("--print-per-mode", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    # Load ckpt
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    decomp_cols = ck.get("decomp_cols", None)

    # Load data
    df = pd.read_csv(args.csv)
    if decomp_cols is None:
        decomp_cols = build_default_13(df)

    # Dataset/Loader
    ds = DecompOnlyDatasetRaw(
        df=df,
        decomp_cols=decomp_cols,
        target_col=args.target_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
    )
    pin = (device == "cuda")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=pin)

    # Build model from ckpt hyperparams when possible
    model = NVMD_Autoencoder(
        in_ch=1,
        base=ck_args.get("base", 128),
        K=ck_args.get("K", len(decomp_cols)),
        signal_len=args.seq_len
    ).to(device)
    model.load_state_dict(ck["model_state"], strict=True)

    # Eval
    metrics, series = eval_epoch_raw(
        model=model,
        loader=dl,
        device=device,
        want_series=bool(args.dump_npz),
        x_mode_for_rrp=args.x_mode,
        per_mode=args.print_per_mode
    )

    print(f"IMF MAE:         {metrics['imf_mae']:.6f}")
    print(f"IMF RMSE:        {metrics['imf_rmse']:.6f}")
    print(f"Sum-consist MAE: {metrics['sum_mae']:.6f}")
    if metrics["recon_vs_rrp_mae"] is not None:
        print(f"Recon vs RRP MAE:{metrics['recon_vs_rrp_mae']:.6f}")

    if args.print_per_mode and metrics["per_mode_mae"] is not None:
        pm = metrics["per_mode_mae"]
        for i, v in enumerate(pm, 1):
            name = f"Mode_{i}" if i <= 12 else "Residual"
            print(f"  {name:>9}: {float(v):.6f}")

    if args.dump_npz and series is not None:
        np.savez_compressed(
            args.dump_npz,
            pred_imfs=series["pred_imfs"],
            true_imfs=series["true_imfs"],
            pred_sum=series["pred_sum"],
            true_sum=series["true_sum"],
            x_input=series["x_input"],
            decomp_cols=np.array(decomp_cols)
        )
        print(f"Saved series to: {args.dump_npz}")


if __name__ == "__main__":
    main()
