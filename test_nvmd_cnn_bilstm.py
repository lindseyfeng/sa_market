

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM

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

    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],         # IMF columns (e.g. Mode_1..Mode_12, Residual)
        seq_len: int = 9,
        rrp_col: str = "RRP",
        imf_mins: np.ndarray | None = None,
        imf_maxs: np.ndarray | None = None,
    ):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)
        self.L = seq_len

        # ----- raw IMFs: (T, K) -> (K, T)
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)   # (T, K)
        self.imfs_raw = torch.tensor(imfs_np).transpose(0, 1)  # (K, T)

        # raw RRP series (for x windows only)
        rrp_np = df[rrp_col].to_numpy(dtype=np.float32)
        self.rrp_raw = torch.tensor(rrp_np, dtype=torch.float32)  # (T,)
        T = len(self.rrp_raw)

        # per-IMF min/max for normalization (global over the dataset or given from train)
        if imf_mins is None or imf_maxs is None:
            imf_mins = self.imfs_raw.min(dim=1).values.numpy()   # (K,)
            imf_maxs = self.imfs_raw.max(dim=1).values.numpy()   # (K,)
        self.imf_mins = np.asarray(imf_mins, dtype=np.float32)
        self.imf_maxs = np.asarray(imf_maxs, dtype=np.float32)

        mins = torch.tensor(self.imf_mins).unsqueeze(1)  # (K, 1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs_raw - mins) / rngs   # (K, T) in [0, 1] ideally

        # number of usable windows (need L points for history + 1 for target)
        self.N = T - self.L - 1
        if self.N < 0:
            self.N = 0

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        """
        i corresponds to window [i, ..., i+L-1] and target at time i+L.
        """
        L = self.L

        # x_raw: raw RRP window, shape (1, L)
        x_raw = self.rrp_raw[i : i + L].unsqueeze(0)            # (1, L)

        # imf_in_norm: normalized IMF window, shape (K, L)
        imf_in_norm = self.imfs_norm[:, i : i + L]              # (K, L)

        # imf_target_norm: normalized IMFs at time t+L, shape (K,)
        imf_target_norm = self.imfs_norm[:, i + L]              # (K,)

        return x_raw, imf_in_norm, imf_target_norm

    def scalers(self):
        """
        For denorm inside training:
           imf_raw = imf_norm * (max - min) + min
        """
        return dict(
            imf_mins=self.imf_mins,
            imf_maxs=self.imf_maxs,
        )


def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 13-series setting: {missing}")
    return cols

@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    imf_mins,
    imf_maxs,
    crop: int = 0,
    return_series: bool = False,
):
    """
    Eval to match training's prediction target:

      y_pred_rrp = sum_k IMF_k_raw_pred(t+L)
      y_true_rrp = sum_k IMF_k_raw_true(t+L)

    Metrics:
      - mae, rmse over y_pred_rrp vs y_true_rrp
      - recon_mae over sum_k IMF_k_raw(t) time series (optional)
    """
    model.eval()

    # scaler for IMFs (same as in training)
    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1,  # (B,K,L)
    )

    y_true_all, y_pred_all = [], []
    recon_mae_accum, n_batches = 0.0, 0

    for x_raw, imf_in_norm, imf_target_norm in loader:
        x_raw          = x_raw.to(device)          # (B,1,L)
        imf_in_norm    = imf_in_norm.to(device)    # (B,K,L)
        imf_target_norm = imf_target_norm.to(device)  # (B,K)

        # Forward pass
        imfs_pred_norm, y_modes_norm_pred = model(x_raw)  # (B,K,L), (B,K)

        # ---- Next-step RRP from IMFs (same as training) ----
        # denorm mode scalars
        y_modes_raw_pred = imf_scaler.denorm(
            y_modes_norm_pred.unsqueeze(-1)
        ).squeeze(-1)                                # (B,K)
        imf_target_raw   = imf_scaler.denorm(
            imf_target_norm.unsqueeze(-1)
        ).squeeze(-1)                                # (B,K)

        y_pred_rrp = y_modes_raw_pred.sum(dim=1, keepdim=True)  # (B,1)
        y_true_rrp = imf_target_raw.sum(dim=1, keepdim=True)    # (B,1)

        y_true_all.append(y_true_rrp)
        y_pred_all.append(y_pred_rrp)

        # ---- Optional reconstruction metric (time series) ----
        if crop > 0:
            imfs_pred_crop = imfs_pred_norm[:, :, crop:-crop]
            imfs_true_crop = imf_in_norm[:, :, crop:-crop]
        else:
            imfs_pred_crop = imfs_pred_norm
            imfs_true_crop = imf_in_norm

        imfs_pred_raw = imf_scaler.denorm(imfs_pred_crop)  # (B,K,L')
        imfs_true_raw = imf_scaler.denorm(imfs_true_crop)  # (B,K,L')

        sig_pred = imfs_pred_raw.sum(dim=1)  # (B,L')
        sig_true = imfs_true_raw.sum(dim=1)  # (B,L')

        recon_mae_batch = torch.mean(torch.abs(sig_pred - sig_true)).item()
        recon_mae_accum += recon_mae_batch
        n_batches += 1

    # ---- aggregate over all batches ----
    y_true_all = torch.cat(y_true_all, dim=0).view(-1)
    y_pred_all = torch.cat(y_pred_all, dim=0).view(-1)

    mae  = torch.mean(torch.abs(y_pred_all - y_true_all)).item()
    rmse = torch.sqrt(torch.mean((y_pred_all - y_true_all) ** 2)).item()
    recon_mae = recon_mae_accum / max(n_batches, 1)

    if return_series:
        return (
            mae,
            rmse,
            recon_mae,
            y_true_all.cpu().numpy(),
            y_pred_all.cpu().numpy(),
        )
    return mae, rmse, recon_mae

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--csv",  default="VMD_modes_with_residual_2021_2022.csv",
                    help="CSV with Mode_1..Mode_12, Residual, RRP")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--target-col", type=str, default="RRP")
    ap.add_argument("--x-mode", type=str, choices=["raw","sum_norm"], default="raw",
                    help="Use raw RRP window (train default) or sum of normalized IMFs")
    # Model
    ap.add_argument("--ckpt", default="./runs_nvmd_joint_mse/best.pt",
                    help="Path to training checkpoint .pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Loader
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    # Metrics
    ap.add_argument("--crop", type=int, default=0, help="Center-crop on time axis when computing recon metric")
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

    # IMF scalers: prefer those saved in ckpt for consistency
    imf_mins = ck_scalers.get("imf_mins", None)
    imf_maxs = ck_scalers.get("imf_maxs", None)

    # Dataset / Loader
    ds = Decomp13Dataset(
        df=df,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col=args.target_col,
        imf_mins=imf_mins,
        imf_maxs=imf_maxs,
        x_mode=args.x_mode
    )
    sc = ds.scalers()
    pin = (device == "cuda")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=pin)

    # Build model to match training hyperparams if present
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
    mae, rmse, recon_mae, y_true, y_pred = eval_epoch(
        model=model,
        loader=dl,
        device=device,
        imf_mins=sc["imf_mins"],
        imf_maxs=sc["imf_maxs"],
        crop=args.crop,
        return_series=True
    )

    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Recon(sum) MAE (crop={args.crop}): {recon_mae:.6f}")

    # Optional CSV
    if args.out_csv:
        out = pd.DataFrame({
            "idx": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_err": np.abs(y_true - y_pred),
        })
        out.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to: {args.out_csv}")

if __name__ == "__main__":
    main()
