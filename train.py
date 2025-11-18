#!/usr/bin/env python3
"""
Joint NVMD + Transformer training with IMF supervision and prediction.

- Dataset:
    x_raw:      (1, L)   window of raw RRP [t, ..., t+L-1]
    imfs_true:  (K, L)   VMD IMFs over same window
    rrp_next:   (1,)     RRP at time t+L

- Model:
    NVMDTransformerJoint:
        x_raw -> HybridSpectralNVMD -> imfs_ref, recon_ref, priors
        x_modes = cat(raw, imfs_ref) -> MultiModeTransformerRRP -> rrp_hat

- Loss:
    loss = w_pred   * MSE(rrp_hat, rrp_next)
         + w_imf    * relRMSE(imfs_ref, imfs_true)
         + w_rrp    * L1(recon_ref, x_raw)
         + w_smooth * smooth_loss
         + w_ortho  * ortho_loss

Example:

    python train_joint_nvmd_transformer_imf.py \
        --train-csv VMD_modes_with_residual_2018_2021.csv \
        --val-csv   VMD_modes_with_residual_2021_2022.csv \
        --seq-len 256 \
        --epochs 100 \
        --out nvmd_transformer_joint_imf.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_nvmd import HybridSpectralNVMD
from train_transformer import MultiModeTransformerRRP


# ============================================================
#                     Dataset (raw + IMFs)
# ============================================================

class JointDecompPredictDataset(Dataset):
    """
    For each start index i:

      x_raw:      (1, L)   raw RRP window [i, ..., i+L-1]
      imfs_true:  (K, L)   VMD IMFs over same window
      rrp_next:   (1,)     raw RRP at time i+L
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 256,
        rrp_col: str = "RRP",
        K: int = 13,
    ):
        super().__init__()
        self.L = seq_len
        self.rrp_col = rrp_col
        self.K = K

        if rrp_col not in df.columns:
            raise ValueError(f"RRP column '{rrp_col}' not in dataframe")

        mode_cols = [f"Mode_{i}" for i in range(1, K)] + ["Residual"]
        for c in mode_cols:
            if c not in df.columns:
                raise ValueError(f"Missing mode column '{c}' in dataframe")
        self.mode_cols = mode_cols

        rrp = df[rrp_col].to_numpy(dtype=np.float32)        # (T,)
        imfs = df[mode_cols].to_numpy(dtype=np.float32)     # (T, K)

        self.rrp = torch.from_numpy(rrp)                    # (T,)
        # (T,K) -> (K,T)
        self.imfs = torch.from_numpy(imfs).transpose(0, 1)  # (K,T)

        T = self.rrp.shape[0]
        # we need rrp[i+L] to exist
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L

        # raw window (1,L)
        x_raw = self.rrp[i:i+L].unsqueeze(0)        # (1,L)

        # IMF window (K,L)
        imfs_true = self.imfs[:, i:i+L]             # (K,L)

        # next-step RRP
        rrp_next = self.rrp[i+L].unsqueeze(0)       # (1,)

        return x_raw, imfs_true, rrp_next


# ============================================================
#               Integrated NVMD + Transformer model
# ============================================================

class NVMDTransformerJoint(nn.Module):
    """
    Integrated NVMD + Transformer:

      - NVMD:   x_raw -> imfs_ref (B,K,L), recon_ref (B,1,L), priors
      - Coupling: x_modes = concat(raw, imfs_ref) -> (B, K+1, L)
      - Transformer: x_modes -> rrp_hat (B,1)
    """
    def __init__(
        self,
        K: int = 13,
        seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        use_raw_as_mode: bool = True,
    ):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.use_raw_as_mode = use_raw_as_mode

        # NVMD decomposer
        self.decomposer = HybridSpectralNVMD(K=K, signal_len=seq_len)

        # how many channels/modes we feed to the Transformer
        K_pred = K + 1 if use_raw_as_mode else K

        # Transformer predictor that expects (B, K_pred, L)
        self.predictor = MultiModeTransformerRRP(
            K=K_pred,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )

    def forward(self, x_raw: torch.Tensor, return_details: bool = False):
        """
        x_raw: (B,1,L)

        Returns:
            if return_details:
                (rrp_hat, imfs_ref, recon_ref, smooth_loss, ortho_loss)
            else:
                rrp_hat
        """
        # NVMD decomposition
        imfs_ref, recon_ref, imfs_lin, recon_lin = self.decomposer(x_raw)  # (B,K,L), (B,1,L), ...

        # build modes for Transformer
        if self.use_raw_as_mode:
            x_modes = torch.cat([x_raw, imfs_ref], dim=1)  # (B,K+1,L)
        else:
            x_modes = imfs_ref                            # (B,K,L)

        # prediction
        rrp_hat = self.predictor(x_modes)  # (B,1)

        # priors on masks
        smooth_loss = self.decomposer.spectral.spectral_smoothness_loss()
        ortho_loss  = self.decomposer.spectral.orthogonality_loss()

        if return_details:
            return rrp_hat, imfs_ref, recon_ref, smooth_loss, ortho_loss
        else:
            return rrp_hat


# ============================================================
#                     Training / Eval Epoch
# ============================================================

def run_epoch(
    model: NVMDTransformerJoint,
    loader: DataLoader,
    device: str,
    optimizer=None,
    w_pred: float = 1.0,
    w_imf: float = 1.0,
    w_rrp: float = 0.1,
    w_smooth: float = 0.1,
    w_ortho: float = 0.1,
    max_grad_norm: float = 10.0,
):
    """
    Joint training/eval epoch.

    Loss = w_pred   * MSE(rrp_hat, rrp_next)
         + w_imf    * relRMSE(imfs_ref, imfs_true)
         + w_rrp    * L1(recon_ref, x_raw)
         + w_smooth * smooth_loss
         + w_ortho  * ortho_loss

    Returns: (avg_mse, avg_mae) on prediction.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0
    eps = 1e-8

    for x_raw, imfs_true, rrp_next in loader:
        x_raw     = x_raw.to(device)      # (B,1,L)
        imfs_true = imfs_true.to(device)  # (B,K,L)
        rrp_next  = rrp_next.to(device)   # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        rrp_hat, imfs_ref, recon_ref, smooth_loss, ortho_loss = model(
            x_raw,
            return_details=True,
        )

        # prediction loss
        mse = F.mse_loss(rrp_hat, rrp_next)
        mae = F.l1_loss(rrp_hat, rrp_next)

        # IMF supervision: relative RMSE
        delta = imfs_ref - imfs_true           # (B,K,L)
        num = (delta ** 2).sum(dim=(1, 2))     # (B,)
        den = (imfs_true ** 2).sum(dim=(1, 2)) + eps
        rel_rmse = torch.sqrt(num / den)       # (B,)
        loss_imf = rel_rmse.mean()

        # RRP reconstruction loss
        loss_rrp = F.l1_loss(recon_ref, x_raw)

        # priors
        loss_smooth = smooth_loss
        loss_ortho  = ortho_loss

        # total loss
        loss = (
            w_pred   * mse
          + w_imf    * loss_imf
          + w_rrp    * loss_rrp
          + w_smooth * loss_smooth
          + w_ortho  * loss_ortho
        )

        if is_train:
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
#                          Utilities
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
#                            MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=256)
    ap.add_argument("--K",         type=int, default=13)

    # model hparams
    ap.add_argument("--d-model",    type=int, default=128)
    ap.add_argument("--n-heads",    type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dim-ff",     type=int, default=256)
    ap.add_argument("--dropout",    type=float, default=0.1)

    # training
    ap.add_argument("--batch",         type=int,   default=256)
    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight-decay",  type=float, default=1e-2)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--num-workers",   type=int,   default=0)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)

    # loss weights
    ap.add_argument("--w-pred",   type=float, default=1.0)
    ap.add_argument("--w-imf",    type=float, default=0.5)
    ap.add_argument("--w-rrp",    type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=0.01)
    ap.add_argument("--w-ortho",  type=float, default=0.01)

    # I/O
    ap.add_argument("--out", type=str, default="nvmd_transformer_joint_imf.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load dataframes
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = JointDecompPredictDataset(
        df_tr,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        K=args.K,
    )
    va_ds = JointDecompPredictDataset(
        df_va,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        K=args.K,
    )

    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train windows: {len(tr_ds)}, Val windows: {len(va_ds)}")

    # model
    model = NVMDTransformerJoint(
        K=args.K,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        use_raw_as_mode=True,  # raw RRP as extra mode
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")

    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(
            model,
            tr_dl,
            device,
            optimizer=optimizer,
            w_pred=args.w_pred,
            w_imf=args.w_imf,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )

        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
            w_pred=args.w_pred,
            w_imf=args.w_imf,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
