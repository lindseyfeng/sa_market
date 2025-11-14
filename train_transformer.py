#!/usr/bin/env python3
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
#                     Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding for 1D sequences.
    Expects input shape (B, L, d_model).
    """
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (L, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


# ============================================================
#               Transformer Predictor (no NVMD)
# ============================================================

class MultiModeTransformerRRP(nn.Module):
    def __init__(
        self,
        K: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.d_model = d_model

        # Project 13-mode vector at each time → d_model
        self.input_proj = nn.Linear(K, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # (B, L, d_model)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.rrp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_modes: torch.Tensor) -> torch.Tensor:
        """
        x_modes: (B, K, L)  — 13 modes over window
        returns:
            rrp_next_hat: (B, 1)
        """
        B, K, L = x_modes.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # Treat time as sequence dimension
        x = x_modes.permute(0, 2, 1)   # (B, L, K)

        x = self.input_proj(x)         # (B, L, d_model)
        x = self.pos_enc(x)            # (B, L, d_model)

        h = self.encoder(x)            # (B, L, d_model)

        # Use last time step as representation
        h_last = h[:, -1, :]           # (B, d_model)

        rrp_next_hat = self.rrp_head(h_last)  # (B, 1)
        return rrp_next_hat


# ============================================================
#     Dataset: windows of 13 VMD modes + next-step RRP
# ============================================================

def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing VMD mode columns: {missing}")
    return cols


class ModesRRPDataset(Dataset):
    """
    For each window i..i+L-1:

      x_modes: (K, L)   → 13 VMD IMFs over window
      rrp_next: (1,)    → raw RRP at time t+L
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        rrp_col: str = "RRP",
        mode_cols: list[str] | None = None,
    ):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")

        if mode_cols is None:
            mode_cols = build_default_13(df)
        self.mode_cols = mode_cols
        K = len(mode_cols)

        modes_np = df[mode_cols].to_numpy(dtype=np.float32)  # (T, K)
        rrp_np   = df[rrp_col].to_numpy(dtype=np.float32)    # (T,)

        self.modes = torch.from_numpy(modes_np)   # (T, K)
        self.rrp   = torch.from_numpy(rrp_np)     # (T,)
        self.K     = K

        T = self.rrp.shape[0]
        # Need t+L for target → max start index T-L-1
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # Window of modes: times [i, ..., i+L-1] → shape (L, K)
        x_modes = self.modes[i:i+L, :]          # (L, K)
        x_modes = x_modes.T.contiguous()        # (K, L)

        # Next-step RRP: time t+L
        rrp_next = self.rrp[i+L].unsqueeze(0)   # (1,)

        return x_modes, rrp_next


# ============================================================
#                  Training / Eval Epochs
# ============================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    max_grad_norm: float = 10.0,
):
    """
    If optimizer is provided → training, otherwise evaluation.

    Loss: MSE on raw RRP.
    Metrics: MSE, MAE on raw RRP.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_modes, rrp_next in loader:
        x_modes   = x_modes.to(device)    # (B,K,L)
        rrp_next  = rrp_next.to(device)   # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        y_hat = model(x_modes)            # (B,1)

        mse = F.mse_loss(y_hat, rrp_next)
        mae = F.l1_loss(y_hat, rrp_next)

        if is_train:
            mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        bs = x_modes.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
#                           MAIN
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", type=str,
                    default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str,
                    default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=256)

    # Model
    ap.add_argument("--K", type=int, default=13,
                    help="Number of modes (default 13: Mode_1..Mode_12 + Residual)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dim-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)

    # I/O
    ap.add_argument("--out", type=str, default="./transformer_only_rrp.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    mode_cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    if args.K != len(mode_cols):
        raise ValueError(f"K={args.K} but default mode_cols length is {len(mode_cols)}")

    tr_ds = ModesRRPDataset(
        df=df_tr,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        mode_cols=mode_cols,
    )
    va_ds = ModesRRPDataset(
        df=df_va,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        mode_cols=mode_cols,
    )

    pin = (device == "cuda")
    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # Model
    model = MultiModeTransformerRRP(
        K=args.K,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
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
            max_grad_norm=args.max_grad_norm,
        )
        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
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
                    "mode_cols": mode_cols,
                    "notes": "Transformer-only RRP predictor on VMD modes",
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
