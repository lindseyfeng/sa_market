#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
#   1. LSTM Baseline Model
# ============================================================

class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,       # Number of modes (K)
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()
        
        # Normalization (Crucial for LSTMs to converge on financial/energy data)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM Core
        # batch_first=True expects input: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # Input x shape from dataset: (Batch, K, Seq_Len)
        # LSTM expects: (Batch, Seq_Len, Features=K)
        
        # 1. Permute to put Time in the middle dimension
        x = x.permute(0, 2, 1)  # (B, L, K)
        
        # 2. Normalize inputs
        x = self.input_norm(x)
        
        # 3. LSTM Pass
        # out shape: (Batch, Seq_Len, Hidden_Dim)
        # (h_n, c_n) are the final states, but we can just grab the last output
        out, (h_n, c_n) = self.lstm(x)
        
        # 4. Take the output of the LAST time step
        # This contains the "summary" of the entire sequence
        last_step_output = out[:, -1, :] # (Batch, Hidden_Dim)
        
        # 5. Prediction
        prediction = self.head(last_step_output) # (Batch, 1)
        
        return prediction


# ============================================================
#   2. Dataset (Same as Transformer)
# ============================================================

def build_default_mode_cols(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        alt_cols = [c for c in df.columns if "Mode" in c or "IMF" in c or "Residual" in c]
        if len(alt_cols) > 0:
            return sorted(alt_cols)
        else:
            raise ValueError(f"Could not find VMD mode columns. Missing: {missing}")
    return cols

class ModesRRPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        rrp_col: str = "RRP",
        mode_cols: list[str] = None,
    ):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            if 'y' in df.columns: rrp_col = 'y'
            elif 'target' in df.columns: rrp_col = 'target'
            else: raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")

        if mode_cols is None:
            mode_cols = build_default_mode_cols(df)
        self.mode_cols = mode_cols
        self.K_dim = len(mode_cols)

        modes_np = df[mode_cols].to_numpy(dtype=np.float32)
        rrp_np   = df[rrp_col].to_numpy(dtype=np.float32)

        self.modes = torch.from_numpy(modes_np)
        self.rrp   = torch.from_numpy(rrp_np)

        T = self.rrp.shape[0]
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # Input: (L, K)
        x_modes = self.modes[i:i+L, :]
        
        # Transpose to (K, L) for consistency with PyTorch Image-style formatting
        # (Though LSTM flips it back, we keep Dataset consistent across scripts)
        x_modes = x_modes.T.contiguous() 
        
        # Target: Next Step
        rrp_next = self.rrp[i+L].unsqueeze(0)
        
        return x_modes, rrp_next


# ============================================================
#   3. Training Loop
# ============================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    max_grad_norm: float = 1.0,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_modes, rrp_next in loader:
        x_modes   = x_modes.to(device)
        rrp_next  = rrp_next.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        y_hat = model(x_modes)

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
#   4. Main
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2018.csv")
    ap.add_argument("--val-csv", type=str, default="VMD_modes_with_residual_2019_2019.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=256)

    # LSTM Model Params
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training Params
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--out", type=str, default="./lstm_baseline.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if not os.path.exists(args.train_csv):
        print(f"Error: Train file '{args.train_csv}' not found.")
        # return

    # Load Data
    try:
        print("Loading data...")
        df_tr = pd.read_csv(args.train_csv)
        df_va = pd.read_csv(args.val_csv)

        tr_ds = ModesRRPDataset(df=df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
        va_ds = ModesRRPDataset(df=df_va, seq_len=args.seq_len, rrp_col=args.rrp_col, mode_cols=tr_ds.mode_cols)
        
        K = tr_ds.K_dim
        print(f"Detected {K} input modes. Seq_Len={args.seq_len}")

        pin = (device == "cuda")
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    print("Initializing LSTM Baseline...")
    model = LSTMBaseline(
        input_dim=K,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")

    print(f"Starting training on {device}...")
    
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(model, tr_dl, device, optimizer=optimizer, max_grad_norm=args.max_grad_norm)
        va_mse, va_mae = run_epoch(model, va_dl, device, optimizer=None, max_grad_norm=args.max_grad_norm)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        tr_rmse = math.sqrt(tr_mse)
        va_rmse = math.sqrt(va_mse)

        print(
            f"[Epoch {ep:03d} | LR:{current_lr:.2e}] "
            f"Tr: RMSE={tr_rmse:.4f} MAE={tr_mae:.4f} | "
            f"Va: RMSE={va_rmse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "val_rmse": va_rmse,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  -> Saved Best: {args.out}")

if __name__ == "__main__":
    main()