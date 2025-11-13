#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nvmd_transformer import NVMDTransformerRRPModel  



def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class RRPWindowDataset(Dataset):

    def __init__(self, df: pd.DataFrame, seq_len: int = 64, rrp_col: str = "RRP"):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe columns")

        rrp = df[rrp_col].to_numpy(dtype=np.float32)
        self.rrp = torch.from_numpy(rrp).contiguous()
        T = len(rrp)
        self.N = max(0, T - seq_len - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        x = self.rrp[i:i+L].unsqueeze(0)   # (1, L)
        y = self.rrp[i+L].unsqueeze(0)     # (1,)
        return x, y



def init_with_pretrained_decomposer(model, ckpt_path: str, verbose: bool = True):
    
    if not os.path.exists(ckpt_path):
        if verbose:
            print(f"[init] WARNING: AE checkpoint not found at {ckpt_path}, "
                  f"decomposer will remain randomly initialized.")
        return model

    if verbose:
        print(f"[init] Loading AE checkpoint from: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu")
    state_dict = ck.get("model_state", ck)

    if any(k.startswith("decomposer.") for k in state_dict.keys()):
        decomp_state = {
            k[len("decomposer."):] : v
            for k, v in state_dict.items()
            if k.startswith("decomposer.")
        }
    else:
        decomp_state = state_dict

    missing, unexpected = model.decomposer.load_state_dict(decomp_state, strict=False)

    if verbose:
        print("[init] Decomposer loaded.")
        if missing:
            print("       Missing keys   :", missing)
        if unexpected:
            print("       Unexpected keys:", unexpected)

    return model

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw = x_raw.to(device)      # (B,1,L)
        rrp_next = rrp_next.to(device)  # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x_raw)           # (B,1)

        mse = F.mse_loss(pred, rrp_next)
        mae = F.l1_loss(pred, rrp_next)

        if is_train:
            mse.backward()
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str,
                    default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str,
                    default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=64)

    # Model hyperparams
    ap.add_argument("--K", type=int, default=13)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--dim-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-freeze-decomposer", action="store_true",
                    help="If set, decomposer will be trainable.")

    # Training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0)

    # Autoencoder checkpoint
    ap.add_argument("--ae-ckpt", type=str, default="./nvmd_ae_imf_recon.pt")

    # Output
    ap.add_argument("--out", type=str, default="./nvmd_transformer_rrp.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------- Data ----------------
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = RRPWindowDataset(df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
    va_ds = RRPWindowDataset(df_va, seq_len=args.seq_len, rrp_col=args.rrp_col)

    pin = device == "cuda"
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

    # ---------------- Model ----------------
    freeze_decomposer = not args.no_freeze_decomposer
    model = NVMDTransformerRRPModel(
        seq_len=args.seq_len,
        K=args.K,
        base=args.base,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        freeze_decomposer=freeze_decomposer,
    ).to(device)

    # Load pretrained decomposer weights
    model = init_with_pretrained_decomposer(model, args.ae_ckpt, verbose=True)

    # Optimizer: only train params that are requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_mae = float("inf")

    # ---------------- Training Loop ----------------
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(model, tr_dl, device, optimizer=opt)
        va_mse, va_mae = run_epoch(model, va_dl, device, optimizer=None)

        print(
            f"[Epoch {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        # Save best based on validation MAE
        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "notes": "NVMD + Transformer RRP predictor (RRP next-step only)",
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
