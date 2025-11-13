#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_price_model import NVMDMultiModePriceModel   # <-- your big model
from nvmd_autoencoder import NVMD_Autoencoder           # <-- decomposer architecture


# ==========================================================
# Utils
# ==========================================================
def set_seed(seed=1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_imf_cols(df):
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    return cols


# ==========================================================
# Dataset — returns x_raw, rrp_next, imf_win, imf_next
# ==========================================================
class RRPIMFNextDataset(Dataset):
    def __init__(self, df, seq_len=64, rrp_col="RRP", imf_cols=None):
        super().__init__()
        self.L = seq_len

        if imf_cols is None:
            imf_cols = build_imf_cols(df)
        self.imf_cols = imf_cols
        self.K = len(imf_cols)

        rrp = df[rrp_col].to_numpy(np.float32)
        imfs = df[imf_cols].to_numpy(np.float32)  # (T,K)

        self.rrp = torch.tensor(rrp)              # (T,)
        self.imfs = torch.tensor(imfs)            # (T,K)

        T = len(rrp)
        self.N = T - seq_len - 1

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L

        x_raw = self.rrp[i:i+L].unsqueeze(0)       # (1,L)
        rrp_next = self.rrp[i+L].unsqueeze(0)

        imf_win = self.imfs[i:i+L].T               # (K,L)
        imf_next = self.imfs[i+L]                  # (K,)

        return x_raw, rrp_next, imf_win, imf_next


# ==========================================================
# Training / Eval epoch
# ==========================================================
def train_or_eval_epoch(model, loader, device, alpha, beta, gamma, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    totL = tot_rrp = tot_win = tot_next = 0.0
    n = 0

    for x_raw, rrp_next, imf_win, imf_next in loader:
        x_raw   = x_raw.to(device)
        rrp_next = rrp_next.to(device)
        imf_win  = imf_win.to(device)
        imf_next = imf_next.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Forward
        imfs_pred, y_modes, y_price = model(x_raw)

        # Losses
        L_rrp = F.mse_loss(y_price, rrp_next)         # RRP next MSE
        L_win = F.l1_loss(imfs_pred, imf_win)         # IMF window L1
        L_nxt = F.l1_loss(y_modes, imf_next)          # IMF next-step L1

        loss = alpha*L_rrp + beta*L_win + gamma*L_nxt

        if is_train:
            loss.backward()
            optimizer.step()

        bs = x_raw.size(0)
        n += bs
        totL    += loss.item() * bs
        tot_rrp += L_rrp.item() * bs
        tot_win += L_win.item() * bs
        tot_next+= L_nxt.item() * bs

    return (
        totL/n,
        tot_rrp/n,
        tot_win/n,
        tot_next/n
    )


# ==========================================================
# Main
# ==========================================================
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--seq-len", type=int, default=64)

    # Model
    ap.add_argument("--K", type=int, default=13)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--bidirectional", action="store_true")

    # Training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)

    # Loss weights
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)

    # Pretrained decomposer
    ap.add_argument("--ae-ckpt", type=str, default="./nvmd_ae_imf_recon.pt")

    # I/O
    ap.add_argument("--out", type=str, default="./nvmd_joint_frozen_predictor.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # ====================
    # Load dataset
    # ====================
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)
    imf_cols = build_imf_cols(df_tr)

    tr_ds = RRPIMFNextDataset(df_tr, args.seq_len, "RRP", imf_cols)
    va_ds = RRPIMFNextDataset(df_va, args.seq_len, "RRP", imf_cols)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)


    # ====================
    # Build model
    # ====================
    model = NVMDMultiModePriceModel(
        signal_len=args.seq_len,
        K=args.K,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        mlp_hidden=args.mlp_hidden,
        freeze_decomposer=True,
    ).to(device)


    if os.path.exists(args.ae_ckpt):
        ck = torch.load(args.ae_ckpt, map_location="cpu")
        ae_state = ck["model_state"] if "model_state" in ck else ck

        missing, unexpected = model.decomposer.load_state_dict(ae_state, strict=False)
        print("Loaded decomposer from:", args.ae_ckpt)
        print("  Missing keys   :", missing)
        print("  Unexpected keys:", unexpected)
    else:
        print("WARNING: no autoencoder checkpoint found → training decomposer from scratch")



    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)


    best_rrp = float("inf")

    for ep in range(1, args.epochs+1):
        tr_tot, tr_rrp, tr_win, tr_next = train_or_eval_epoch(
            model, tr_dl, device, args.alpha, args.beta, args.gamma, optimizer=opt
        )
        va_tot, va_rrp, va_win, va_next = train_or_eval_epoch(
            model, va_dl, device, args.alpha, args.beta, args.gamma, optimizer=None
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train: total={tr_tot:.4f} rrp={tr_rrp:.4f} win={tr_win:.4f} next={tr_next:.4f} | "
            f"val: total={va_tot:.4f} rrp={va_rrp:.4f} win={va_win:.4f} next={va_next:.4f}"
        )

        if va_rrp < best_rrp:
            best_rrp = va_rrp
            torch.save(model.state_dict(), args.out)
            print(f"  → saved best (rrp={best_rrp:.4f})")



if __name__ == "__main__":
    main()
