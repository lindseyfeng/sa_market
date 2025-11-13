import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMDMultiModePriceModel  


# ==========================================================
# Utils
# ==========================================================
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_imf_cols(df: pd.DataFrame):
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing IMF columns in dataframe: {missing}")
    return cols


# ==========================================================
# Dataset: RRP window + IMF window + next-step targets
# ==========================================================
class RRPIMFNextDataset(Dataset):
    """
    For each i, returns:
        x_raw:    (1, L)      RRP window [i .. i+L-1]
        rrp_next: (1,)        RRP at time i+L
        imf_win:  (K, L)      IMF window [i .. i+L-1] for all K modes
        imf_next: (K,)        IMF at time i+L for all K modes
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        rrp_col: str = "RRP",
        imf_cols: list[str] | None = None,
    ):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")

        if imf_cols is None:
            imf_cols = build_imf_cols(df)
        self.imf_cols = imf_cols
        self.K = len(imf_cols)

        rrp = df[rrp_col].to_numpy(dtype=np.float32)          # (T,)
        imfs = df[imf_cols].to_numpy(dtype=np.float32)        # (T, K)

        self.rrp = torch.from_numpy(rrp)                      # (T,)
        self.imfs = torch.from_numpy(imfs)                    # (T, K)

        T = len(rrp)
        # need [i .. i+L-1] for windows and i+L for next-step targets
        self.N = T - seq_len - 1

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L

        # RRP window + next-step
        x_raw = self.rrp[i:i+L].unsqueeze(0)         # (1,L)
        rrp_next = self.rrp[i+L].unsqueeze(0)        # (1,)

        # IMF window + next-step
        imf_win = self.imfs[i:i+L]                   # (L,K)
        imf_win = imf_win.T                          # (K,L)
        imf_next = self.imfs[i+L]                    # (K,)

        return x_raw, rrp_next, imf_win, imf_next


# ==========================================================
# One epoch: joint RRP + IMF losses
# ==========================================================
def train_or_eval_epoch_joint(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    alpha: float,
    beta: float,
    gamma: float,
    optimizer=None,
):
    """
    Joint objective:

        L_rrp      = MSE( y_price,  rrp_next )         # main task
        L_imf_win  = L1(  imfs_pred, imf_win )         # IMF window reconstruction
        L_imf_next = L1(  y_modes,   imf_next )        # per-mode next-step IMF scalar

        L_total = alpha * L_rrp + beta * L_imf_win + gamma * L_imf_next

    Returns:
        mean_total_loss,
        mean_rrp_mse,
        mean_imf_win_l1,
        mean_imf_next_l1
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    total_rrp_mse_sum = 0.0
    total_imf_win_l1_sum = 0.0
    total_imf_next_l1_sum = 0.0
    n_samples = 0

    for x_raw, rrp_next, imf_win, imf_next in loader:
        x_raw    = x_raw.to(device)       # (B,1,L)
        rrp_next = rrp_next.to(device)    # (B,1)
        imf_win  = imf_win.to(device)     # (B,K,L)
        imf_next = imf_next.to(device)    # (B,K)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Forward through big model
        imfs_pred, y_modes, y_price = model(x_raw)
        # imfs_pred: (B,K,L)
        # y_modes:   (B,K)
        # y_price:   (B,1)

        # 1) Final RRP loss (MSE, as you requested)
        L_rrp = F.mse_loss(y_price, rrp_next)

        L_imf_win = F.l1_loss(imfs_pred, imf_win)
        
        L_imf_next = F.l1_loss(y_modes, imf_next)

        loss = alpha * L_rrp + beta * L_imf_win + gamma * L_imf_next

        if is_train:
            loss.backward()
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_loss_sum       += loss.item()       * bs
        total_rrp_mse_sum    += L_rrp.item()      * bs
        total_imf_win_l1_sum += L_imf_win.item()  * bs
        total_imf_next_l1_sum+= L_imf_next.item() * bs

    denom = max(n_samples, 1)
    return (
        total_loss_sum       / denom,
        total_rrp_mse_sum    / denom,
        total_imf_win_l1_sum / denom,
        total_imf_next_l1_sum/ denom,
    )


# ==========================================================
# Main
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=64)

    # Model hyperparams
    ap.add_argument("--K",            type=int, default=13)
    ap.add_argument("--base",         type=int, default=64,
                    help="base channels in NVMD decomposer")
    ap.add_argument("--lstm-hidden",  type=int, default=256)
    ap.add_argument("--lstm-layers",  type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--mlp-hidden",   type=int, default=64)

    ap.add_argument("--freeze-decomposer", action="store_true",
                    help="if set, do NOT update decomposer weights")

    ap.add_argument("--decomp-ckpt", type=str, default="",
                    help="optional: path to pretrained NVMD AE checkpoint")

    # Training hyperparams
    ap.add_argument("--epochs", type=int,   default=20)
    ap.add_argument("--batch",  type=int,   default=256)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed",   type=int,   default=1337)
    ap.add_argument("--num-workers", type=int, default=0)

    # Loss weights
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight for RRP next-step MSE loss")
    ap.add_argument("--beta",  type=float, default=1.0,
                    help="weight for IMF window L1 loss")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="weight for IMF next-step L1 loss")

    # I/O
    ap.add_argument("--out", type=str, default="./nvmd_multimode_joint_best.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------- Load data -----------------
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    imf_cols = build_imf_cols(df_tr)

    tr_ds = RRPIMFNextDataset(df_tr, seq_len=args.seq_len,
                              rrp_col=args.rrp_col, imf_cols=imf_cols)
    va_ds = RRPIMFNextDataset(df_va, seq_len=args.seq_len,
                              rrp_col=args.rrp_col, imf_cols=imf_cols)

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

    # ----------------- Model -----------------
    model = NVMDMultiModePriceModel(
        signal_len=args.seq_len,
        K=args.K,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        freeze_decomposer=args.freeze_decomposer,
        mlp_hidden=args.mlp_hidden,
    ).to(device)

    # Optionally load a pretrained decomposer (IMF recon AE)
    if args.decomp_ckpt:
        ck = torch.load(args.decomp_ckpt, map_location="cpu")
        state = ck.get("model_state", ck)
        missing, unexpected = model.decomposer.load_state_dict(state, strict=False)
        print(f"Loaded decomposer from {args.decomp_ckpt}")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)

    # ----------------- Optimizer -----------------
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rrp = float("inf")

    # ----------------- Training loop -----------------
    for ep in range(1, args.epochs + 1):
        tr_tot, tr_rrp, tr_imf_win, tr_imf_next = train_or_eval_epoch_joint(
            model,
            tr_dl,
            device,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            optimizer=opt,
        )
        va_tot, va_rrp, va_imf_win, va_imf_next = train_or_eval_epoch_joint(
            model,
            va_dl,
            device,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            optimizer=None,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train: total={tr_tot:.4f} RRP_MSE={tr_rrp:.4f} IMF_win_L1={tr_imf_win:.4f} IMF_next_L1={tr_imf_next:.4f} | "
            f"val:   total={va_tot:.4f} RRP_MSE={va_rrp:.4f} IMF_win_L1={va_imf_win:.4f} IMF_next_L1={va_imf_next:.4f}"
        )

        # Save based on RRP performance (primary task)
        if va_rrp < best_val_rrp:
            best_val_rrp = va_rrp
            ckpt = {
                "epoch": ep,
                "val_rrp_mse": best_val_rrp,
                "model_state": model.state_dict(),
                "args": vars(args),
                "notes": "NVMDMultiModePriceModel joint training: RRP + IMF window + IMF next",
            }
            torch.save(ckpt, args.out)
            print(f"  → saved best checkpoint to {args.out} (val RRP MSE={best_val_rrp:.4f})")


if __name__ == "__main__":
    main()
