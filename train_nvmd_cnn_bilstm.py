#!/usr/bin/env python3
import argparse, os, json, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset (NO normalization)
# -------------------------
class Decomp13DatasetRaw(Dataset):
    """
    Raw (unnormalized) 13 signals: Mode_1..Mode_12, Residual.
    x_raw = sum of raw 13 signals.
    y_true = raw RRP.
    """
    def __init__(self, df: pd.DataFrame, decomp_cols: list[str], seq_len: int = 256, target_col: str = "RRP"):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)
        self.L = seq_len

        # (T, K) -> tensors
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)       # (T,K)
        self.imfs = torch.tensor(imfs_np).transpose(0,1)           # (K,T)

        target_np = df[target_col].to_numpy(dtype=np.float32)      # (T,)
        self.target = torch.tensor(target_np, dtype=torch.float32) # (T,)

        # raw summed signal
        self.signal_raw = self.imfs.sum(dim=0)                     # (T,)

        T = len(self.target)
        self.N = max(0, T - self.L - 1)

    def __len__(self): return self.N

    def __getitem__(self, i):
        L = self.L
        x = self.signal_raw[i:i+L].unsqueeze(0)    # (1,L)
        imf_win = self.imfs[:, i:i+L]             # (K,L)
        y = self.target[i+L]                      # predict next point
        return x, imf_win, y.unsqueeze(0)         # (1,L), (K,L), (1,)


# -------------------------
# Training / Eval (NO normalization)
# -------------------------
def _set_trainable(mod, on: bool):
    for p in mod.parameters(): p.requires_grad = on

def _freeze_for_stage(model, stage: str):
    """stage: 'decomp' -> train decomposer; 'pred' -> train predictors; 'both' -> train all."""
    assert stage in {"decomp", "pred", "both"}
    _set_trainable(model.decomposer, stage in {"decomp", "both"})
    for pred in model.predictors:
        _set_trainable(pred, stage in {"pred", "both"})

def _retie_optimizer_to_trainables(optimizer, model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(optimizer.param_groups) == 0:
        optimizer.add_param_group({"params": trainable})
    else:
        optimizer.param_groups[0]["params"] = trainable

@torch.no_grad()
def eval_epoch(loader, model, device, sum_reg: float):
    model.eval()
    total, sum_d, sum_p = 0, 0.0, 0.0
    for xb, imfs_true, yb in loader:
        xb = xb.to(device)               # (B,1,L)
        imfs_true = imfs_true.to(device) # (B,K,L)
        yb = yb.to(device)               # (B,1)

        imfs_pred, y_pred = model(xb)    # (B,K,L), (B,1)

        # raw-scale losses
        loss_decomp = F.mse_loss(imfs_pred, imfs_true)
        sig_pred = imfs_pred.sum(dim=1)  # (B,L)
        sig_true = imfs_true.sum(dim=1)  # (B,L)
        loss_sumcons = F.mse_loss(sig_pred, sig_true)
        loss_pred = F.mse_loss(y_pred, yb)

        # regularized decomp (for logging)
        loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons

        bs = xb.size(0); total += bs
        sum_d += loss_decomp_reg.item() * bs
        sum_p += loss_pred.item() * bs

    return (sum_d / max(total,1), sum_p / max(total,1))

def train_epoch(loader, model, device, optimizer, clip_grad: float,
                alpha: float, beta: float, sum_reg: float, stage: str):
    """
    stage: 'decomp' -> optimize decomp only
           'pred'   -> optimize pred only
           'both'   -> alpha * decomp_reg + beta * pred
    """
    assert stage in {"decomp", "pred", "both"}
    model.train()
    total, sum_d, sum_p = 0, 0.0, 0.0

    for xb, imfs_true, yb in loader:
        xb = xb.to(device)
        imfs_true = imfs_true.to(device)
        yb = yb.to(device)

        imfs_pred, y_pred = model(xb)

        # raw-scale losses
        loss_decomp = F.mse_loss(imfs_pred, imfs_true)
        sig_pred = imfs_pred.sum(dim=1)
        sig_true = imfs_true.sum(dim=1)
        loss_sumcons = F.mse_loss(sig_pred, sig_true)
        loss_pred = F.mse_loss(y_pred, yb)

        loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons

        if stage == "decomp":
            loss = loss_decomp_reg
        elif stage == "pred":
            loss = loss_pred
        else:
            loss = alpha * loss_decomp_reg + beta * loss_pred

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        bs = xb.size(0); total += bs
        sum_d += loss_decomp_reg.item() * bs
        sum_p += loss_pred.item() * bs

    return (sum_d / max(total,1), sum_p / max(total,1))


# -------------------------
# Col helper
# -------------------------
def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 13-series setting: {missing}")
    return cols


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--seq-len", type=int, default=256)

    # Model
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)

    # Train
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--sum-reg", type=float, default=1.0)      # γ for sum-consistency on raw scale
    ap.add_argument("--clip-grad", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--stage", type=str, default="both", choices=["decomp","pred","both"],
                    help="What to optimize in this run")

    # Logs / save
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_mrc_bilstm_raw")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)
    decomp_cols = build_default_13(df_tr)

    trds = Decomp13DatasetRaw(df_tr, decomp_cols, seq_len=args.seq_len, target_col="RRP")
    vads = Decomp13DatasetRaw(df_va, decomp_cols, seq_len=args.seq_len, target_col="RRP")

    trdl = DataLoader(trds, batch_size=args.batch, shuffle=False,  num_workers=args.num_workers, pin_memory=True)
    vadl = DataLoader(vads, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    K = len(decomp_cols)
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len, K=K, base=args.base,
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional, freeze_decomposer=False
    ).to(device)

    # Stage freeze
    _freeze_for_stage(model, args.stage)

    # Optimizer (only trainable params)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Train loop
    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        # ensure optimizer tracks current trainables (useful if you change stage mid-run)
        _retie_optimizer_to_trainables(opt, model)

        tr_ld, tr_lp = train_epoch(
            trdl, model, device, opt, args.clip_grad,
            args.alpha, args.beta, args.sum_reg, args.stage
        )
        va_ld, va_lp = eval_epoch(vadl, model, device, args.sum_reg)

        # Aggregate "total" same as both-mode for logging
        tr_total = args.alpha * tr_ld + args.beta * tr_lp if args.stage == "both" else (tr_ld if args.stage=="decomp" else tr_lp)
        va_total = args.alpha * va_ld + args.beta * va_lp if args.stage == "both" else (va_ld if args.stage=="decomp" else va_lp)

        print(f"[Epoch {ep:02d}] "
              f"train: total={tr_total:.6f} decomp={tr_ld:.6f} pred={tr_lp:.6f} | "
              f"val:   total={va_total:.6f} decomp={va_ld:.6f} pred={va_lp:.6f}")

        if va_total < best_val:
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # checkpoint
        torch.save({
            "epoch": ep,
            "val_best": best_val,
            "model_state": model.state_dict(),
            "args": vars(args),
            "decomp_cols": decomp_cols,
        }, os.path.join(args.outdir, f"epoch_{ep:02d}.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({
            "epoch": "best",
            "val_best": best_val,
            "model_state": best_state,
            "args": vars(args),
            "decomp_cols": decomp_cols,
        }, os.path.join(args.outdir, "best.pt"))
        print(f"Saved best checkpoint: val_total={best_val:.6f} → {os.path.join(args.outdir,'best.pt')}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
