#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- your modules ----
from nvmd_autoencoder import MultiModeNVMD       
from nvmd_transformer import MultiModeTransformerRRP  



class FullRRPModel(nn.Module):
    def __init__(self, decomposer: nn.Module, predictor: nn.Module):
        super().__init__()
        self.decomposer = decomposer
        self.predictor = predictor

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # decomposer: (B,1,L) → (B,K,L), (B,1,L)
        imfs, _ = self.decomposer(x_raw)
        # predictor: (B,K,L) → (B,1)
        rrp_next_hat = self.predictor(imfs)
        return rrp_next_hat

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RRPDataset(Dataset):
    """
    Simple next-step RRP dataset:
      x_raw: (1, L)  window of RRP
      y:     (1,)    next-step RRP
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 64, rrp_col: str = "RRP"):
        super().__init__()
        self.L = seq_len
        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe columns.")

        rrp = df[rrp_col].to_numpy(dtype=np.float32)
        self.rrp = torch.from_numpy(rrp)
        T = len(rrp)
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        x = self.rrp[i:i+L].unsqueeze(0)   # (1,L)
        y = self.rrp[i+L].unsqueeze(0)     # (1,)
        return x, y


def _extract_args_from_ckpt(ck):
    """
    Try to extract the 'args' dict from a checkpoint if present.
    Returns {} if not found.
    """
    if isinstance(ck, dict) and "args" in ck and isinstance(ck["args"], dict):
        return ck["args"]
    return {}


def _strip_module(sd):
    """
    Strip leading 'module.' from keys (for DataParallel checkpoints).
    """
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def build_decomposer_from_ckpt(path: str, device: str):
    """
    Build a MultiModeNVMD with hyperparameters read from the decomposer checkpoint.
    Expects ckpt to contain at least a state_dict (raw or under 'model_state'/ 'state_dict')
    and, optionally, an 'args' dict with K/base/signal_len info.
    """
    print(f"Loading decomposer from: {path}")
    ck = torch.load(path, map_location=device)

    # Get args (best-effort)
    decomp_args = _extract_args_from_ckpt(ck)

    # How many modes?
    K = decomp_args.get("K", 13)
    base = decomp_args.get("base", 64)

    # Build module
    decomposer = MultiModeNVMD(K=K, base=base).to(device)

    # Extract state_dict
    if isinstance(ck, dict):
        for key in ["model_state", "state_dict", "model_state_dict"]:
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break

    if not isinstance(ck, dict):
        raise ValueError(f"Decomposer checkpoint at {path} has no usable state_dict.")

    sd = _strip_module(ck)
    missing, unexpected = decomposer.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [Decomposer] Missing keys when loading: {missing}")
    if unexpected:
        print(f"  [Decomposer] Unexpected keys when loading: {unexpected}")

    return decomposer, K, decomp_args


def build_predictor_from_ckpt(path: str, K: int, seq_len: int, device: str):
    """
    Build a MultiModeTransformerRRP with hyperparameters read from predictor checkpoint.
    Expects ckpt with state_dict + 'args' including d_model, n_heads, num_layers, dim_ff, dropout.
    """
    print(f"Loading predictor from: {path}")
    ck = torch.load(path, map_location=device)

    pred_args = _extract_args_from_ckpt(ck)

    d_model   = pred_args.get("d_model", 128)
    n_heads   = pred_args.get("n_heads", 4)
    num_layers= pred_args.get("num_layers", 3)
    dim_ff    = pred_args.get("dim_ff", 256)
    dropout   = pred_args.get("dropout", 0.1)

    predictor = MultiModeTransformerRRP(
        K=K,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
    ).to(device)

    # Extract state_dict
    if isinstance(ck, dict):
        for key in ["model_state", "state_dict", "model_state_dict"]:
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break

    if not isinstance(ck, dict):
        raise ValueError(f"Predictor checkpoint at {path} has no usable state_dict.")

    sd = _strip_module(ck)
    missing, unexpected = predictor.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [Predictor] Missing keys when loading: {missing}")
    if unexpected:
        print(f"  [Predictor] Unexpected keys when loading: {unexpected}")

    return predictor, pred_args


def maybe_freeze(module: nn.Module, freeze: bool, name: str):
    if not freeze:
        return
    print(f"Freezing {name} parameters.")
    for p in module.parameters():
        p.requires_grad = False
    module.eval()





def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    grad_clip: float | None = 10.0,
):
    """
    One epoch of train or eval.
    Loss: MSE on RRP next-step; also report MAE.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw = x_raw.to(device)        # (B,1,L)
        rrp_next = rrp_next.to(device)  # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        y_pred = model(x_raw)           # (B,1)

        mse = F.mse_loss(y_pred, rrp_next)
        mae = F.l1_loss(y_pred, rrp_next)

        if is_train:
            mse.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ==========================================================
#                            Main
# ==========================================================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=64,
                    help="Will be overridden by decomposer ckpt if it contains seq_len/signal_len.")

    # Checkpoints (these carry the hyperparameters)
    ap.add_argument("--decomposer-ckpt", type=str, default="./nvmd_ae_imf_recon.pt",
                    help="Path to pretrained MultiModeNVMD checkpoint.")
    ap.add_argument("--predictor-ckpt",  type=str, default="./transformer_only_rrp.pt",
                    help="Path to pretrained MultiModeTransformerRRP checkpoint.")

    # Freeze options
    ap.add_argument("--freeze-decomposer", action="store_true",
                    help="Freeze decomposer parameters during retrain.")
    ap.add_argument("--freeze-predictor",  action="store_true",
                    help="Freeze predictor parameters during retrain.")

    # Training
    ap.add_argument("--epochs",    type=int, default=50)
    ap.add_argument("--batch",     type=int, default=256)
    ap.add_argument("--lr",        type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    # Output
    ap.add_argument("--out", type=str, default="./nvmd_transformer_rrp_finetuned.pt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------- Build decomposer from ckpt (hyperparams from ckpt.args) ----------
    decomposer, K, decomp_args = build_decomposer_from_ckpt(args.decomposer_ckpt, device=device)

    # Try to derive seq_len from decomposer args if present
    seq_from_ckpt = decomp_args.get("seq_len", decomp_args.get("signal_len", None))
    if seq_from_ckpt is not None:
        if seq_from_ckpt != args.seq_len:
            print(f"[Info] Overriding seq_len {args.seq_len} → {seq_from_ckpt} from decomposer ckpt.")
        args.seq_len = int(seq_from_ckpt)

    # --------- Build predictor from ckpt (hyperparams from ckpt.args) ----------
    predictor, pred_args = build_predictor_from_ckpt(
        args.predictor_ckpt,
        K=K,
        seq_len=args.seq_len,
        device=device,
    )

    # --------- Wrap into a single model ----------
    model = FullRRPModel(decomposer, predictor).to(device)

    # --------- Apply freezing if requested ----------
    maybe_freeze(model.decomposer, args.freeze_decomposer, "decomposer")
    maybe_freeze(model.predictor,  args.freeze_predictor,  "predictor")

    # --------- Data ----------
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = RRPDataset(df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
    va_ds = RRPDataset(df_va, seq_len=args.seq_len, rrp_col=args.rrp_col)

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

    # --------- Optimizer on only trainable params ----------
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        train_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")

    # --------- Training loop ----------
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(
            model,
            tr_dl,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
            grad_clip=None,
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
                    "decomposer_ckpt": args.decomposer_ckpt,
                    "predictor_ckpt": args.predictor_ckpt,
                    "K": K,
                    "seq_len": args.seq_len,
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
