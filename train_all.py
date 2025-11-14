#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_nvmd import HybridSpectralNVMD
from train_transformer import MultiModeTransformerRRP


class VMD13IMFNextDataset(Dataset):
    def __init__(self, df, seq_len=64, rrp_col="RRP", K=13):
        self.L = seq_len
        self.K = K

        rrp = df[rrp_col].to_numpy(dtype="float32")
        mode_cols = [f"Mode_{i}" for i in range(1, K)] + ["Residual"]
        imfs = df[mode_cols].to_numpy(dtype="float32")

        self.rrp = torch.tensor(rrp)              # (T,)
        self.imfs = torch.tensor(imfs).T          # (K,T)
        T = len(rrp)
        self.N = T - seq_len - 1                  # last index where rrp_next exists

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L
        x_raw = self.rrp[i:i+L].unsqueeze(0)      
        imfs_true = self.imfs[:, i:i+L]           
        rrp_next = self.rrp[i+L].unsqueeze(0)     
        return x_raw, imfs_true, rrp_next

def train_epoch(
    decomposer,
    predictor,
    loader,
    optimizer,
    device,
    w_pred,
    w_imf,
    w_rrp,
    w_smooth,
    w_ortho,
    clip_grad=10.0,
):
    decomposer.train()
    predictor.train()

    total_pred = total_imf = total_rrp = 0.0
    n = 0

    for x_raw, imfs_true, rrp_next in loader:
        x_raw = x_raw.to(device)
        imfs_true = imfs_true.to(device)
        rrp_next = rrp_next.to(device)

        optimizer.zero_grad()

        # forward pass
        imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)
        rrp_next_hat = predictor(imfs_ref)

        # losses
        loss_pred = F.mse_loss(rrp_next_hat, rrp_next)
        loss_imf = F.l1_loss(imfs_ref, imfs_true)
        loss_rrp = F.l1_loss(recon_ref, x_raw)
        loss_smooth = decomposer.spectral.spectral_smoothness_loss()
        loss_ortho = decomposer.spectral.orthogonality_loss()

        loss = (
            w_pred * loss_pred +
            w_imf * loss_imf +
            w_rrp * loss_rrp +
            w_smooth * loss_smooth +
            w_ortho * loss_ortho
        )

        loss.backward()
        nn.utils.clip_grad_norm_(list(decomposer.parameters()) + list(predictor.parameters()), clip_grad)
        optimizer.step()

        # logging
        bs = x_raw.size(0)
        n += bs
        total_pred += loss_pred.item() * bs
        total_imf += loss_imf.item() * bs
        total_rrp += loss_rrp.item() * bs

    return total_pred/n, total_imf/n, total_rrp/n


def eval_epoch(decomposer, predictor, loader, device):
    decomposer.eval()
    predictor.eval()

    total_pred = total_imf = total_rrp = total_mae = 0.0
    n = 0

    with torch.no_grad():
        for x_raw, imfs_true, rrp_next in loader:
            x_raw = x_raw.to(device)
            imfs_true = imfs_true.to(device)
            rrp_next = rrp_next.to(device)

            imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)
            rrp_next_hat = predictor(imfs_ref)

            loss_pred = F.mse_loss(rrp_next_hat, rrp_next)
            loss_imf = F.l1_loss(imfs_ref, imfs_true)
            loss_rrp = F.l1_loss(recon_ref, x_raw)
            mae = F.l1_loss(rrp_next_hat, rrp_next)

            bs = x_raw.size(0)
            n += bs
            total_pred += loss_pred.item() * bs
            total_imf += loss_imf.item() * bs
            total_rrp += loss_rrp.item() * bs
            total_mae += mae.item() * bs

    return (
        total_pred/n,
        total_imf/n,
        total_rrp/n,
        total_mae/n,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--K", type=int, default=13)

    # pretrained weights
    ap.add_argument("--decomposer_ckpt", type=str, default="./hybrid_spectral_nvmd.pt")
    ap.add_argument("--predictor_ckpt", type=str, default="./transformer_only_rrp.pt")

    # training
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)

    # loss weights
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--w-imf", type=float, default=1.0)
    ap.add_argument("--w-rrp", type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-ortho", type=float, default=1e-3)

    ap.add_argument("--freeze-decomposer", action="store_true")

    ap.add_argument("--out", type=str, default="./joint_finetune.pt")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = VMD13IMFNextDataset(df_tr, seq_len=args.seq_len, K=args.K)
    va_ds = VMD13IMFNextDataset(df_va, seq_len=args.seq_len, K=args.K)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)


    decomposer = HybridSpectralNVMD(K=args.K, signal_len=args.seq_len).to(device)
    dec_ckpt = torch.load(args.decomposer_ckpt, map_location="cpu")
    
    if "model_state" in dec_ckpt:
        dec_state = dec_ckpt["model_state"]
    elif "decomposer_state" in dec_ckpt:
        dec_state = dec_ckpt["decomposer_state"]
    else:
        dec_state = dec_ckpt
    
    missing_d, unexpected_d = decomposer.load_state_dict(dec_state, strict=False)
    print("DECOMPOSER missing:", missing_d)
    print("DECOMPOSER unexpected:", unexpected_d)
    

    predictor = MultiModeTransformerRRP(K=args.K, seq_len=args.seq_len).to(device)
    pred_ckpt = torch.load(args.predictor_ckpt, map_location="cpu")
    
    if "model_state" in pred_ckpt:
        pred_state = pred_ckpt["model_state"]
    elif "predictor_state" in pred_ckpt:
        pred_state = pred_ckpt["predictor_state"]
    else:
        pred_state = pred_ckpt
    
    missing_p, unexpected_p = predictor.load_state_dict(pred_state, strict=False)
    print("PREDICTOR missing:", missing_p)
    print("PREDICTOR unexpected:", unexpected_p)
    
    # freeze option
    if args.freeze_decomposer:
        for p in decomposer.parameters():
            p.requires_grad = False

    params = list(predictor.parameters()) if args.freeze_decomposer else \
             list(decomposer.parameters()) + list(predictor.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_mae = float("inf")

    for ep in range(1, args.epochs+1):
        tr_mse, tr_imf, tr_rrp = train_epoch(
            decomposer, predictor, tr_dl, optimizer, device,
            args.w_pred, args.w_imf, args.w_rrp, args.w_smooth, args.w_ortho
        )

        va_mse, va_imf, va_rrp, va_mae = eval_epoch(
            decomposer, predictor, va_dl, device
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train MSE={tr_mse:.4f} IMF={tr_imf:.4f} RRPwin={tr_rrp:.4f} | "
            f"val MSE={va_mse:.4f} IMF={va_imf:.4f} RRPwin={va_rrp:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_mae:
            best_mae = va_mae
            ckpt = {
                "epoch": ep,
                "decomposer_state": decomposer.state_dict(),
                "predictor_state": predictor.state_dict(),
                "best_val_mae": best_mae,
            }
            torch.save(ckpt, args.out)
            print(f" → Saved best checkpoint (val MAE={best_mae:.4f})")



if __name__ == "__main__":
    main()
