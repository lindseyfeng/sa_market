#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_nvmd import HybridSpectralNVMD
from train_transformer import MultiModeTransformerRRP


# ============================================================
# Dataset: returns  (x_raw, imfs_true, rrp_next)
# ============================================================
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
        self.N = T - seq_len - 1

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L
        x_raw = self.rrp[i:i+L].unsqueeze(0)      
        imfs_true = self.imfs[:, i:i+L]           
        rrp_next = self.rrp[i+L].unsqueeze(0)     
        return x_raw, imfs_true, rrp_next



# ============================================================
# Training functions
# ============================================================
def train_predictor_only(decomposer, predictor, loader, opt, device):
    decomposer.eval()                 # freeze structure
    predictor.train()

    tot_pred = 0.0
    n = 0

    for x_raw, imfs_true, rrp_next in loader:
        x_raw, rrp_next = x_raw.to(device), rrp_next.to(device)

        opt.zero_grad()

        # forward: use frozen decomposer IMFs only
        with torch.no_grad():
            imfs_ref, _, _, _ = decomposer(x_raw)

        rrp_hat = predictor(imfs_ref)

        loss = F.mse_loss(rrp_hat, rrp_next)
        loss.backward()
        opt.step()

        bs = x_raw.size(0)
        tot_pred += loss.item() * bs
        n += bs

    return tot_pred / n



def train_joint(
    decomposer, predictor, loader, opt, device,
    w_pred, w_imf, w_rrp, w_smooth, w_ortho,
):
    decomposer.train()
    predictor.train()

    tot_pred = 0.0
    n = 0

    for x_raw, imfs_true, rrp_next in loader:
        x_raw = x_raw.to(device)
        imfs_true = imfs_true.to(device)
        rrp_next = rrp_next.to(device)

        opt.zero_grad()

        # Decomposer forward
        imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)

        # Predictor forward
        rrp_hat = predictor(imfs_ref)

        # Combined joint loss
        loss_pred = F.mse_loss(rrp_hat, rrp_next)
        loss_imf  = F.l1_loss(imfs_ref, imfs_true)
        loss_rrp  = F.l1_loss(recon_ref, x_raw)

        loss_smooth = decomposer.spectral.spectral_smoothness_loss()
        loss_ortho  = decomposer.spectral.orthogonality_loss()

        loss = (
            w_pred * loss_pred
          + w_imf  * loss_imf
          + w_rrp  * loss_rrp
          + w_smooth * loss_smooth
          + w_ortho  * loss_ortho
        )

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(decomposer.parameters()) + list(predictor.parameters()), 10.0
        )
        opt.step()

        bs = x_raw.size(0)
        tot_pred += loss_pred.item() * bs
        n += bs

    return tot_pred / n



def eval_all(decomposer, predictor, loader, device):
    decomposer.eval()
    predictor.eval()

    tot_mse = tot_mae = 0.0
    n = 0

    with torch.no_grad():
        for x_raw, imfs_true, rrp_next in loader:
            x_raw = x_raw.to(device)
            rrp_next = rrp_next.to(device)

            imfs_ref, _, _, _ = decomposer(x_raw)
            rrp_hat = predictor(imfs_ref)

            mse = F.mse_loss(rrp_hat, rrp_next)
            mae = F.l1_loss(rrp_hat, rrp_next)

            bs = x_raw.size(0)
            tot_mse += mse.item() * bs
            tot_mae += mae.item() * bs
            n += bs

    return tot_mse / n, tot_mae / n



# ============================================================
# Main: warmup → joint
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-csv", type=str, required=True)

    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--K", type=int, default=13)

    ap.add_argument("--decomposer-ckpt", type=str, required=True)
    ap.add_argument("--predictor-ckpt", type=str, required=True)

    # training stages
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--joint-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)

    # joint loss weights
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--w-imf", type=float, default=1.0)
    ap.add_argument("--w-rrp", type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-ortho", type=float, default=1e-3)

    ap.add_argument("--out", type=str, default="joint.pt")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== data ====
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)
    tr_dl = DataLoader(VMD13IMFNextDataset(df_tr, seq_len=args.seq_len, K=args.K),
                       batch_size=256, shuffle=True, drop_last=True)
    va_dl = DataLoader(VMD13IMFNextDataset(df_va, seq_len=args.seq_len, K=args.K),
                       batch_size=256, shuffle=False)

    # ==== load models ====
    decomposer = HybridSpectralNVMD(K=args.K, signal_len=args.seq_len).to(device)
    dec_sd = torch.load(args.decomposer_ckpt, map_location="cpu")
    dec_sd = dec_sd.get("decomposer_state", dec_sd)
    decomposer.load_state_dict(dec_sd, strict=False)

    predictor = MultiModeTransformerRRP(K=args.K, seq_len=args.seq_len).to(device)
    pred_sd = torch.load(args.predictor_ckpt, map_location="cpu")
    pred_sd = pred_sd.get("predictor_state", pred_sd)
    predictor.load_state_dict(pred_sd, strict=False)

    # ===========================
    #     STAGE 1 — WARMUP
    # ===========================
    print("\n====== Stage 1: Train predictor only (frozen decomposer) ======\n")

    for p in decomposer.parameters():
        p.requires_grad = False

    opt_pred = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    for ep in range(1, args.warmup_epochs + 1):
        tr = train_predictor_only(decomposer, predictor, tr_dl, opt_pred, device)
        va_mse, va_mae = eval_all(decomposer, predictor, va_dl, device)
        print(f"[Warmup {ep:03d}] train pred={tr:.4f} | val MSE={va_mse:.4f} MAE={va_mae:.4f}")

    # ===========================
    #     STAGE 2 — JOINT
    # ===========================
    print("\n====== Stage 2: Joint training ======\n")

    for p in decomposer.parameters():
        p.requires_grad = True

    opt_joint = torch.optim.Adam(
        list(decomposer.parameters()) + list(predictor.parameters()),
        lr=args.lr
    )

    for ep in range(1, args.joint_epochs + 1):
        tr = train_joint(
            decomposer, predictor, tr_dl, opt_joint, device,
            args.w_pred, args.w_imf, args.w_rrp, args.w_smooth, args.w_ortho
        )
        va_mse, va_mae = eval_all(decomposer, predictor, va_dl, device)
        print(f"[Joint {ep:03d}] train pred={tr:.4f} | val MSE={va_mse:.4f} MAE={va_mae:.4f}")

        # save best
        ckpt = {
            "decomposer_state": decomposer.state_dict(),
            "predictor_state": predictor.state_dict(),
            "epoch": ep,
        }
        torch.save(ckpt, args.out)


if __name__ == "__main__":
    main()
