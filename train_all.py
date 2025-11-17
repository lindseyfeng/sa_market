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
# Dataset: long decomposer window → next-step RRP
# ============================================================

class LongContextRRPDataset(Dataset):
    """
    For each index i:

      long window for decomposer: [i, ..., i+L_dec-1]
      target: RRP at time i+L_dec

    Outputs:
        x_long:   (1, L_dec)  → decomposer input
        rrp_next:(1,)         → prediction target
    """
    def __init__(self, df, dec_len=2048, rrp_col="RRP"):
        super().__init__()
        self.L_dec = dec_len

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not found in dataframe")

        rrp = df[rrp_col].to_numpy(dtype="float32")  # (T,)
        self.rrp = torch.tensor(rrp)                 # (T,)

        T = len(rrp)
        # need i+L_dec for context + i+L_dec for target
        self.N = T - dec_len - 1
        if self.N <= 0:
            raise ValueError(
                f"Not enough samples for dec_len={dec_len}, T={T} (N={self.N})"
            )

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        Ld = self.L_dec
        # long context window: [i, ..., i+Ld-1]
        x_long = self.rrp[i : i + Ld].unsqueeze(0)   # (1, L_dec)
        # next-step target at time i+Ld
        rrp_next = self.rrp[i + Ld].unsqueeze(0)     # (1,)
        return x_long, rrp_next


# ============================================================
# Training functions
# ============================================================

def train_predictor_only(
    decomposer,
    predictor,
    loader,
    opt,
    device,
    pred_len: int,
):
    """
    Stage 1: freeze decomposer, train predictor only.

    For each batch:
      x_long (B,1,L_dec) → decomposer → imfs_ref (B,K,L_dec)
      imfs_short = imfs_ref[:, :, -pred_len:] → predictor → rrp_hat (B,1)
    """
    decomposer.eval()   # frozen
    predictor.train()

    tot_pred = 0.0
    n = 0

    for x_long, rrp_next in loader:
        x_long = x_long.to(device)         # (B,1,L_dec)
        rrp_next = rrp_next.to(device)     # (B,1)

        opt.zero_grad()

        with torch.no_grad():
            imfs_ref, _, _, _ = decomposer(x_long)   # (B,K,L_dec)

        # take last pred_len steps for prediction
        imfs_short = imfs_ref[:, :, -pred_len:]      # (B,K,pred_len)

        rrp_hat = predictor(imfs_short)              # (B,1)

        loss = F.mse_loss(rrp_hat, rrp_next)
        loss.backward()
        opt.step()

        bs = x_long.size(0)
        tot_pred += loss.item() * bs
        n += bs

    return tot_pred / max(n, 1)


def train_joint(
    decomposer,
    predictor,
    loader,
    opt,
    device,
    pred_len: int,
    w_pred: float,
    w_rrp: float,
    w_smooth: float,
    w_ortho: float,
):
    """
    Stage 2: joint training of decomposer + predictor.

      Loss = w_pred * MSE(rrp_hat, rrp_next)
           + w_rrp  * L1(recon_ref, x_long)
           + w_smooth * spectral_smoothness
           + w_ortho  * orthogonality
    """
    decomposer.train()
    predictor.train()

    tot_pred = 0.0
    n = 0

    for x_long, rrp_next in loader:
        x_long = x_long.to(device)         # (B,1,L_dec)
        rrp_next = rrp_next.to(device)     # (B,1)

        opt.zero_grad()

        # Decomposer forward on long window
        imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_long)  # (B,K,L_dec), ...

        # Predictor only sees last pred_len timesteps
        imfs_short = imfs_ref[:, :, -pred_len:]      # (B,K,pred_len)

        rrp_hat = predictor(imfs_short)              # (B,1)

        # prediction loss
        loss_pred = F.mse_loss(rrp_hat, rrp_next)
        # reconstruction loss on long window
        loss_rrp  = F.l1_loss(recon_ref, x_long)
        # spectral regularizers
        loss_smooth = decomposer.spectral.spectral_smoothness_loss()
        loss_ortho  = decomposer.spectral.orthogonality_loss()

        loss = (
            w_pred   * loss_pred
          + w_rrp    * loss_rrp
          + w_smooth * loss_smooth
          + w_ortho  * loss_ortho
        )

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(decomposer.parameters()) + list(predictor.parameters()),
            max_norm=10.0,
        )
        opt.step()

        bs = x_long.size(0)
        tot_pred += loss_pred.item() * bs
        n += bs

    return tot_pred / max(n, 1)


def eval_all(
    decomposer,
    predictor,
    loader,
    device,
    pred_len: int,
):
    decomposer.eval()
    predictor.eval()

    tot_mse = 0.0
    tot_mae = 0.0
    n = 0

    with torch.no_grad():
        for x_long, rrp_next in loader:
            x_long = x_long.to(device)
            rrp_next = rrp_next.to(device)

            imfs_ref, _, _, _ = decomposer(x_long)       # (B,K,L_dec)
            imfs_short = imfs_ref[:, :, -pred_len:]      # (B,K,pred_len)

            rrp_hat = predictor(imfs_short)

            mse = F.mse_loss(rrp_hat, rrp_next)
            mae = F.l1_loss(rrp_hat, rrp_next)

            bs = x_long.size(0)
            tot_mse += mse.item() * bs
            tot_mae += mae.item() * bs
            n += bs

    denom = max(n, 1)
    return tot_mse / denom, tot_mae / denom


# ============================================================
# Main: warmup → joint with dec_len != pred_len
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-csv", type=str, required=True)

    # decomposer vs predictor lengths
    ap.add_argument("--dec-len", type=int, default=2048,
                    help="Sequence length for decomposer (HybridSpectralNVMD)")
    ap.add_argument("--pred-len", type=int, default=7,
                    help="Sequence length for predictor (MultiModeTransformerRRP)")

    ap.add_argument("--K", type=int, default=13)

    ap.add_argument("--decomposer-ckpt", type=str, required=True)
    ap.add_argument("--predictor-ckpt", type=str, required=True)

    # training stages
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--joint-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)

    # joint loss weights
    ap.add_argument("--w-pred",   type=float, default=1.0)
    ap.add_argument("--w-rrp",    type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-ortho",  type=float, default=1e-3)

    ap.add_argument("--out", type=str, default="joint.pt")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ==== data ====
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = LongContextRRPDataset(df_tr, dec_len=args.dec_len)
    va_ds = LongContextRRPDataset(df_va, dec_len=args.dec_len)

    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=256, shuffle=False)

    # ==== load models ====
    # Decomposer uses long length
    decomposer = HybridSpectralNVMD(K=args.K, signal_len=args.dec_len).to(device)
    dec_sd = torch.load(args.decomposer_ckpt, map_location="cpu")
    dec_sd = dec_sd.get("decomposer_state", dec_sd)
    decomposer.load_state_dict(dec_sd, strict=False)

    # Predictor uses short length
    predictor = MultiModeTransformerRRP(
        K=args.K,
        seq_len=args.pred_len,
    ).to(device)
    ckpt = torch.load(args.predictor_ckpt, map_location="cpu")
    if "model_state" in ckpt:
        predictor.load_state_dict(ckpt["model_state"])
    else:
        print("wrong!!")
        predictor.load_state_dict(ckpt)  


    # ========================================================
    # Stage 1: predictor warmup (decomposer frozen)
    # ========================================================
    print("\n====== Stage 1: Train predictor only (frozen decomposer) ======\n")

    for p in decomposer.parameters():
        p.requires_grad = False

    opt_pred = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    for ep in range(1, args.warmup_epochs + 1):
        tr = train_predictor_only(
            decomposer,
            predictor,
            tr_dl,
            opt_pred,
            device,
            pred_len=args.pred_len,
        )
        va_mse, va_mae = eval_all(
            decomposer,
            predictor,
            va_dl,
            device,
            pred_len=args.pred_len,
        )
        print(f"[Warmup {ep:03d}] train pred={tr:.4f} | val MSE={va_mse:.4f} MAE={va_mae:.4f}")

    # ========================================================
    # Stage 2: joint training
    # ========================================================
    print("\n====== Stage 2: Joint training ======\n")

    for p in decomposer.parameters():
        p.requires_grad = True

    opt_joint = torch.optim.Adam(
        list(decomposer.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    best_val_mae = float("inf")

    for ep in range(1, args.joint_epochs + 1):
        tr = train_joint(
            decomposer,
            predictor,
            tr_dl,
            opt_joint,
            device,
            pred_len=args.pred_len,
            w_pred=args.w_pred,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
        )
        va_mse, va_mae = eval_all(
            decomposer,
            predictor,
            va_dl,
            device,
            pred_len=args.pred_len,
        )
        print(f"[Joint {ep:03d}] train pred={tr:.4f} | val MSE={va_mse:.4f} MAE={va_mae:.4f}")

        # simple best-on-val-MAE checkpoint
        if va_mae < best_val_mae:
            best_val_mae = va_mae
            ckpt = {
                "decomposer_state": decomposer.state_dict(),
                "predictor_state": predictor.state_dict(),
                "epoch": ep,
                "dec_len": args.dec_len,
                "pred_len": args.pred_len,
            }
            torch.save(ckpt, args.out)
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
