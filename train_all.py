import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from hybrid_spectral_nvmd import HybridSpectralNVMD
from transformer_predictor import MultiModeTransformerRRP


# =======================
#  DATASET
# =======================

class VMD13IMFNextDataset(Dataset):
    """
    For each index i, returns:

      x_raw:     raw RRP window, shape (1, L)
      imfs_true: benchmark VMD IMFs, shape (K, L)
      rrp_next:  raw RRP at time i+L, shape (1,)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        rrp_col: str = "RRP",
        K: int = 13,
    ):
        super().__init__()
        self.L = seq_len
        self.rrp_col = rrp_col
        self.K = K

        if rrp_col not in df.columns:
            raise ValueError(f"RRP column '{rrp_col}' not in dataframe")

        mode_cols = [f"Mode_{i}" for i in range(1, K)] + ["Residual"]
        for c in mode_cols:
            if c not in df.columns:
                raise ValueError(f"Missing mode column '{c}' in dataframe")
        self.mode_cols = mode_cols

        rrp = df[rrp_col].to_numpy(dtype="float32")       # (T,)
        imfs = df[mode_cols].to_numpy(dtype="float32")    # (T, K)

        self.rrp = torch.from_numpy(rrp)                  # (T,)
        self.imfs = torch.from_numpy(imfs).transpose(0, 1)  # (K, T)

        T = self.rrp.shape[0]
        # Need window [i, ..., i+L-1] and target at i+L → max start idx = T-L-1
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L

        x_raw = self.rrp[i:i+L].unsqueeze(0)     # (1, L)
        imfs_true = self.imfs[:, i:i+L]          # (K, L)
        rrp_next = self.rrp[i+L].unsqueeze(0)    # (1,)

        return x_raw, imfs_true, rrp_next


# =======================
#  TRAIN / EVAL EPOCHS
# =======================

def train_epoch(
    decomposer: HybridSpectralNVMD,
    predictor: MultiModeTransformerRRP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    w_pred: float,
    w_imf: float,
    w_rrp: float,
    w_smooth: float,
    w_ortho: float,
    clip_grad: float | None = 10.0,
):
    decomposer.train()
    predictor.train()

    total_pred = 0.0
    total_imf = 0.0
    total_rrp = 0.0
    n = 0

    for x_raw, imfs_true, rrp_next in loader:
        x_raw     = x_raw.to(device)      # (B,1,L)
        imfs_true = imfs_true.to(device)  # (B,K,L)
        rrp_next  = rrp_next.to(device)   # (B,1)

        optimizer.zero_grad(set_to_none=True)

        # 1) neural decomposition
        imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)
        # imfs_ref: (B,K,L)
        # recon_ref: (B,1,L)

        # 2) transformer prediction on refined IMFs
        rrp_next_hat = predictor(imfs_ref)       # (B,1)

        # --- losses ---
        # main forecasting loss
        loss_pred = F.mse_loss(rrp_next_hat, rrp_next)

        # IMF window reconstruction
        loss_imf = F.l1_loss(imfs_ref, imfs_true)

        # RRP window reconstruction (sum of IMFs vs raw RRP window)
        loss_rrp = F.l1_loss(recon_ref, x_raw)

        # spectral regularizers from decomposer.spectral
        loss_smooth = decomposer.spectral.spectral_smoothness_loss()
        loss_ortho  = decomposer.spectral.orthogonality_loss()

        loss = (
            w_pred   * loss_pred +
            w_imf    * loss_imf +
            w_rrp    * loss_rrp +
            w_smooth * loss_smooth +
            w_ortho  * loss_ortho
        )

        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(
                list(decomposer.parameters()) + list(predictor.parameters()),
                clip_grad
            )
        optimizer.step()

        bs = x_raw.size(0)
        n += bs
        total_pred += loss_pred.item() * bs
        total_imf  += loss_imf.item() * bs
        total_rrp  += loss_rrp.item() * bs

    denom = max(n, 1)
    return (
        total_pred / denom,   # RRP MSE
        total_imf  / denom,   # IMF-window MAE
        total_rrp  / denom,   # RRP-window MAE
    )


def eval_epoch(
    decomposer: HybridSpectralNVMD,
    predictor: MultiModeTransformerRRP,
    loader: DataLoader,
    device: str,
):
    decomposer.eval()
    predictor.eval()

    total_pred = 0.0   # MSE
    total_imf  = 0.0   # IMF-window MAE
    total_rrp  = 0.0   # RRP-window MAE
    total_mae  = 0.0   # RRP next-step MAE
    n = 0

    with torch.no_grad():
        for x_raw, imfs_true, rrp_next in loader:
            x_raw     = x_raw.to(device)
            imfs_true = imfs_true.to(device)
            rrp_next  = rrp_next.to(device)

            imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)
            rrp_next_hat = predictor(imfs_ref)

            loss_pred = F.mse_loss(rrp_next_hat, rrp_next)
            loss_imf  = F.l1_loss(imfs_ref, imfs_true)
            loss_rrp  = F.l1_loss(recon_ref, x_raw)
            mae       = F.l1_loss(rrp_next_hat, rrp_next)

            bs = x_raw.size(0)
            n += bs
            total_pred += loss_pred.item() * bs
            total_imf  += loss_imf.item() * bs
            total_rrp  += loss_rrp.item() * bs
            total_mae  += mae.item() * bs

    denom = max(n, 1)
    return (
        total_pred / denom,   # RRP MSE
        total_imf  / denom,   # IMF MAE
        total_rrp  / denom,   # RRP-window MAE
        total_mae  / denom,   # RRP next-step MAE
    )


# =======================
#  MAIN
# =======================

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train-csv", type=str,
                    default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str,
                    default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--K", type=int, default=13)

    # training
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=10.0)

    # loss weights
    ap.add_argument("--w-pred", type=float, default=1.0,
                    help="weight for next-step RRP MSE")
    ap.add_argument("--w-imf", type=float, default=1.0,
                    help="weight for IMF window L1")
    ap.add_argument("--w-rrp", type=float, default=0.1,
                    help="weight for RRP window L1")
    ap.add_argument("--w-smooth", type=float, default=1e-3,
                    help="weight for spectral smoothness")
    ap.add_argument("--w-ortho", type=float, default=1e-3,
                    help="weight for spectral orthogonality")

    # freezing option
    ap.add_argument("--freeze-decomposer", action="store_true")

    # I/O
    ap.add_argument("--out", type=str,
                    default="./hybrid_nvmd_plus_transformer.pt")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load CSVs
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    # datasets / loaders
    tr_ds = VMD13IMFNextDataset(df_tr, seq_len=args.seq_len,
                                rrp_col=args.rrp_col, K=args.K)
    va_ds = VMD13IMFNextDataset(df_va, seq_len=args.seq_len,
                                rrp_col=args.rrp_col, K=args.K)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch,
                       shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch,
                       shuffle=False)

    # -----------------------
    #  MODELS
    # -----------------------
    decomposer = HybridSpectralNVMD(
        K=args.K,
        signal_len=args.seq_len,
    ).to(device)

    predictor = MultiModeTransformerRRP(
        K=args.K,
        seq_len=args.seq_len,
        # if your class has extra args (d_model, n_heads, etc.),
        # just rely on its defaults or add them to argparse.
    ).to(device)

    if args.freeze_decomposer:
        for p in decomposer.parameters():
            p.requires_grad = False
        decomposer.eval()
        params = list(predictor.parameters())
    else:
        params = list(decomposer.parameters()) + list(predictor.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_val_mae = float("inf")

    for ep in range(1, args.epochs + 1):
        tr_mse, tr_imf, tr_rrp = train_epoch(
            decomposer,
            predictor,
            tr_dl,
            optimizer,
            device,
            w_pred=args.w_pred,
            w_imf=args.w_imf,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            clip_grad=args.clip_grad,
        )

        va_mse, va_imf, va_rrp, va_mae = eval_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train RRP MSE={tr_mse:.4f} | train IMF MAE={tr_imf:.4f} | train RRP-win MAE={tr_rrp:.4f} || "
            f"val RRP MSE={va_mse:.4f} | val IMF MAE={va_imf:.4f} | val RRP-win MAE={va_rrp:.4f} | "
            f"val next-step RRP MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            ckpt = {
                "epoch": ep,
                "decomposer_state": decomposer.state_dict(),
                "predictor_state": predictor.state_dict(),
                "val_next_step_mae": best_val_mae,
                "args": vars(args),
            }
            torch.save(ckpt, args.out)
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
