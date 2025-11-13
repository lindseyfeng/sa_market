
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nvmd_autoencoder import MultiModeNVMD


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


class VMD13ReconDataset(Dataset):
    """
    For each index i, returns:
      x_raw:   (1, L)   RRP window [i .. i+L-1]
      imf_win: (K, L)   IMF window [i .. i+L-1] for all K modes
    """
    def __init__(self, df, seq_len=64, rrp_col="RRP", imf_cols=None):
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
        self.N = T - seq_len  # only need window, no t+L next

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L
        # RRP window
        x = self.rrp[i:i+L].unsqueeze(0)          # (1,L)

        # IMF window
        imf_win = self.imfs[i:i+L]                # (L,K)
        imf_win = imf_win.T                       # (K,L) to match model output

        return x, imf_win                         # DataLoader → (B,1,L), (B,K,L)


# ===============================================================
#                 REGULARIZERS (DECORR + BANDWIDTH)
# ===============================================================

def decorrelation_loss(imfs: torch.Tensor) -> torch.Tensor:
    """
    Encourage modes to be decorrelated.
    imfs: (B,K,L)
    """
    B, K, L = imfs.shape
    loss = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            ci = imfs[:, i] - imfs[:, i].mean(dim=1, keepdim=True)
            cj = imfs[:, j] - imfs[:, j].mean(dim=1, keepdim=True)
            corr = (ci * cj).mean()
            loss += corr.abs()
    return loss / (K * (K - 1) / 2)


def bandwidth_loss(imfs: torch.Tensor) -> torch.Tensor:
    """
    Encourage narrow-band IMFs via weighted FFT magnitudes.
    """
    freqs = torch.fft.rfft(imfs, dim=-1).abs()  # (B,K,F)
    w = torch.linspace(0, 1, freqs.size(-1), device=imfs.device)  # (F,)
    return (freqs * w).mean()


# ===============================================================
#                     TRAIN / EVAL EPOCH
# ===============================================================

def train_epoch_recon(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    w_corr: float = 0.0,
    w_band: float = 0.0,
):
    """
    Pure IMF reconstruction training.

    Core loss:
        L_imf = L1(IMF_pred, IMF_true)

    Total:
        L_total = L_imf + w_corr * L_corr + w_band * L_band
    """
    model.train()
    total_imf = 0.0
    n = 0

    for x, imf_win in loader:
        x = x.to(device)              # (B,1,L)
        imf_win = imf_win.to(device)  # (B,K,L)

        opt.zero_grad()
        imfs_pred, _ = model(x)       # (B,K,L), (B,1,L)

        L_imf = F.l1_loss(imfs_pred, imf_win)

        loss = L_imf
        if w_corr > 0.0:
            loss = loss + w_corr * decorrelation_loss(imfs_pred)
        if w_band > 0.0:
            loss = loss + w_band * bandwidth_loss(imfs_pred)

        loss.backward()
        opt.step()

        bs = x.size(0)
        n += bs
        total_imf += L_imf.item() * bs

    return total_imf / max(n, 1)


def eval_epoch_recon(
    model: nn.Module,
    loader: DataLoader,
    device: str,
):
    """
    Evaluate IMF reconstruction only.

    Returns:
        imf_mae: mean L1 over all IMFs and timesteps.
    """
    model.eval()
    total_imf = 0.0
    n = 0

    with torch.no_grad():
        for x, imf_win in loader:
            x = x.to(device)
            imf_win = imf_win.to(device)

            imfs_pred, _ = model(x)

            L_imf = F.l1_loss(imfs_pred, imf_win)

            bs = x.size(0)
            n += bs
            total_imf += L_imf.item() * bs

    return total_imf / max(n, 1)


# ===============================================================
#                           MAIN
# ===============================================================

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--test-csv",  default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=64)

    # Model / training
    ap.add_argument("--base",   type=int,   default=64)
    ap.add_argument("--batch",  type=int,   default=256)
    ap.add_argument("--epochs", type=int,   default=20)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--w-corr", type=float, default=0.0,
                    help="weight for decorrelation loss (0 to disable)")
    ap.add_argument("--w-band", type=float, default=0.0,
                    help="weight for bandwidth loss (0 to disable)")
    ap.add_argument("--seed",   type=int,   default=1337)

    # I/O
    ap.add_argument("--out", type=str, default="./nvmd_ae_imf_recon.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)

    # Build IMF columns from train df
    imf_cols = build_imf_cols(df_train)
    K = len(imf_cols)

    tr_ds = VMD13ReconDataset(
        df_train,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        imf_cols=imf_cols,
    )
    te_ds = VMD13ReconDataset(
        df_test,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        imf_cols=imf_cols,
    )

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    # Model
    model = MultiModeNVMD(K=K, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")

    for ep in range(1, args.epochs + 1):
        tr_imf = train_epoch_recon(
            model,
            tr_dl,
            opt,
            device,
            w_corr=args.w_corr,
            w_band=args.w_band,
        )
        te_imf = eval_epoch_recon(model, te_dl, device)

        print(
            f"[Epoch {ep:03d}] "
            f"train IMF-win MAE={tr_imf:.4f} | "
            f"test IMF-win MAE={te_imf:.4f}"
        )

        if te_imf < best:
            best = te_imf
            torch.save(model.state_dict(), args.out)
            print(f"  → saved best checkpoint (IMF-win MAE={best:.4f})")


if __name__ == "__main__":
    main()
