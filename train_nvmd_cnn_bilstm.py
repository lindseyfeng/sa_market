#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  # your joint model: decomposer + predictors


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MinMaxScalerND:
    """
    Per-channel min-max scaler for arbitrary ND tensors.
    We'll use it to denorm per-mode scalar predictions back to raw IMF scale.
    """
    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor, channel_axis: int = 1, eps: float = 1e-12):
        assert mins.numel() == maxs.numel()
        self.mins = mins
        self.maxs = maxs
        self.axis = channel_axis
        self.eps  = eps

    def _view_shape(self, x: torch.Tensor):
        shape = [1] * x.ndim
        shape[self.axis] = self.mins.numel()
        return shape

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        mins = self.mins.view(*self._view_shape(x))
        rng  = (self.maxs - self.mins).view(*self._view_shape(x)).clamp_min(self.eps)
        return (x - mins) / rng

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        mins = self.mins.view(*self._view_shape(x))
        rng  = (self.maxs - self.mins).view(*self._view_shape(x)).clamp_min(self.eps)
        return x * rng + mins


# -------------------------
# Dataset: IMFs + RRP
# -------------------------
class Decomp13Dataset(Dataset):
    """
    Inputs:
      - x: dummy (1, L) – NVMD_MRC_BiLSTM will ignore if you don't use it,
           but we keep it to match the (x, imfs, y) interface you already have.
      - imf_win_norm: (K, L) per-sample, normalized per-IMF using global min/max
      - y: raw RRP(t+L), shape (1,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],
        seq_len: int = 9,
        target_col: str = "RRP",
        imf_mins: np.ndarray | None = None,
        imf_maxs: np.ndarray | None = None,
    ):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)
        self.L = seq_len

        # raw IMFs: (T,K) -> (K,T)
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)
        self.imfs = torch.tensor(imfs_np).transpose(0, 1)   # (K,T)

        # raw RRP series
        rrp_np = df[target_col].to_numpy(dtype=np.float32)
        self.rrp = torch.tensor(rrp_np, dtype=torch.float32)  # (T,)
        T = len(self.rrp)

        # per-IMF min/max
        if imf_mins is None or imf_maxs is None:
            imf_mins = self.imfs.min(dim=1).values.numpy()
            imf_maxs = self.imfs.max(dim=1).values.numpy()
        self.imf_mins = np.asarray(imf_mins)
        self.imf_maxs = np.asarray(imf_maxs)

        mins = torch.tensor(self.imf_mins).unsqueeze(1)  # (K,1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs - mins) / rngs       # (K,T) in [0,1]

        # windows with L history + 1-step ahead
        self.N = T - self.L - 1
        if self.N < 0:
            self.N = 0

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L

        # normalized GT IMFs (K,L)
        imf_win = self.imfs_norm[:, i:i+L]  # (K,L)

        # raw RRP target at t+L
        y = self.rrp[i+L]                   # scalar

        # dummy x: shape (1,L) – you can swap to raw RRP window if your NVMD_MRC_BiLSTM uses it
        x_dummy = torch.zeros(1, L, dtype=torch.float32)

        return x_dummy, imf_win, y.unsqueeze(0)  # (1,L), (K,L), (1,)

    def scalers(self):
        return dict(imf_mins=self.imf_mins, imf_maxs=self.imf_maxs)


def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in dataframe.")
    return cols


# -------------------------
# Joint train/eval (MSE only)
# -------------------------
def train_or_eval_epoch(
    model,
    loader,
    device,
    alpha,          # weight for IMF reconstruction MSE
    beta,           # weight for final prediction MSE
    imf_mins,
    imf_maxs,
    optimizer=None,
    clip_grad=None,
):
    """
    Joint training:

      - Total: alpha * IMF_MSE + beta * Pred_MSE
    """
    is_train = optimizer is not None
    model.train(is_train)

    total, sum_loss, sum_dec, sum_pred = 0, 0.0, 0.0, 0.0

    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1
    )

    for xb, imfs_true_norm, yb in loader:
        xb = xb.to(device)                         # (B,1,L)
        imfs_true_norm = imfs_true_norm.to(device) # (B,K,L)
        yb = yb.to(device)                         # (B,1)

        # forward: NVMD_MRC_BiLSTM returns normalized IMFs + normalized per-mode scalars
        imfs_pred_norm, y_modes_norm = model(xb)   # (B,K,L), (B,K) or (B,K,1)/(B,1,K)

        # unify y_modes_norm to (B,K)
        if y_modes_norm.dim() == 3:
            if y_modes_norm.size(-1) == 1:
                y_modes_norm = y_modes_norm.squeeze(-1)
            elif y_modes_norm.size(1) == 1:
                y_modes_norm = y_modes_norm.squeeze(1)

        # # IMF MSE in normalized space
        # loss_decomp = F.mse_loss(imfs_pred_norm, imfs_true_norm)

        # denorm per-mode scalar predictions to raw IMF scale
        y_modes_raw = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
        y_pred = y_modes_raw.sum(dim=1, keepdim=True)                            # (B,1)

        # final RRP prediction MSE
        loss_pred = F.l1_loss(y_pred, yb)

        loss = loss_pred

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = xb.size(0)
        total    += bs
        sum_loss += loss.item()        * bs
        sum_dec  += loss_decomp.item() * bs
        sum_pred += loss_pred.item()   * bs

    return (
        sum_loss / max(total, 1),
        sum_dec  / max(total, 1),
        sum_pred / max(total, 1),
    )


# -------------------------
# Main: second-stage joint training
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021_with_EWT.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022_with_EWT.csv")
    ap.add_argument("--seq-len", type=int, default=8)

    # Model
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for IMF MSE")
    ap.add_argument("--beta",  type=float, default=1.0, help="weight for prediction MSE")
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    # Decomposer checkpoint (first-stage NVMD_Autoencoder)
    ap.add_argument("--dec-ckpt", type=str,  default="./runs_nvmd_autoencoder_raw/best.pt",
                    help="Path to pretrained NVMD_Autoencoder checkpoint (with 'model_state')")

    # Logs / save
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_joint_mse")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataframes
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    # IMF columns
    decomp_cols = build_default_13(df_tr)

    # Datasets
    trds = Decomp13Dataset(
        df=df_tr,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col="RRP",
    )
    sc = trds.scalers()  # contains imf_mins, imf_maxs

    vads = Decomp13Dataset(
        df=df_va,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col="RRP",
        imf_mins=sc["imf_mins"],
        imf_maxs=sc["imf_maxs"],
    )

    # Loaders
    pin = (device == "cuda")
    trdl = DataLoader(trds, batch_size=args.batch, shuffle=True,
                      num_workers=args.num_workers, pin_memory=pin)
    vadl = DataLoader(vads, batch_size=args.batch, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pin)

    # Joint model: decomposer + predictors
    K = len(decomp_cols)
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len,  # this is the length your predictor expects
        K=K,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        freeze_decomposer=False,  # we WANT gradients through decomposer in this stage
    ).to(device)

    # -------------------------
    # Load pretrained decomposer
    # -------------------------
    print(f"Loading decomposer checkpoint from: {args.dec_ckpt}")
    dec_ck = torch.load(args.dec_ckpt, map_location="cpu")

    # assume it was saved as {"model_state": nvmd_state_dict, ...}
    dec_state = dec_ck.get("model_state", dec_ck)

    # handle potential "module." prefixes
    cleaned_state = {}
    for k, v in dec_state.items():
        if k.startswith("module."):
            cleaned_state[k[len("module."):]] = v
        else:
            cleaned_state[k] = v

    missing, unexpected = model.decomposer.load_state_dict(cleaned_state, strict=False)
    if missing:
        print("Warning: missing keys in decomposer state_dict:", missing)
    if unexpected:
        print("Warning: unexpected keys in decomposer state_dict:", unexpected)

    print("Decomposer weights loaded. Training jointly with predictors using MSE losses.")

    # Optimizer: ALL params (decomposer + predictors)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_ld, tr_lp = train_or_eval_epoch(
            model, trdl, device,
            alpha=args.alpha, beta=args.beta,
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
            optimizer=opt,
            clip_grad=args.clip_grad,
        )

        va_loss, va_ld, va_lp = train_or_eval_epoch(
            model, vadl, device,
            alpha=args.alpha, beta=args.beta,
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
            optimizer=None,
            clip_grad=None,
        )

        print(
            f"[Epoch {ep:02d}] "
            f"train: total={tr_loss:.6f} decomp={tr_ld:.6f} pred={tr_lp:.6f} | "
            f"val: total={va_loss:.6f} decomp={va_ld:.6f} pred={va_lp:.6f}"
        )

        # track best by prediction loss (rmse proxy)
        if va_lp < best_val:
            best_val = va_lp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ck = {
            "epoch": ep,
            "val_best": best_val,
            "model_state": model.state_dict(),
            "args": vars(args),
            "scalers": sc,
            "decomp_cols": decomp_cols,
        }
        torch.save(ck, os.path.join(args.outdir, f"epoch_{ep:02d}.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "epoch": "best",
                "val_best": best_val,
                "model_state": best_state,
                "args": vars(args),
                "scalers": sc,
                "decomp_cols": decomp_cols,
            },
            os.path.join(args.outdir, "best.pt"),
        )
        print(f"Saved best joint MSE checkpoint with val_pred_MSE={best_val:.6f}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
