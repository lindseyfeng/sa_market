#!/usr/bin/env python3
import argparse, os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  # uses NVMD_Autoencoder inside


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MinMaxScalerND:
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
# Dataset: normalized IMFs + raw RRP
# -------------------------
class Decomp13Dataset(Dataset):
    """
    Inputs per item:
      - x: raw RRP window, shape (1, L)
      - imf_win: 13 normalized IMF windows (per-mode min/max), shape (K, L)
      - y: raw RRP at t+L, shape (1,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],   # 13 IMF columns
        seq_len: int = 9,
        target_col: str = "RRP",
        imf_mins: np.ndarray | None = None,
        imf_maxs: np.ndarray | None = None,
        x_mode: str = "raw",      # "raw" or "sum_norm"
        add_rrp_scaler: bool = False,
    ):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)   # 13
        self.L = seq_len
        self.x_mode = x_mode

        # ----- raw IMFs: (T,K) -> (K,T)
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)
        self.imfs = torch.tensor(imfs_np).transpose(0, 1)   # (K,T)

        # raw target series (RRP)
        rrp_np = df[target_col].to_numpy(dtype=np.float32)
        self.rrp = torch.tensor(rrp_np, dtype=torch.float32)   # (T,)
        T = len(self.rrp)

        #  per-IMF min/max (for normalization)
        if imf_mins is None or imf_maxs is None:
            imf_mins = self.imfs.min(dim=1).values.numpy()
            imf_maxs = self.imfs.max(dim=1).values.numpy()
        self.imf_mins = np.asarray(imf_mins)
        self.imf_maxs = np.asarray(imf_maxs)

        mins = torch.tensor(self.imf_mins).unsqueeze(1)  # (K,1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs - mins) / rngs       # (K,T) in [0,1] ideally

        # x
        if self.x_mode == "raw":
            self.x_series = self.rrp
        elif self.x_mode == "sum_norm":
            self.x_series = self.imfs_norm.sum(dim=0)    # (T,)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum_norm'")

        # optional scaler for raw RRP (not strictly needed here)
        if add_rrp_scaler:
            self.rrp_min = float(self.rrp.min().item())
            self.rrp_max = float(self.rrp.max().item())
        else:
            self.rrp_min = self.rrp_max = None

        self.N = T - self.L - 1
        if self.N < 0:
            self.N = 0

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L
        # x window (1,L)
        x = self.x_series[i:i+L].unsqueeze(0)             # (1,L)

        # 13 normalized IMF windows (K,L)
        imf_win = self.imfs_norm[:, i:i+L]                # (K,L)

        # raw target RRP(t+L)
        y = self.rrp[i+L]                                 # scalar

        return x, imf_win, y.unsqueeze(0)                 # (1,L), (K,L), (1,)

    def scalers(self):
        d = dict(imf_mins=self.imf_mins, imf_maxs=self.imf_maxs)
        if self.rrp_min is not None:
            d.update(rrp_min=self.rrp_min, rrp_max=self.rrp_max)
        return d


# -------------------------
# Core train/eval epoch
# -------------------------
def train_or_eval_epoch(
    model,
    loader,
    device,
    alpha,
    beta,
    imf_mins,
    imf_maxs,
    optimizer=None,          # used only when separate=False (joint)
    clip_grad=None,
    sum_reg: float = 0.2,
    separate: bool = False,  # if True: decomposer & predictors trained separately
    opt_dec=None,
    opt_pred=None,
):
    """
    If separate=False (joint):
        - x → decomposer → predictors
        - Loss = alpha * IMF recon + beta * price pred

    If separate=True:
        A: Decomposer step: IMF recon only (normalized IMFs)
        B: Predictor step: use GT normalized IMFs as input to predictors,
                           predict price and update predictor params only.
    """
    is_train_joint = (optimizer is not None) and (not separate)

    if separate:
        model.train(True)   # we'll control which params get grads
    else:
        model.train(is_train_joint)

    # if decomposer is frozen, keep it in eval mode
    decomposer_frozen = all(not p.requires_grad for p in model.decomposer.parameters())
    if decomposer_frozen:
        model.decomposer.eval()

    total, log_loss, log_dec, log_pred = 0, 0.0, 0.0, 0.0

    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),
        channel_axis=1
    )

    loss_fn_imf = torch.nn.HuberLoss(delta=10.0)

    for xb, imfs_true_norm, yb in loader:
        xb = xb.to(device)                         # (B,1,L)
        imfs_true_norm = imfs_true_norm.to(device) # (B,K,L)
        yb = yb.to(device)                         # (B,1)

        if not separate:
            # ----------------------------
            # JOINT STEP
            # ----------------------------
            imfs_pred_norm, y_modes_norm = model(xb)   # (B,K,L), (B,K or B,K,1)
            if y_modes_norm.dim() == 3:
                if y_modes_norm.size(-1) == 1:
                    y_modes_norm = y_modes_norm.squeeze(-1)
                elif y_modes_norm.size(1) == 1:
                    y_modes_norm = y_modes_norm.squeeze(1)

            # decomp loss in normalized space
            loss_decomp   = loss_fn_imf(imfs_pred_norm, imfs_true_norm)
            loss_sumcons  = F.l1_loss(imfs_pred_norm.sum(dim=1), imfs_true_norm.sum(dim=1))
            loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons

            # price prediction loss in raw space
            y_modes_raw = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
            y_pred = y_modes_raw.sum(dim=1, keepdim=True)                            # (B,1)
            loss_pred = F.l1_loss(y_pred, yb)

            loss = alpha * loss_decomp_reg + beta * loss_pred

            if is_train_joint:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            bs = xb.size(0)
            total   += bs
            log_loss += loss.item() * bs
            log_dec  += loss_decomp_reg.item() * bs
            log_pred += loss_pred.item() * bs

        else:
            # ----------------------------
            # SEPARATE TRAINING
            # ----------------------------

            # A) Decomposer-only step (IMF recon)
            if opt_dec is not None:
                model.train(True)
                # NVMD_MRC_BiLSTM forward applies sigmoid inside; here call decomposer directly
                imfs_pred_raw = model.decomposer(xb)               # (B,K,L) raw
                imfs_pred_norm = torch.sigmoid(imfs_pred_raw)      # match normalized targets

                loss_decomp   = loss_fn_imf(imfs_pred_norm, imfs_true_norm)
                loss_sumcons  = F.l1_loss(imfs_pred_norm.sum(dim=1), imfs_true_norm.sum(dim=1))
                loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons

                opt_dec.zero_grad(set_to_none=True)
                loss_decomp_reg.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.decomposer.parameters(), clip_grad)
                opt_dec.step()
            else:
                with torch.no_grad():
                    imfs_pred_raw = model.decomposer(xb)
                    imfs_pred_norm = torch.sigmoid(imfs_pred_raw)
                    loss_decomp   = loss_fn_imf(imfs_pred_norm, imfs_true_norm)
                    loss_sumcons  = F.l1_loss(imfs_pred_norm.sum(dim=1), imfs_true_norm.sum(dim=1))
                    loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons

            # B) Predictor-only step (teacher forcing with GT IMFs)
            if opt_pred is not None:
                model.train(True)
                y_list = []
                K = imfs_true_norm.size(1)
                for k in range(K):
                    yk_norm = model.predictors[k](imfs_true_norm[:, k:k+1, :])  # (B,1)
                    y_list.append(yk_norm)
                y_modes_norm = torch.cat(y_list, dim=1)  # (B,K)
                y_modes_raw  = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
                y_pred       = y_modes_raw.sum(dim=1, keepdim=True)                       # (B,1)
                loss_pred    = F.l1_loss(y_pred, yb)

                opt_pred.zero_grad(set_to_none=True)
                loss_pred.backward()
                if clip_grad is not None:
                    preds_params = [p for n, p in model.named_parameters() if n.startswith("predictors") and p.requires_grad]
                    if preds_params:
                        torch.nn.utils.clip_grad_norm_(preds_params, clip_grad)
                opt_pred.step()
            else:
                with torch.no_grad():
                    y_list = []
                    K = imfs_true_norm.size(1)
                    for k in range(K):
                        yk_norm = model.predictors[k](imfs_true_norm[:, k:k+1, :])
                        y_list.append(yk_norm)
                    y_modes_norm = torch.cat(y_list, dim=1)
                    y_modes_raw  = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)
                    y_pred       = y_modes_raw.sum(dim=1, keepdim=True)
                    loss_pred    = F.l1_loss(y_pred, yb)

            bs = xb.size(0)
            total    += bs
            log_dec  += loss_decomp_reg.item() * bs
            log_pred += loss_pred.item() * bs
            log_loss += (alpha * loss_decomp_reg.item() + beta * loss_pred.item()) * bs

    return (
        log_loss / max(total, 1),
        log_dec  / max(total, 1),
        log_pred / max(total, 1),
    )


# -------------------------
# Misc helpers
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
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--x-mode", type=str, choices=["raw", "sum_norm"], default="raw")

    # Model
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--freeze-decomposer", action="store_true", default=False)

    # Pretrained NVMD
    ap.add_argument("--nvmd-ckpt", type=str, default="runs_nvmd_autoencoder_raw/best.pt",
                    help="Checkpoint from nvmd_autoencoder pretraining")

    # Training schedule
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.2)    # weight for IMF loss
    ap.add_argument("--beta",  type=float, default=1.0)    # weight for pred loss (joint phase)
    ap.add_argument("--sum-reg", type=float, default=0.2)
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--warmup-epochs", type=int, default=50,
                    help="Number of epochs for separate dec/pred warmup")

    # Logs / save
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_mrc_bilstm_stage2")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataframes
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    decomp_cols = build_default_13(df_tr)

    # Datasets
    trds = Decomp13Dataset(
        df=df_tr,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col="RRP",
        x_mode=args.x_mode,
    )
    sc = trds.scalers()

    vads = Decomp13Dataset(
        df=df_va,
        decomp_cols=decomp_cols,
        seq_len=args.seq_len,
        target_col="RRP",
        imf_mins=sc["imf_mins"],
        imf_maxs=sc["imf_maxs"],
        x_mode=args.x_mode,
    )

    pin = (device == "cuda")
    trdl = DataLoader(trds, batch_size=args.batch, shuffle=True,
                      num_workers=args.num_workers, pin_memory=pin)
    vadl = DataLoader(vads, batch_size=args.batch, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pin)

    # Model
    K = len(decomp_cols)
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len,
        K=K,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        freeze_decomposer=args.freeze_decomposer
    ).to(device)

    # -------------------------
    # Load pretrained decomposer
    # -------------------------
    if args.nvmd_ckpt and os.path.isfile(args.nvmd_ckpt):
        ck_n = torch.load(args.nvmd_ckpt, map_location="cpu", weights_only=False)
        # try to be robust: either full state_dict, or dict with "model_state"
        if isinstance(ck_n, dict) and "model_state" in ck_n:
            nvmd_state = ck_n["model_state"]
        else:
            nvmd_state = ck_n
        missing, unexpected = model.decomposer.load_state_dict(nvmd_state, strict=False)
        print(f"[INFO] Loaded NVMD decomposer from {args.nvmd_ckpt}")
        if missing:
            print(f"[INFO] Missing keys in decomposer: {missing}")
        if unexpected:
            print(f"[INFO] Unexpected keys in decomposer: {unexpected}")
    else:
        print("[WARN] No NVMD checkpoint found / provided. Decomposer starts from scratch.")

    # Optimizers for separate training
    opt_dec  = torch.optim.Adam(model.decomposer.parameters(), lr=args.lr)

    def pred_params(m):
        for n, p in m.named_parameters():
            if n.startswith("predictors") and p.requires_grad:
                yield p

    opt_pred = torch.optim.Adam(list(pred_params(model)), lr=args.lr)

    opt_joint = None
    best_val = float("inf")
    best_state = None

    warmup = args.warmup_epochs

    for ep in range(1, args.epochs + 1):
        if ep == warmup + 1:
            # after warmup switch to joint fine-tuning
            args.alpha, args.beta = 0.2, 1.0
            for p in model.decomposer.parameters():
                p.requires_grad = True
            model.decomposer.train(True)
            opt_joint = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr
            )
            print(f"[INFO] Switched to joint training at epoch {ep}. alpha={args.alpha}, beta={args.beta}")

        # TRAIN
        if ep <= warmup:
            tr_loss, tr_ld, tr_lp = train_or_eval_epoch(
                model, trdl, device,
                alpha=args.alpha, beta=args.beta,
                imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
                optimizer=None,
                clip_grad=args.clip_grad,
                sum_reg=args.sum_reg,
                separate=True,
                opt_dec=opt_dec,
                opt_pred=opt_pred
            )
        else:
            tr_loss, tr_ld, tr_lp = train_or_eval_epoch(
                model, trdl, device,
                alpha=args.alpha, beta=args.beta,
                imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
                optimizer=opt_joint,
                clip_grad=args.clip_grad,
                sum_reg=args.sum_reg,
                separate=False
            )

        # VAL (always joint-style forward, no grads)
        va_loss, va_ld, va_lp = train_or_eval_epoch(
            model, vadl, device,
            alpha=args.alpha, beta=args.beta,
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
            optimizer=None,
            clip_grad=None,
            sum_reg=args.sum_reg,
            separate=False
        )

        print(f"[Epoch {ep:02d}] "
              f"train: total={tr_loss:.6f} decomp={tr_ld:.6f} pred={tr_lp:.6f} | "
              f"val: total={va_loss:.6f} decomp={va_ld:.6f} pred={va_lp:.6f}")

        # track best by prediction loss
        if va_lp < best_val:
            best_val = va_lp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # per-epoch checkpoint
        ck = {
            "epoch": ep,
            "val_best": best_val,
            "model_state": model.state_dict(),
            "args": vars(args),
            "scalers": sc,
            "decomp_cols": decomp_cols,
        }
        torch.save(ck, os.path.join(args.outdir, f"epoch_{ep:02d}.pt"))

    # Save best
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({
            "epoch": "best",
            "val_best": best_val,
            "model_state": best_state,
            "args": vars(args),
            "scalers": sc,
            "decomp_cols": decomp_cols,
        }, os.path.join(args.outdir, "best.pt"))
        print(f"Saved best checkpoint → {os.path.join(args.outdir, 'best.pt')}")
    else:
        print("No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
