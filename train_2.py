import argparse, os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM


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




class Decomp13Dataset(Dataset):
    """
    13 decomposition signals:
        Mode_1 ... Mode_12, Residual

    Each IMF is normalized by its own min/max.
    x_norm = sum of normalized 13 signals.
    y_true = raw RRP value.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        decomp_cols: list[str],     # 13 input columns
        seq_len: int = 9,
        target_col: str = "RRP",
        sig_mins: np.ndarray | None = None,
        sig_maxs: np.ndarray | None = None,
        imf_mins: np.ndarray | None = None,
        imf_maxs: np.ndarray | None = None,
    ):
        self.decomp_cols = decomp_cols
        self.K = len(decomp_cols)      # = 13
        self.L = seq_len

        # ----- raw IMF matrix: (T,13) -> (13,T)
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)
        self.imfs = torch.tensor(imfs_np).transpose(0,1)    # (13,T)

        # ----- raw target series RRP
        target_np = df[target_col].to_numpy(dtype=np.float32)
        self.target = torch.tensor(target_np, dtype=torch.float32)  # (T,)
        T = len(self.target)

        # ----- per-signal min/max (for each IMF)
        if sig_mins is None or sig_maxs is None:
            sig_mins = imfs_np.min(axis=0)    # (13,)
            sig_maxs = imfs_np.max(axis=0)
        self.sig_mins = np.asarray(sig_mins)
        self.sig_maxs = np.asarray(sig_maxs)

        # ----- per-IMF min/max (same for this data)
        if imf_mins is None or imf_maxs is None:
            imf_mins = self.imfs.min(dim=1).values.numpy()
            imf_maxs = self.imfs.max(dim=1).values.numpy()
        self.imf_mins = np.asarray(imf_mins)
        self.imf_maxs = np.asarray(imf_maxs)

        # ----- normalize IMFs per mode
        mins = torch.tensor(self.imf_mins).unsqueeze(1)     # (13,1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs - mins) / rngs          # (13,T)

        # ----- x_norm = sum of 13 normalized IMFs
        self.signal_norm = self.imfs_norm.sum(dim=0)        # (T,)

        # ----- number of windows
        self.N = T - self.L - 1
        if self.N < 0:
            self.N = 0

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        L = self.L

        # x_norm window (1,L)
        x = self.signal_norm[i:i+L].unsqueeze(0)

        # 13 IMF windows (13,L)
        imf_win = self.imfs_norm[:, i:i+L]

        # raw target RRP(t+L)
        y = self.target[i+L]

        return x, imf_win, y.unsqueeze(0)    # (1,L), (13,L), (1,)

    def scalers(self):
        """Return ONLY what training needs — per-IMF min/max."""
        return dict(
            imf_mins=self.imf_mins,   # (13,)
            imf_maxs=self.imf_maxs,   # (13,)
        )


def train_or_eval_epoch(model, loader, device, alpha, beta,
                        imf_mins, imf_maxs,
                        optimizer=None, clip_grad=None,
                        sum_reg: float = 1.0): 

    is_train = optimizer is not None
    model.train(is_train)

    total, sum_loss, sum_d, sum_p = 0, 0.0, 0.0, 0.0

    # Per-mode scaler → broadcast across (B,K,L)
    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),  # (K,)
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),  # (K,)
        channel_axis=1
    )

    loss_fn_imf = torch.nn.HuberLoss(delta=1.0)

    for xb, imfs_true_norm, yb in loader:
        xb = xb.to(device)                         # (B,1,L)
        imfs_true_norm = imfs_true_norm.to(device) # (B,K,L)
        yb = yb.to(device)                         # (B,1)

        imfs_pred_norm, y_pred = model(xb)         # (B,K,L), (B,1)

        # Denorm for losses on original scale
        imfs_pred = imf_scaler.denorm(imfs_pred_norm)  # (B,K,L)
        imfs_true = imf_scaler.denorm(imfs_true_norm)  # (B,K,L)

        loss_decomp =  F.mse_loss(imfs_pred, imfs_true)


        sig_pred = imfs_pred.sum(dim=1)  # (B,L)
        sig_true = imfs_true.sum(dim=1)  # (B,L)
        loss_sumcons = F.l1_loss(sig_pred, sig_true)

        loss_pred = loss_fn_imf(y_pred, yb)
        
        loss_decomp_reg = loss_decomp + sum_reg * loss_sumcons
        loss = alpha * loss_decomp_reg + beta * loss_pred

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = xb.size(0)
        total += bs
        sum_loss += loss.item() * bs
        sum_d    += loss_decomp_reg.item() * bs  # log the regularized decomp loss
        sum_p    += loss_pred.item() * bs

    return (sum_loss / max(total,1),
            sum_d    / max(total,1),
            sum_p    / max(total,1))


def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)]
    cols.append("Residual")
    # validate they exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 13-series setting: {missing}")
    return cols


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-csv", default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--seq-len", type=int, default=512)

    # Model
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--freeze-decomposer", action="store_true", default=False)

    # Train
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=15e-4)
    ap.add_argument("--alpha", type=float, default=1)      # weight for IMF MSE
    ap.add_argument("--beta",  type=float, default=0.1)      # weight for pred MSE
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--num-workers", type=int, default=0)

    # Logs / save
    ap.add_argument("--outdir", type=str, default="./runs_nvmd_mrc_bilstm")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataframes
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)


    decomp_cols = build_default_13(df_tr)

    trds = Decomp13Dataset(
    df=df_tr,
    decomp_cols=decomp_cols,        
    seq_len=args.seq_len,          
    target_col="RRP"              
    )

    sc = trds.scalers()  

    vads = Decomp13Dataset(
    df=df_va,
    decomp_cols=decomp_cols,       
    seq_len=args.seq_len,       
    target_col="RRP",
    imf_mins=sc["imf_mins"],
    imf_maxs=sc["imf_maxs"], 
                                    
    )

    trdl = DataLoader(trds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    vadl = DataLoader(vads, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    K = len(decomp_cols)  # 13
    model = NVMD_MRC_BiLSTM(
        signal_len=args.seq_len,
        K=K,
        base=args.base,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        freeze_decomposer=args.freeze_decomposer
    ).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train loop
    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_ld, tr_lp = train_or_eval_epoch(
            model, trdl, device,
            alpha=args.alpha, beta=args.beta,    
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
            optimizer=opt, clip_grad=args.clip_grad
        )
        va_loss, va_ld, va_lp = train_or_eval_epoch(
            model, vadl, device,
            alpha=args.alpha, beta=args.beta,
            imf_mins=sc["imf_mins"], imf_maxs=sc["imf_maxs"],
            optimizer=None
        )
        if ep > 20:
            args.alpha, args.beta = 0.2, 1
            
        print(f"[Epoch {ep:02d}] "
              f"train: total={tr_loss:.6f} decomp={tr_ld:.6f} pred={tr_lp:.6f} | "
              f"val: total={va_loss:.6f} decomp={va_ld:.6f} pred={va_lp:.6f}")

        if va_lp < best_val:
            best_val = va_lp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # simple checkpoint each epoch
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
        print(f"Saved best checkpoint: val_total={best_val:.6f} → {os.path.join(args.outdir,'best.pt')}")
    else:
        print("No best state captured; nothing saved.")

if __name__ == "__main__":
    main()
