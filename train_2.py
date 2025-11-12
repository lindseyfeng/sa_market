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
    Inputs:
      - x: raw RRP window
      - imf_win: 13 normalized IMF windows (per-mode min/max)
      - y: raw RRP at t+L
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

        # ----- raw IMFs: (T,13) -> (13,T)
        imfs_np = df[decomp_cols].to_numpy(dtype=np.float32)
        self.imfs = torch.tensor(imfs_np).transpose(0, 1)   # (13,T)

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

        mins = torch.tensor(self.imf_mins).unsqueeze(1)  # (13,1)
        rngs = (torch.tensor(self.imf_maxs) - torch.tensor(self.imf_mins)).unsqueeze(1) + 1e-12
        self.imfs_norm = (self.imfs - mins) / rngs       # (13,T) in [0,1] ideally

        # x
        if self.x_mode == "raw":
            self.x_series = self.rrp                      
        elif self.x_mode == "sum_norm":  # x is sum of normalized IMFs
            self.x_series = self.imfs_norm.sum(dim=0)      # (T,)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum_norm'")

        # optional scaler for raw RRP (useful later)
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

        # 13 normalized IMF windows (13,L)
        imf_win = self.imfs_norm[:, i:i+L]                # (13,L)

        # raw target RRP(t+L)
        y = self.rrp[i+L]                                 # scalar

        return x, imf_win, y.unsqueeze(0)                 # (1,L), (13,L), (1,)

    def scalers(self):
        d = dict(imf_mins=self.imf_mins, imf_maxs=self.imf_maxs)
        if self.rrp_min is not None:
            d.update(rrp_min=self.rrp_min, rrp_max=self.rrp_max)
        return d


def train_or_eval_epoch(model, loader, device, alpha, beta,
                        imf_mins, imf_maxs,
                        optimizer=None, clip_grad=None,
                        sum_reg: float = 0.2): 

    is_train = optimizer is not None
    model.train(is_train)

    decomposer_frozen = all(not p.requires_grad for p in model.decomposer.parameters())
    if decomposer_frozen:
        model.decomposer.eval()

    total, sum_loss, sum_d, sum_p = 0, 0.0, 0.0, 0.0

    # Per-mode scaler → broadcast across (B,K,L)
    imf_scaler = MinMaxScalerND(
        mins=torch.tensor(imf_mins, dtype=torch.float32, device=device),  # (K,)
        maxs=torch.tensor(imf_maxs, dtype=torch.float32, device=device),  # (K,)
        channel_axis=1
    )

    loss_fn_imf = torch.nn.HuberLoss(delta=10.0)

    for xb, imfs_true_norm, yb in loader:
        xb = xb.to(device)                         # (B,1,L)
        imfs_true_norm = imfs_true_norm.to(device) # (B,K,L)
        yb = yb.to(device)                         # (B,1)

        imfs_pred_norm, y_modes_norm = model(xb)           # (B,K,L), (B,K)
        y_modes_raw = imf_scaler.denorm(y_modes_norm.unsqueeze(-1)).squeeze(-1)  # (B,K)
        y_pred = y_modes_raw.sum(dim=1, keepdim=True)      # (B,1)
        imfs_pred = imf_scaler.denorm(imfs_pred_norm)      # (B,K,L)
        imfs_true = imf_scaler.denorm(imfs_true_norm)      # (B,K,L)

        loss_decomp   = loss_fn_imf(imfs_pred_norm, imfs_true_norm)
        loss_sumcons  = F.l1_loss(imfs_pred_norm.sum(dim=1), imfs_true_norm.sum(dim=1))
        
        loss_decomp1   = F.l1_loss(imfs_pred, imfs_true)
        loss_sumcons1  = F.l1_loss(imfs_pred.sum(dim=1), imfs_true.sum(dim=1))
        loss_decomp_reg1 = loss_decomp1 + sum_reg * loss_sumcons1
        loss_pred     = F.l1_loss(y_pred, yb)
        
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
        sum_d    += loss_decomp_reg1.item() * bs  # log the regularized decomp loss
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
    ap.add_argument("--seq-len", type=int, default=64)

    # Model
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--freeze-decomposer", action="store_true", default=False)

    # Train
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1)      # weight for IMF
    ap.add_argument("--beta",  type=float, default=0)      # weight for pred
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
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
        if ep > 50:
            args.alpha, args.beta = 0.2, 1
            for name, param in model.decomposer.named_parameters():
                param.requires_grad = False
            model.decomposer.eval() 
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
