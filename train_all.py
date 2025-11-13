

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nvmd_cnn_bilstm import MultiModeNVMD_MRC_BiLSTM 
from nvmd_cnn_bilstm import NVMD_MRC_BiLSTM  


def build_multimode_from_checkpoints(
    ckpt_dir: str,
    mode_cols: list[str],
    use_sigmoid: bool,
    device: str,
) -> MultiModeNVMD_MRC_BiLSTM:
    """
    For each mode_col, load its *_best.pt, read saved args, construct
    a NVMD_MRC_BiLSTM with those hyperparams, load weights, and
    wrap everything in MultiModeNVMD_MRC_BiLSTM.
    """
    models = []
    seq_len_ref = None

    for mode_col in mode_cols:
        ckpt_path = os.path.join(ckpt_dir, f"{mode_col}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu")

        if "model_state" in state:
            sd = state["model_state"]
        else:
            sd = state

        args_m = state.get("args", {})
        # fallbacks in case some keys are missing
        seq_len = args_m.get("seq_len") or args_m.get("seq-len")
        base = args_m.get("base", 128)
        lstm_hidden = args_m.get("lstm_hidden", 128)
        lstm_layers = args_m.get("lstm_layers", 3)
        bidirectional = args_m.get("bidirectional", True)

        if seq_len_ref is None:
            seq_len_ref = seq_len
        else:
            assert seq_len == seq_len_ref, \
                f"seq_len mismatch: {seq_len} vs {seq_len_ref}"

        # build per-mode model with its own hyperparams
        m = NVMD_MRC_BiLSTM(
            signal_len=seq_len,
            base=base,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            freeze_decomposer=False,
            use_sigmoid=use_sigmoid,
        ).to(device)

        m.load_state_dict(sd, strict=False)
        models.append(m)
        print(f"[build] loaded {mode_col} with base={base}, hidden={lstm_hidden}, "
              f"layers={lstm_layers}, bidir={bidirectional}")

    # now wrap all submodels into one multimode model
    multimode = MultiModeNVMD_MRC_BiLSTM(models=models, use_sigmoid=use_sigmoid).to(device)
    return multimode



def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PriceDataset(Dataset):
    """
    For each window i..i+L-1:

      x_raw:    input series window, shape (1, L)
      rrp_next: raw RRP at time t+L, shape (1,)

    If x_mode == "raw":
        x_raw = RRP
    If x_mode == "sum":
        x_raw = sum of all IMF columns (Mode_1..Mode_12 + Residual)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        rrp_col: str = "RRP",
        seq_len: int = 64,
        x_mode: str = "raw",         # "raw" or "sum"
        all_decomp_cols: list[str] | None = None,
    ):
        super().__init__()
        self.L = seq_len
        self.x_mode = x_mode

        if rrp_col not in df.columns:
            raise ValueError(f"rrp_col '{rrp_col}' not in dataframe.")

        rrp_np = df[rrp_col].to_numpy(dtype=np.float32)   # (T,)
        self.rrp = torch.from_numpy(rrp_np).contiguous()  # (T,)
        T = self.rrp.shape[0]

        if x_mode == "raw":
            self.x_series = self.rrp                      # (T,)
        elif x_mode == "sum":
            if all_decomp_cols is None:
                raise ValueError("all_decomp_cols must be provided when x_mode='sum'")
            imfs_np_all = df[all_decomp_cols].to_numpy(dtype=np.float32)  # (T,K_all)
            imfs_all = torch.from_numpy(imfs_np_all).transpose(0, 1).contiguous()  # (K_all, T)
            self.x_series = imfs_all.sum(dim=0)           # (T,)
        else:
            raise ValueError("x_mode must be 'raw' or 'sum'")

        # Need t+L for target
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        x_raw = self.x_series[i:i+L].unsqueeze(0)   # (1, L)
        rrp_next = self.rrp[i+L].unsqueeze(0)       # (1,)
        return x_raw, rrp_next


def build_default_13(df: pd.DataFrame) -> list[str]:
    cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing IMF columns: {missing}")
    return cols


def default_mode_cols() -> list[str]:
    return [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]


def load_per_mode_checkpoints(
    model: MultiModeNVMD_MRC_BiLSTM,
    ckpt_dir: str,
    mode_cols: list[str],
):
    assert len(mode_cols) == model.K, \
        f"model.K={model.K}, but got {len(mode_cols)} mode_cols."

    for k, mode_col in enumerate(mode_cols):
        ckpt_path = os.path.join(ckpt_dir, f"{mode_col}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {mode_col}: {ckpt_path}")

        print(f"[load] mode {k} ({mode_col}) from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")

        if isinstance(state, dict) and "model_state" in state:
            sd = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state  # assume plain state_dict

        # strict=False to ignore optimizer keys etc.
        model.models[k].load_state_dict(sd, strict=False)

def train_or_eval_epoch_rrp(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
):
    """
    One epoch over loader.

    Loss: L1(real RRP, predicted RRP) on raw scale.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw = x_raw.to(device)          # (B,1,L)
        rrp_next = rrp_next.to(device)    # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # MultiMode forward; we only care about the final price
        _, _, y_price = model(x_raw)      # y_price: (B,1)

        loss = F.l1_loss(y_price, rrp_next)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_loss_sum += loss.item() * bs

    denom = max(n_samples, 1)
    return total_loss_sum / denom  # mean L1

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str,
                    default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv", type=str,
                    default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col", type=str, default="RRP")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--x-mode", type=str, choices=["raw", "sum"], default="raw",
                    help="Model input: raw RRP window (default) or sum of raw IMFs")

    # Model
    ap.add_argument("--K", type=int, default=13,
                    help="Number of modes, e.g. 12 IMFs + Residual = 13")
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--use-sigmoid", action="store_true", default=True)

    # Training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)

    # Checkpoints / output
    ap.add_argument("--ckpt-dir", type=str,
                    default="./runs_nvmd_cnn_bilstm_mode",
                    help="Directory containing per-mode *_best.pt checkpoints")
    ap.add_argument("--outdir", type=str,
                    default="./runs_nvmd_multimode_price")
    ap.add_argument("--save-every", type=int, default=1)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")

    # Load data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    all_decomp_cols = None
    if args.x_mode == "sum":
        all_decomp_cols = build_default_13(df_tr)  # Mode_1..Mode_12 + Residual

    # Datasets / loaders
    tr_ds = PriceDataset(
        df=df_tr,
        rrp_col=args.rrp_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
        all_decomp_cols=all_decomp_cols,
    )
    va_ds = PriceDataset(
        df=df_va,
        rrp_col=args.rrp_col,
        seq_len=args.seq_len,
        x_mode=args.x_mode,
        all_decomp_cols=all_decomp_cols,
    )

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

    print(f"[info] Train samples: {len(tr_ds)}, Val samples: {len(va_ds)}")

    mode_cols = default_mode_cols()  # ["Mode_1", ..., "Mode_12", "Residual"]

    mode_cols = [f"Mode_{i}" for i in range(1, 13)] + ["Residual"]  # or whatever you use

    model = build_multimode_from_checkpoints(
        ckpt_dir=args.ckpt_dir,
        mode_cols=mode_cols,
        use_sigmoid=args.use_sigmoid,
        device=device,
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_mae = train_or_eval_epoch_rrp(
            model=model,
            loader=tr_dl,
            device=device,
            optimizer=optimizer,
        )
        va_mae = train_or_eval_epoch_rrp(
            model=model,
            loader=va_dl,
            device=device,
            optimizer=None,
        )

        print(f"[Epoch {ep:03d}] train MAE={tr_mae:.6f} | val MAE={va_mae:.6f}")

        if va_mae < best_val:
            best_val = va_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        # periodic checkpoint
        if (args.save_every > 0) and (ep % args.save_every == 0):
            ck = {
                "epoch": ep,
                "val_best": best_val,
                "model_state": model.state_dict(),
                "args": vars(args),
                "notes": "Multimode NVMD + price_head (L1 on RRP)",
            }
            out_ck = os.path.join(args.outdir, f"epoch_{ep:03d}.pt")
            torch.save(ck, out_ck)
            print(f"  [save] checkpoint -> {out_ck}")

    # Save best
    if best_state is not None:
        out_best = os.path.join(args.outdir, "best.pt")
        torch.save(
            {
                "epoch": "best",
                "val_best": best_val,
                "model_state": best_state,
                "args": vars(args),
                "notes": "Best multimode NVMD + price_head (L1 on RRP)",
            },
            out_best,
        )
        print(f"[done] Saved best checkpoint -> {out_best}")
    else:
        print("[warn] No best state captured; nothing saved.")


if __name__ == "__main__":
    main()
