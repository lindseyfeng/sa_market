# train_one_mode_fixed_k.py
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import optuna

# Your model (expects input x: (B, 1, k) and returns y: (B, 1))
from cnn_bilstm import MRC_BiLSTM
def denorm(v, s_min, s_max):
    return v * (s_max - s_min) + s_min

K_BY_MODE = {
    # VMD modes (Mode_1 .. Mode_12)
    "Mode_1": 9,   "Mode_2": 8,   "Mode_3": 9,   "Mode_4": 11,
    "Mode_5": 10,  "Mode_6": 10,  "Mode_7": 10,  "Mode_8": 9,
    "Mode_9": 7,   "Mode_10": 7,  "Mode_11": 7,  "Mode_12": 6,

    # EWT components (EWT_Component_1 .. EWT_Component_12)
    "EWT_Component_1": 7,   "EWT_Component_2": 10,  "EWT_Component_3": 10,  "EWT_Component_4": 7,
    "EWT_Component_5": 7,   "EWT_Component_6": 8,   "EWT_Component_7": 7,   "EWT_Component_8": 6,
    "EWT_Component_9": 8,   "EWT_Component_10": 11, "EWT_Component_11": 9,  "EWT_Component_12": 10,

    # EWT residual component
    "Residual_minus_EWT_Sum": 5,
}

def get_k(col_name: str) -> int:
    if col_name not in K_BY_MODE:
        raise KeyError(f"'{col_name}' not in K_BY_MODE. Valid keys: {list(K_BY_MODE.keys())}")
    return K_BY_MODE[col_name]


# ----------------------------
# Dataset: fixed contiguous k
# ----------------------------
class OneModeKDataset(Dataset):
    """
    For a single decomposed mode column.

    Given series s[0..T-1] and lookback k:
      sample at time t (t >= k) uses:
         x = s[t-k : t]          # previous k values (length k)
         y = s[t]                # next value (one step ahead)
    Returns:
      x: (1, k)  for Conv1d(C=1, L=k)
      y: (1,)    scalar target
    """
    def __init__(self, series_raw, k: int, scaler=None):
        super().__init__()
        self.s = np.asarray(series_raw, dtype=np.float32)
        k = int(k)
        if len(self.s) < k + 1:
            raise ValueError(f"Series length {len(s)} must be >= k+1 ({k+1}).")

        if scaler is None:
            s_min, s_max = float(s.min()), float(s.max())
            if s_max <= s_min:
                s_max = s_min + 1e-6
        else:
            s_min, s_max = float(scaler[0]), float(scaler[1])
        self.s_min, self.s_max = s_min, s_max

        self.series = (self.s - s_min) / (s_max - s_min)
        self.k = k
        self.n = len(self.series) - k  # valid t in [k .. T-1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        t = self.k + idx
        x = self.series[t - self.k : t]              # (k,)
        y = self.s[t]                            # scalar
        x = torch.from_numpy(x).unsqueeze(0)         # (1, k)
        y = torch.tensor([y], dtype=torch.float32)   # (1,)
        return x, y

    def scaler(self):
        return (self.s_min, self.s_max)


# ----------------------------
# Train one mode with fixed k
# ----------------------------
def train_one_mode_fixedL(series_raw,
                          k: int,
                          model_ctor,               # callable: model_ctor(k) -> nn.Module
                          epochs: int = 5,          # default 5 as requested
                          batch_size: int = 8,
                          lr: float = 1e-3,
                          weight_decay: float = 1e-6,
                          device: str | None = None,
                          print_every: int = 5):
    """
    Train a single-mode model with a fixed contiguous lookback `k` (from the paper).
    - Scaler is fit on the TRAIN split only.
    - The model is constructed with seq_len = k.
    Prints:
      - dataset/sample info
      - training loss every `print_every` steps.
    Returns: dict(model=..., scaler=(min,max), k=k)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s_train = np.asarray(series_raw, dtype=np.float32)
    if len(s_train) < k + 1:
        raise ValueError(f"Series length {len(s)} must be >= k+1 ({k+1}).")

    df_eval = pd.read_csv("../../VMD_modes_with_residual_2018_2021_with_EWT.csv")
    s_val = torch.tensor(df_eval[args.mode_col].to_numpy(), dtype=torch.float32) 

    s_min, s_max = float(s_train.min()), float(s_train.max())
    if s_max <= s_min:
        s_max = s_min + 1e-6
    scaler = (s_min, s_max)

    train_ds = OneModeKDataset(s_train, k=k, scaler=scaler)
    if len(s_val) < k + 1:
        s_val = s_train[-(k+1):]  # ensure we can form at least 1 validation sample
    val_ds   = OneModeKDataset(s_val, k=k, scaler=scaler)

    print(f"[INFO] train_ds_len={len(train_ds)}, val_ds_len={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model_ctor(k).to(device)
    print(f"[INFO] Model built with seq_len={k}; device={device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_state = float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        step = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)         # xb: (B,1,k), yb: (B,1)
            opt.zero_grad()
            yhat = denorm(model(xb), train_ds.s_min, train_ds.s_max)                               # (B,1)
            loss = F.mse_loss(yhat, yb)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            if step == 1 or (step % print_every == 0):
                print(f"[EP {ep}] step {step:4d} | train_loss={loss.item():.6f}")

        # Validate
        model.eval()
        va_loss, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                yhat = denorm(model(xb), train_ds.s_min, train_ds.s_max)
                loss = F.mse_loss(yhat, yb)
                va_loss += loss.item() * xb.size(0)
                m += xb.size(0)
        va_loss /= max(m, 1)
        print(f"[EP {ep}] -------- val_loss={va_loss:.6f}")
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best model with val_loss={best_loss:.6f}")

    return {"model": model, "scaler": scaler, "k": k}


# ----------------------------
# Main (only --mode-col changes)
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-col", default="Mode_1", help="Column name of the mode to train on")
    args = ap.parse_args()

    # CSV path in a parent folder (adjust if needed)
    csv_path = "../../VMD_modes_with_residual_2018_2021_with_EWT.csv"
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if args.mode_col not in df.columns:
        raise ValueError(f"{args.mode_col} not in CSV columns: {list(df.columns)}")
    if args.mode_col not in K_BY_MODE:
        raise ValueError(f"{args.mode_col} not in K_BY_MODE mapping. Keys: {list(K_BY_MODE.keys())}")

    series = df[args.mode_col].to_numpy(dtype=np.float32)
    k = get_k(args.mode_col)
    print(f"[INFO] Mode={args.mode_col} | paper_k={k}")

    # model ctor uses seq_len=k
    def model_ctor(seq_len_k: int):
        return MRC_BiLSTM(input_dim=1, seq_len=seq_len_k,
                          lstm_hidden=128, lstm_layers=3, bidirectional=True)


    print(model_ctor(k))
    out = train_one_mode_fixedL(
        series_raw=series,
        k=k,
        model_ctor=model_ctor,
        epochs=10,                   
        batch_size=500,                  
        lr=1e-3,
        weight_decay=1e-6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        print_every=100,
    )

    model, scaler = out["model"], out["scaler"]
    print(f"\n✅ Trained {args.mode_col} with fixed k = {k}")
    print(f"Scaler (min, max) = ({scaler[0]:.6f}, {scaler[1]:.6f})")

    # --- Evaluate last 5 samples: predicted vs. actual (denormalized) ---
    ds_full = OneModeKDataset(series, k=k, scaler=scaler)
    s_min, s_max = scaler
    scale_range = s_max - s_min

    model.eval()
    print(f"\n[CHECK] Last 5 predictions for {args.mode_col} (denormalized):")
    print(f"{'Idx':>8} | {'Pred':>12} | {'Actual':>12} | {'AbsError':>12}")
    print("-" * 50)

    with torch.no_grad():
        for i in range(-5, 0):
            idx = len(ds_full) + i
            if idx < 0:
                continue
            x_i, y_i = ds_full[idx]
            y_true_norm = y_i.item()
            pred_norm = model(x_i.unsqueeze(0)).item()

            # Denormalize
            y_true = y_true_norm * scale_range + s_min
            y_pred = pred_norm * scale_range + s_min
            abs_err = abs(y_pred - y_true)

            print(f"{idx:8d} | {y_pred:12.6f} | {y_true:12.6f} | {abs_err:12.6f}")

    print("-" * 50)


    # --- Inference on last one ---
    x_last, y_last = ds_full[len(ds_full) - 1]
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_last.unsqueeze(0)).item()

    y_true_norm = y_last.item()
    pred = pred_norm * scale_range + s_min
    y_true = y_true_norm * scale_range + s_min

    print("\n[CHECK] Final last-sample inference:")
    print(f"  pred (denorm) = {pred:.6f}, true_next (denorm) = {y_true:.6f}")

    scaler_path = f"scalar/scaler_{args.mode_col}.txt"
    with open(scaler_path, "w") as f:
        f.write(f"{s_min},{s_max}\n")
    print(f"\n✅ Saved scaler for {args.mode_col} to {scaler_path}")


    ckpt_dir   = f"checkpoints/{args.mode_col}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"mrc_bilstm_{args.mode_col}_k{k}.pt")

    checkpoint = {
        "mode_col": args.mode_col,
        "k": k,
        "scaler": (float(s_min), float(s_max)),
        "model_state_dict": model.state_dict(),
        # if you ever change ctor, adjust these to match your model’s args
        "model_kwargs": {
            "input_dim": 1,
            "seq_len": k,
            "lstm_hidden": 128,
            "lstm_layers": 3,
            "bidirectional": True,
        },
    }
    torch.save(checkpoint, ckpt_path)
    print(f"✅ Saved checkpoint to {ckpt_path}")