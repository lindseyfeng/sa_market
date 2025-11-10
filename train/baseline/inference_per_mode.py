# infer_one_mode.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from cnn_bilstm import MRC_BiLSTM  # same class used at training time


class InferenceDataset(Dataset):
    """
    Build contiguous windows of length k over the FULL series:
      x_t = s[t-k : t] (normalized using provided scaler)
      y_t = s[t]       (normalized; used for metrics/inspection)
    Returns:
      x: (1, k), y: (1,), t_index: int (target's absolute index)
    """
    def __init__(self, series_raw, k: int, scaler):
        super().__init__()
        s = np.asarray(series_raw, dtype=np.float32)
        if len(s) < k + 1:
            raise ValueError(f"Series length {len(s)} must be >= k+1 ({k+1}).")
        s_min, s_max = float(scaler[0]), float(scaler[1])
        if s_max <= s_min:
            s_max = s_min + 1e-6

        self.s_min, self.s_max = s_min, s_max
        self.k = int(k)
        self.s_norm = (s - s_min) / (s_max - s_min)
        self.n = len(self.s_norm) - self.k  # valid targets are t in [k..T-1]

    def __len__(self): return self.n

    def __getitem__(self, idx):
        t = self.k + idx
        x = self.s_norm[t - self.k: t]            # (k,)
        y = self.s_norm[t]                         # scalar
        x = torch.from_numpy(x).unsqueeze(0)       # (1, k)
        y = torch.tensor([y], dtype=torch.float32) # (1,)
        return x, y, t


def denorm(v, s_min, s_max):
    return v * (s_max - s_min) + s_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-col", required=True, help="e.g., Mode_1 or EWT_Component_6 or Residual_minus_EWT_Sum")
    ap.add_argument("--ckpt", required=True, help="Path to training checkpoint .pt")
    ap.add_argument("--csv", default="../../VMD_modes_with_residual_2021_2022_with_EWT.csv",
                    help="CSV to run inference on (default: current folder)")
    ap.add_argument("--bsz", type=int, default=256, help="Batch size for inference")
    ap.add_argument("--out-dir", default="predictions", help="Folder to save predictions CSV")
    args = ap.parse_args()

    # --- Load checkpoint (must contain: model_state_dict, model_kwargs, scaler, k) ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    needed = {"model_state_dict", "model_kwargs", "scaler", "k"}
    missing = needed.difference(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {missing}")

    model_kwargs = ckpt["model_kwargs"]
    s_min, s_max = ckpt["scaler"]
    k = int(ckpt["k"])
    print(f"[INFO] Loaded ckpt: {args.ckpt}")
    print(f"[INFO] mode_in_ckpt={ckpt.get('mode_col', 'N/A')} | k={k} | scaler=(min={s_min:.6f}, max={s_max:.6f})")
    print(f"[INFO] model_kwargs={model_kwargs}")

    # --- Rebuild model, load weights ---
    model = MRC_BiLSTM(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[INFO] Model on {device} with {sum(p.numel() for p in model.parameters()):,} params")

    # --- Load CSV and column ---
    df = pd.read_csv(args.csv)
    if args.mode_col not in df.columns:
        raise ValueError(f"Column '{args.mode_col}' not in CSV. Available: {list(df.columns)}")
    series = df[args.mode_col].to_numpy(dtype=np.float32)

    # --- Dataset / Loader ---
    ds = InferenceDataset(series, k=k, scaler=(s_min, s_max))
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=False)
    print(f"[INFO] Inference dataset len={len(ds)} | k={k}")

    # --- Inference ---
    preds_norm, trues_norm, t_idx_list = [], [], []
    with torch.no_grad():
        for xb, yb, tb in dl:
            yhat = model(xb.to(device))            # (B, 1)
            preds_norm.append(yhat.cpu().squeeze(1))  # (B,)
            trues_norm.append(yb.squeeze(1))          # (B,)
            t_idx_list.append(tb)

    preds_norm = torch.cat(preds_norm, dim=0).numpy()
    trues_norm = torch.cat(trues_norm, dim=0).numpy()
    t_idx = torch.cat(t_idx_list, dim=0).numpy()

    # --- Denormalize ---
    preds = denorm(preds_norm, s_min, s_max)
    trues = denorm(trues_norm, s_min, s_max)

    # --- Assemble result dataframe (align to target indices) ---
    out = {
        "t_index": t_idx,
        "pred": preds,
        "actual": trues,
        "abs_error": np.abs(preds - trues),
    }
    if "SETTLEMENTDATE" in df.columns:
        out["SETTLEMENTDATE"] = df.loc[t_idx, "SETTLEMENTDATE"].reset_index(drop=True)
    out_df = pd.DataFrame(out)
    cols = ["t_index"] + (["SETTLEMENTDATE"] if "SETTLEMENTDATE" in out else []) + ["pred", "actual", "abs_error"]
    out_df = out_df[cols]

    # --- Metrics ---
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    print(f"[METRIC] MAE={mae:.6f} | RMSE={rmse:.6f}")

    # --- Save predictions ---
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = os.path.join(args.out_dir, f"{args.mode_col}_pred_{base}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved predictions to: {out_path}")

    # --- Quick tail preview ---
    print("\n[TAIL 5]")
    print(out_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
