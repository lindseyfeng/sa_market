import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from train_bo_dropout import MRC_BiLSTM  


# ✅ fixed dataset path (edit this once if needed)
CSV_PATH = "../../VMD_modes_with_residual_2021_2022_with_EWT.csv"


class InferenceDataset(Dataset):
    def __init__(self, series_raw, k: int, scaler):
        s = np.asarray(series_raw, dtype=np.float32)
        if len(s) < k + 1:
            raise ValueError(f"Series length {len(s)} < k+1 ({k+1}).")

        s_min, s_max = float(scaler[0]), float(scaler[1])
        if s_max <= s_min:
            s_max = s_min + 1e-6
        
        self.s_min, self.s_max = s_min, s_max
        self.k = k
        self.s_norm = (s - s_min) / (s_max - s_min)
        self.n = len(self.s_norm) - k

    def __len__(self): return self.n

    def __getitem__(self, idx):
        t = self.k + idx
        x = self.s_norm[t-self.k:t]
        y = self.s_norm[t]
        x = torch.from_numpy(x).unsqueeze(0) 
        y = torch.tensor([y], dtype=torch.float32)
        return x, y, t


def denorm(v, s_min, s_max):
    return v * (s_max - s_min) + s_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-col", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bsz", type=int, default=512)
    ap.add_argument("--out-dir", default="predictions")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    needed = {"model_state_dict", "model_kwargs", "scaler", "k"}
    missing = needed - ckpt.keys()
    if missing:
        raise ValueError(f"Checkpoint missing keys {missing}")

    model_kwargs = ckpt["model_kwargs"]
    s_min, s_max = ckpt["scaler"]
    k = int(ckpt["k"])

    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    print(f"[INFO] model_kwargs = {model_kwargs}")
    print(f"[INFO] k={k}, scaler=({s_min:.6f}, {s_max:.6f})")
    if "seq_len" in model_kwargs:
        model_kwargs["k"] = model_kwargs.pop("seq_len")
    model = MRC_BiLSTM(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    df = pd.read_csv(CSV_PATH)
    series = df[args.mode_col].to_numpy(np.float32)

    ds = InferenceDataset(series, k=k, scaler=(s_min, s_max))
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=False)

    preds_norm, trues_norm, idxs = [], [], []

    with torch.no_grad():
        for x, y, t in dl:
            yhat = model(x.to(device))
            preds_norm.append(yhat.squeeze(1).cpu())
            trues_norm.append(y.squeeze(1))
            idxs.append(t)

    preds_norm = torch.cat(preds_norm).numpy()
    trues_norm = torch.cat(trues_norm).numpy()
    idxs = torch.cat(idxs).numpy()

    preds = denorm(preds_norm, s_min, s_max)
    trues = denorm(trues_norm, s_min, s_max)

    out = {
        "t_index": idxs,
        "pred": preds,
        "actual": trues,
        "abs_error": np.abs(preds - trues),
    }
    if "SETTLEMENTDATE" in df.columns:
        out["SETTLEMENTDATE"] = df.loc[idxs, "SETTLEMENTDATE"].reset_index(drop=True)

    out_df = pd.DataFrame(out)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    print(f"[METRICS] MAE={mae:.6f} | RMSE={rmse:.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(CSV_PATH))[0]
    save_path = os.path.join(args.out_dir, f"{args.mode_col}_pred_{base}.csv")
    out_df.to_csv(save_path, index=False)

    print(f"[INFO] Saved predictions → {save_path}")
    print(out_df.tail(5))


if __name__ == "__main__":
    main()
