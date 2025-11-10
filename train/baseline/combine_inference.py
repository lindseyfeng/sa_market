# combine_inference.py
import os, glob
import numpy as np
import pandas as pd
from functools import reduce

def main(pred_dir="predictions", out_csv="predictions/final_reconstructed_prediction.csv"):
    files = sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {pred_dir}")

    # Peek first file to decide the key
    probe = pd.read_csv(files[0])
    key = "SETTLEMENTDATE" if "SETTLEMENTDATE" in probe.columns else "t_index"
    print(f"[INFO] Using merge key: {key}")

    dfs = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]  # e.g., Mode_1_pred_...
        df = pd.read_csv(f)

        # Ensure the key exists
        if key not in df.columns:
            raise ValueError(f"{f} does not have merge key '{key}'. Columns: {list(df.columns)}")

        # Keep ONLY the chosen key + pred/actual; drop everything else to avoid suffix clashes
        need = [key, "pred", "actual"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing required columns {missing}")

        # Rename pred/actual to be unique for this component
        df = df[need].rename(columns={
            "pred":   f"pred__{name}",
            "actual": f"actual__{name}",
        })

        # If key is datetime-like string, ensure same dtype across files
        if key == "SETTLEMENTDATE":
            df[key] = pd.to_datetime(df[key])

        dfs.append(df)
        print(f"[LOAD] {name:40s} -> rows={len(df)}")

    # Merge all on the single key (inner: keep intersection only)
    merged = reduce(lambda l, r: pd.merge(l, r, on=key, how="inner"), dfs)
    print(f"[INFO] Merged shape: {merged.shape}")

    # Collect columns
    pred_cols   = [c for c in merged.columns if c.startswith("pred__")]
    actual_cols = [c for c in merged.columns if c.startswith("actual__")]

    if not pred_cols or not actual_cols:
        raise RuntimeError("No pred/actual columns found after merge.")

    # Sum across all components to reconstruct the final price
    merged["pred_sum"]   = merged[pred_cols].sum(axis=1)
    merged["actual_sum"] = merged[actual_cols].sum(axis=1)
    merged["abs_error_sum"] = (merged["pred_sum"] - merged["actual_sum"]).abs()

    # Metrics
    mae  = float(merged["abs_error_sum"].mean())
    rmse = float(np.sqrt(((merged["pred_sum"] - merged["actual_sum"])**2).mean()))
    print(f"\n[RESULT] Reconstructed totals: MAE={mae:.6f} | RMSE={rmse:.6f}")

    # Save compact result
    out_cols = [key, "pred_sum", "actual_sum", "abs_error_sum"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged[out_cols].to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}")

    # Show tail
    print("\n[TAIL 5]")
    print(merged[out_cols].tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
