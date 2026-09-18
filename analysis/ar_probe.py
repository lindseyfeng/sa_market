#!/usr/bin/env python3
"""
Leakage probe for decomposition-based forecasting.

Fits a plain linear AR(p) model to each mode independently on the training
CSV, then measures one-step-ahead error on the test CSV.

Why this detects leakage
------------------------
If a decomposition is computed over a window that includes the future (global
VMD, per-period VMD), each mode is a narrowband component fitted with
knowledge of what comes next.  Extrapolating it one step is then trivial --
least squares inverts it almost exactly -- and summing the modes reconstructs
the price without anything having been forecast.

Rule of thumb: if per-mode AR error is below ~1% of the RRP standard
deviation, the modes carry future information and any downstream accuracy
number is void.  A causal decomposition shows per-mode errors of the same
order as the signal's own step-to-step variation.

Usage:
    python ar_probe.py \
        --method "Per-year VMD" train_modes.csv test_modes.csv \
        --method "Causal VMD"   causal_train.csv causal_test.csv \
        --lags 48
"""

import argparse

import numpy as np
import pandas as pd

from analysis.benchmark import detect_mode_cols


def _design(a: np.ndarray, p: int):
    X = np.stack([a[i:i + p] for i in range(len(a) - p)])
    return np.c_[X, np.ones(len(X))], a[p:]


def probe(train_csv: str, test_csv: str, p: int, target_col: str = "RRP"):
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(test_csv)
    cols = detect_mode_cols(df_tr)

    pred_sum = None
    per_mode = []
    for c in cols:
        X_tr, y_tr = _design(df_tr[c].to_numpy(float), p)
        X_va, y_va = _design(df_va[c].to_numpy(float), p)
        w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        y_hat = X_va @ w
        per_mode.append(float(np.abs(y_hat - y_va).mean()))
        pred_sum = y_hat if pred_sum is None else pred_sum + y_hat

    rrp = df_va[target_col].to_numpy(float)[p:]
    std = float(rrp.std())
    return {
        "cols": cols,
        "per_mode": per_mode,
        "mae": float(np.abs(pred_sum - rrp).mean()),
        "rmse": float(np.sqrt(((pred_sum - rrp) ** 2).mean())),
        "rrp_std": std,
        # leakage score: median per-mode error as a fraction of the series std
        "leak_pct": float(np.median(per_mode[:-1]) / std * 100.0),
    }


def main():
    ap = argparse.ArgumentParser(description="AR leakage probe for mode CSVs")
    ap.add_argument("--method", nargs=3, action="append", required=True,
                    metavar=("NAME", "TRAIN_CSV", "TEST_CSV"))
    ap.add_argument("--lags", type=int, default=48)
    ap.add_argument("--target-col", default="RRP")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="flag as leaking below this %% of RRP std")
    args = ap.parse_args()

    print(f"\nAR({args.lags}) one-step probe -- modes fitted independently\n")
    print(f"  {'Method':<22s} {'sum MAE':>9s} {'sum RMSE':>9s} "
          f"{'median mode err':>16s} {'% of std':>9s}  verdict")
    print("  " + "-" * 82)

    results = {}
    for name, tr, va in args.method:
        r = probe(tr, va, args.lags, args.target_col)
        results[name] = r
        verdict = "LEAKING" if r["leak_pct"] < args.threshold else "ok"
        print(f"  {name:<22s} {r['mae']:>9.3f} {r['rmse']:>9.3f} "
              f"{np.median(r['per_mode'][:-1]):>16.3f} {r['leak_pct']:>8.2f}%  {verdict}")

    print()
    for name, r in results.items():
        print(f"  {name}  (RRP std = {r['rrp_std']:.2f})")
        print("    per-mode AR MAE: " + " ".join(f"{m:.2f}" for m in r["per_mode"]))
    print()


if __name__ == "__main__":
    main()
