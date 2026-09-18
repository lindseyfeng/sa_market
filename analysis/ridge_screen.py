#!/usr/bin/env python3
"""
Fast closed-form ridge screen over decomposition CSVs.

Purpose: the full multi-seed benchmark costs ~5 h per method pair, which makes an
18-point VMD hyperparameter grid infeasible (~80 h).  This reproduces the *Linear*
column -- the cell where NVMD's margin is largest, i.e. the most VMD-favourable
test -- in seconds, so the grid can be screened and only the winners promoted to
the full benchmark.

Feature construction mirrors benchmark.ModeWindowDataset exactly:
    x = modes[i : i+W]  (z-scored with TRAIN stats),  y = RRP[i+W+h-1]

Ridge lambda is chosen on a held-out tail of the TRAIN year, never on the test
year, so the screen does not hand the baseline test-set tuning.
"""

import argparse
import numpy as np
import pandas as pd


def detect_mode_cols(df):
    cols = []
    for i in range(1, 100):
        c = f"Mode_{i}"
        if c in df.columns:
            cols.append(c)
        else:
            break
    if "Residual" in df.columns:
        cols.append("Residual")
    return cols


def windows(df, mode_cols, W, h, mm, ms, tm, ts):
    modes = df[mode_cols].to_numpy(dtype=np.float64)
    target = df["RRP"].to_numpy(dtype=np.float64)
    modes = (modes - mm) / ms
    N = max(0, len(target) - W - (h - 1))
    # strided view -> (N, W, K) -> flatten to (N, W*K)
    idx = np.arange(N)[:, None] + np.arange(W)[None, :]
    X = modes[idx].reshape(N, -1)
    y = target[np.arange(N) + W + h - 1]
    return X, y


def fit_ridge(X, y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb
    A[np.diag_indices_from(A)] += lam
    A[-1, -1] -= lam                      # do not penalise the intercept
    return np.linalg.solve(A, Xb.T @ y)


def predict(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def run_one(train_csv, test_csv, W, h, lams, val_frac):
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    cols = detect_mode_cols(tr)
    m = tr[cols].to_numpy(dtype=np.float64)
    mm, ms = m.mean(axis=0), m.std(axis=0) + 1e-8
    tm, ts = float(tr["RRP"].mean()), float(tr["RRP"].std()) + 1e-8

    Xtr, ytr = windows(tr, cols, W, h, mm, ms, tm, ts)
    Xte, yte = windows(te, cols, W, h, mm, ms, tm, ts)

    # lambda selected on a tail slice of the TRAIN year only
    cut = int(len(Xtr) * (1 - val_frac))
    best_lam, best_val = None, np.inf
    for lam in lams:
        w = fit_ridge(Xtr[:cut], ytr[:cut], lam)
        v = np.abs(predict(Xtr[cut:], w) - ytr[cut:]).mean()
        if v < best_val:
            best_val, best_lam = v, lam

    w = fit_ridge(Xtr, ytr, best_lam)
    p = predict(Xte, w)
    mae = float(np.abs(p - yte).mean())
    rmse = float(np.sqrt(((p - yte) ** 2).mean()))
    return dict(K=len(cols), lam=best_lam, val_mae=float(best_val),
                mae=mae, rmse=rmse, n_tr=len(Xtr), n_te=len(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs=3, action="append", required=True,
                    metavar=("NAME", "TRAIN_CSV", "TEST_CSV"))
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-2, 1e-1, 1, 10, 100, 1000, 1e4])
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    rows = []
    for name, tr, te in args.method:
        try:
            r = run_one(tr, te, args.window, args.horizon, args.lams, args.val_frac)
        except Exception as e:
            print(f"  {name:<24} FAILED: {e}")
            continue
        r["name"] = name
        rows.append(r)
        print(f"  {name:<24} K={r['K']:<3} lam={r['lam']:<8g} "
              f"test MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  (trainval {r['val_mae']:.3f})")

    if not rows:
        return
    rows.sort(key=lambda r: r["mae"])
    print("\n" + "=" * 74)
    print(f"  Ridge screen, W={args.window} h={args.horizon}   (ranked by test MAE)")
    print("=" * 74)
    ref = next((r for r in rows if r["name"] == args.ref), None)
    for r in rows:
        d = ""
        if ref is not None and r is not ref:
            gain = r["mae"] - ref["mae"]
            d = f"   {gain:+.3f}  ({gain / ref['mae'] * 100:+.1f}% vs {args.ref})"
        print(f"  {r['name']:<24} {r['mae']:8.3f}{d}")
    print("\n  NOTE: single deterministic fit -- no seed noise, but this is a SCREEN.")
    print("  Promote winners to benchmark_seeds.py before reporting anything.")


if __name__ == "__main__":
    main()
