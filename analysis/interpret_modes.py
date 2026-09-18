#!/usr/bin/env python3
"""
Interpretability analysis for a mode decomposition.

Produces the evidence a reviewer will ask for when you claim a decomposition
is "interpretable".  Four things, none of which depend on the method used:

  1. Band table      -- spectral centroid, bandwidth, and physical period per
                        mode, plus whether centres are monotonically ordered.
  2. Band separation -- adjacent-centre gaps in units of bandwidth.  Values
                        below ~1 mean the bands overlap and the "modes" are
                        not distinct components.
  3. Energy share    -- fraction of total variance per mode, and the
                        participation ratio (how many modes are really used).
  4. Ablation        -- drop each mode and measure the increase in one-step
                        linear forecast error.  A mode that can be removed at
                        no cost is not carrying interpretable content.

    python interpret_modes.py \
        --method "Causal VMD" causal_train.csv causal_test.csv \
        --method "NVMD v3"    v3_train.csv     v3_test.csv \
        --hours-per-step 0.5
"""

import argparse

import numpy as np
import pandas as pd

from analysis.benchmark import detect_mode_cols


def band_table(df, cols, hours_per_step):
    M = df[cols].to_numpy(float)
    M = M - M.mean(0)
    P = np.abs(np.fft.rfft(M, axis=0)) ** 2
    f = np.fft.rfftfreq(len(M))
    p = P / (P.sum(0, keepdims=True) + 1e-12)
    centre = (p * f[:, None]).sum(0)
    bw = np.sqrt((p * (f[:, None] - centre) ** 2).sum(0))
    with np.errstate(divide="ignore"):
        period = np.where(centre > 1e-9, hours_per_step / np.maximum(centre, 1e-12), np.inf)
    return centre, bw, period


def ablation(df_tr, df_va, cols, p=48, target="RRP"):
    """One-step linear forecast error when each mode is dropped in turn."""
    def design(a):
        X = np.stack([a[i:i + p] for i in range(len(a) - p)])
        return X

    Xtr = np.concatenate([design(df_tr[c].to_numpy(float)) for c in cols], axis=1)
    Xva = np.concatenate([design(df_va[c].to_numpy(float)) for c in cols], axis=1)
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]
    Xva = np.c_[Xva, np.ones(len(Xva))]
    ytr = df_tr[target].to_numpy(float)[p:]
    yva = df_va[target].to_numpy(float)[p:]

    def fit_mae(keep_cols, lam=10.0):
        # Ridge, not OLS: with K*p features an unregularised fit overfits, and
        # dropping columns then *improves* val error, which inverts the sign of
        # every ablation and makes the whole table meaningless.
        idx = []
        for j, _ in enumerate(cols):
            if j in keep_cols:
                idx.extend(range(j * p, (j + 1) * p))
        idx.append(Xtr.shape[1] - 1)
        A = Xtr[:, idx]
        G = A.T @ A
        reg = lam * np.eye(len(idx))
        reg[-1, -1] = 0.0                      # never penalise the intercept
        w = np.linalg.solve(G + reg, A.T @ ytr)
        return float(np.abs(Xva[:, idx] @ w - yva).mean())

    base = fit_mae(set(range(len(cols))))
    deltas = []
    for j in range(len(cols)):
        keep = set(range(len(cols))) - {j}
        deltas.append(fit_mae(keep) - base)
    return base, np.array(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs=3, action="append", required=True,
                    metavar=("NAME", "TRAIN_CSV", "TEST_CSV"))
    ap.add_argument("--hours-per-step", type=float, default=0.5)
    ap.add_argument("--lags", type=int, default=48)
    ap.add_argument("--target-col", default="RRP")
    args = ap.parse_args()

    for name, tr_csv, va_csv in args.method:
        df_tr, df_va = pd.read_csv(tr_csv), pd.read_csv(va_csv)
        cols = detect_mode_cols(df_va)
        centre, bw, period = band_table(df_va, cols, args.hours_per_step)
        M = df_va[cols].to_numpy(float)
        share = M.var(0) / (M.var(0).sum() + 1e-12)
        base, delta = ablation(df_tr, df_va, cols, args.lags, args.target_col)

        order = np.argsort(centre)
        monotonic = bool(np.all(np.diff(centre[:-1]) > 0))
        gaps = np.diff(np.sort(centre))
        sep = gaps / (0.5 * (np.sort(bw)[:-1] + np.sort(bw)[1:]) + 1e-12)

        print("\n" + "=" * 78)
        print(f"  {name}   ({len(cols)} modes)   base linear MAE = {base:.3f}")
        print("=" * 78)
        print(f"  {'mode':<12s}{'centre':>9s}{'bw':>9s}{'period(h)':>12s}"
              f"{'energy%':>10s}{'ablation':>11s}")
        print("  " + "-" * 74)
        for j, c in enumerate(cols):
            per = f"{period[j]:.1f}" if np.isfinite(period[j]) else "inf"
            print(f"  {c:<12s}{centre[j]:>9.4f}{bw[j]:>9.4f}{per:>12s}"
                  f"{100*share[j]:>10.2f}{delta[j]:>+11.3f}")
        print("  " + "-" * 74)
        print(f"  centres monotonically ordered : {monotonic}")
        print(f"  min adjacent separation       : {sep.min():.2f} bandwidths"
              f"   (<1.0 = overlapping bands)")
        print(f"  participation ratio           : "
              f"{1.0/np.sum(share**2):.2f} of {len(cols)} modes")
        dead = int((delta < 0.01).sum())
        print(f"  modes removable at <0.01 MAE  : {dead}/{len(cols)}")
    print()


if __name__ == "__main__":
    main()
