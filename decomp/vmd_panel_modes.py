#!/usr/bin/env python3
"""
Causal VMD modes for every channel of the compound panel.

This is arm 2 of the three-arm comparison in attic/RESULTS-superseded.md section 12: give
classical VMD the same exogenous panel the spatio-temporal NVMD gets, one
univariate decomposition per channel.  Arm 1 (price only) is the SA1_price
slice of the same output, so both VMD arms come from one pass.

Causal by construction: window [t-W, t) only, modes read at the last step,
exactly as generate_modes.py does for the price-only CSVs.  K=8 and
alpha=1000 are the tuned configuration from section 10 -- the *best by test*
VMD config, i.e. the choice most favourable to the baseline.

One .npy per (year, channel) so an interrupted run resumes.

    python vmd_panel_modes.py --years 2018,2019
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from vmdpy import VMD

CAL_PREFIX = "cal_"


def _one_window(t, sig, W, K, alpha, pad):
    chunk = np.pad(sig[t - W:t], (0, pad), mode="edge")
    try:
        u, _, omega = VMD(chunk, alpha, 0.0, K, 0, 1, 1e-7)
        return u[omega[-1].argsort()][:, W - 1]
    except Exception:
        return np.full(K, np.nan)


def channel_modes(sig, W, K, alpha, n_jobs):
    """(T,) -> (T, K); rows before W-1 are NaN, as the window is not yet full."""
    T = len(sig)
    pad = min(20, W // 4)
    out = np.full((T, K), np.nan, dtype=np.float32)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_one_window)(t, sig, W, K, alpha, pad) for t in range(W, T + 1)
    )
    for t, m in zip(range(W, T + 1), res):
        out[t - 1] = m
    return out


def cached_ok(path, W, K):
    """A cache hit counts only if the file loads and is shaped and filled right.

    np.save is not atomic, so a kill during the write can leave a truncated
    file that a plain os.path.exists() would treat as done.  Checking the
    array costs milliseconds and turns that silent corruption into a re-run.
    """
    if not os.path.exists(path):
        return False
    try:
        a = np.load(path)
    except Exception:
        return False
    return (a.ndim == 2 and a.shape[1] == K and a.shape[0] > W
            and not np.isnan(a[W - 1:]).any())


def save_atomic(path, arr):
    tmp = path + ".tmp"
    np.save(tmp, arr)
    os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_2018_2022.csv")
    ap.add_argument("--years", default="2018,2019")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1000)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--outdir", default="cache/vmd_panel_K8_a1000_W96")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.panel, parse_dates=["SETTLEMENTDATE"])
    chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
    decomp = [c for c in chans if not c.startswith(CAL_PREFIX)]
    years = [int(v) for v in args.years.split(",")]

    print(f"panel {args.panel}: {len(chans)} channels, "
          f"{len(decomp)} to decompose, {len(chans)-len(decomp)} calendar")
    print(f"K={args.K} alpha={args.alpha:g} W={args.window} -> {args.outdir}\n",
          flush=True)

    todo = [(y, c) for y in years for c in decomp]
    t_start = time.time()
    for i, (year, ch) in enumerate(todo, 1):
        path = os.path.join(args.outdir, f"{year}_{ch}.npy")
        if cached_ok(path, args.window, args.K):
            print(f"[{i}/{len(todo)}] {year} {ch}: cached", flush=True)
            continue
        sig = df[df.SETTLEMENTDATE.dt.year == year][ch].to_numpy(np.float32)
        t0 = time.time()
        m = channel_modes(sig, args.window, args.K, args.alpha, args.n_jobs)
        save_atomic(path, m)
        bad = int(np.isnan(m[args.window - 1:]).any(axis=1).sum())
        done, left = i, len(todo) - i
        eta = (time.time() - t_start) / done * left
        print(f"[{i}/{len(todo)}] {year} {ch}: {len(sig)} rows, "
              f"{time.time()-t0:.0f}s, {bad} failed windows | ETA {eta/60:.0f} min",
              flush=True)

    print(f"\ndone in {(time.time()-t_start)/60:.1f} min -> {args.outdir}")


if __name__ == "__main__":
    main()
