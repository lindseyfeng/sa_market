#!/usr/bin/env python3
"""
How much does each decomposition's basis move between adjacent windows?

This is the proposed x-axis: if cross-window basis stability, rather than
decomposition quality, is what limits a sequence model, then the arms in
stability_results.json should line up against this scalar.

Two numbers per method, because one of them has a floor:

  drift   mean |delta centroid| between adjacent windows, as a fraction of the
          mean inter-mode gap.  For an adaptive method this mixes genuine basis
          movement with within-band signal variation; for a fixed basis only
          the second term survives, so the fixed methods measure the floor and
          the interesting quantity is the excess above it.

  churn   fraction of adjacent-window steps in which some mode's centroid moves
          more than half a band gap -- i.e. channel k stops meaning what it
          meant one step earlier.

    python basis_stability.py --n 1500
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ZOO = {"ewt": "adaptive", "emd": "adaptive", "wpt": "fixed", "bank": "fixed"}


def stats(cen):
    """cen: (n_windows, K) sorted centroids -> drift, churn, per-mode std."""
    cen = cen[~np.isnan(cen).any(1)]
    gap = np.diff(cen, axis=1).mean()
    step = np.abs(np.diff(cen, axis=0))
    drift = step.mean() / gap
    churn = (step > 0.5 * gap).any(axis=1).mean()
    return drift, churn, cen.std(0), gap


def vmd_centroids(n, W=96, K=8, alpha=1000):
    """VMD was cached as last-step modes only, so re-run a sample to get the
    full windows its centroids need.  Same settings as the cached run."""
    from joblib import Parallel, delayed
    from vmdpy import VMD
    from decomp import decomp_zoo as dz

    df = pd.read_csv("data/raw/compound_2018_2022.csv", parse_dates=["SETTLEMENTDATE"])
    sig = df[df.SETTLEMENTDATE.dt.year == 2018]["SA1_price"].to_numpy(float)

    def one(t):
        c = np.pad(sig[t - W:t], (0, 20), mode="edge")
        u, _, om = VMD(c, alpha, 0.0, K, 0, 1, 1e-7)
        cen = dz.spectral_centroid(u[:, :W])
        return np.sort(cen), np.sort(om[-1])

    out = Parallel(n_jobs=6)(delayed(one)(t) for t in range(W, W + n))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--out", default="results/basis_stability.json")
    args = ap.parse_args()

    rows = []
    cen, om = vmd_centroids(args.n, args.window, args.K)
    d, c, sd, gap = stats(cen)
    om_step = np.abs(np.diff(om, axis=0)).mean() / np.diff(om.mean(0)).mean()
    rows.append(("vmd", "adaptive", d, c, om_step))

    for m, kind in ZOO.items():
        p = f"cache/decomp_{m}_K{args.K}_W{args.window}/2018_SA1_price_centroids.npy"
        if not os.path.exists(p):
            print(f"{m}: centroids not generated yet, skipping")
            continue
        cen = np.load(p)[args.window - 1:args.window - 1 + args.n]
        d, c, sd, gap = stats(cen)
        rows.append((m, kind, d, c, 0.0 if kind == "fixed" else np.nan))

    rows.sort(key=lambda r: r[2])
    print(f"\nbasis movement between adjacent windows, {args.n} windows, 2018 price\n")
    print(f"{'method':>8}{'kind':>10}{'drift':>10}{'churn':>9}{'explicit':>10}")
    print("-" * 47)
    for m, kind, d, c, e in rows:
        es = "0 (fixed)" if kind == "fixed" else (f"{e:.1%}" if e == e else "--")
        print(f"{m:>8}{kind:>10}{d:>9.1%}{c:>9.1%}{es:>10}")
    print("\ndrift    |delta centroid| per step / mean inter-mode gap")
    print("churn    steps where some mode's centroid jumps > half a band gap")
    print("explicit movement of the method's own band parameters, where it has them")
    print("\nThe fixed rows are the floor: their bands never move, so whatever")
    print("drift they show is the signal changing inside a fixed band. The")
    print("excess of an adaptive method over that floor is basis movement.")

    import json
    json.dump([{"method": m, "kind": k, "drift": float(d), "churn": float(c),
                "explicit": (None if e != e else float(e))}
               for m, k, d, c, e in rows], open(args.out, "w"), indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
