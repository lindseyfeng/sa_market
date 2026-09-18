#!/usr/bin/env python3
"""
Concatenate NVMD mode CSVs decomposed at different window lengths.

Motivation: with an L-sample window the frequency resolution is 1/L, so at
L=96 the low bands (v3's DC bandwidth is 0.0025 vs a resolution of 0.0104)
are under-resolved.  Long windows resolve low frequencies; short windows
track fast dynamics with less lag.  Decomposing at several scales and
concatenating gives the predictor both.

This is cheap for NVMD (one forward pass per scale) and impractical for VMD,
which must re-solve an optimisation per window per scale.

Output columns are Mode_1 .. Mode_{sum of K}, then RRP, so benchmark.py's
detect_mode_cols() picks them all up.

NOTE: no Residual column is written.  Modes from different scales each sum to
the signal, so a combined sum would multiply-count.  This means ar_probe.py's
*summed* metric is meaningless here -- only its per-mode column applies.

    python merge_scales.py --out ms_2019.csv \
        v3_L96_2019.csv v3_L192_2019.csv v3_L384_2019.csv
"""

import argparse

import pandas as pd

from analysis.benchmark import detect_mode_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="mode CSVs, coarsest scale last")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="SETTLEMENTDATE")
    ap.add_argument("--target-col", default="RRP")
    args = ap.parse_args()

    frames = []
    next_idx = 1
    rrp_ref = None

    for path in args.csvs:
        df = pd.read_csv(path)
        if args.key not in df.columns:
            raise ValueError(f"{path} has no '{args.key}' column to align on")
        df[args.key] = pd.to_datetime(df[args.key])

        cols = [c for c in detect_mode_cols(df) if c != "Residual"]
        ren = {c: f"Mode_{next_idx + i}" for i, c in enumerate(cols)}
        next_idx += len(cols)

        keep = df[[args.key] + cols].rename(columns=ren)
        frames.append(keep)
        if rrp_ref is None:
            rrp_ref = df[[args.key, args.target_col]]
        print(f"  {path}: {len(cols)} modes -> "
              f"{ren[cols[0]]}..{ren[cols[-1]]}  ({len(df)} rows)")

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=args.key, how="inner")
    merged = merged.merge(rrp_ref, on=args.key, how="inner")

    # Longer windows start later, so the inner join trims the leading rows
    # that the coarsest scale cannot cover.
    merged = merged.sort_values(args.key).reset_index(drop=True)
    mode_cols = [c for c in merged.columns if c.startswith("Mode_")]
    merged = merged[mode_cols + [args.target_col, args.key]]
    merged.to_csv(args.out, index=False)
    print(f"merged: {len(merged)} rows x {len(mode_cols)} modes -> {args.out}")


if __name__ == "__main__":
    main()
