#!/usr/bin/env python3
"""
Build an aligned multi-region NEM price panel.

Target region (SA1) rows are taken from the same filtered set used everywhere
else in this project (RRP in [1, 981.65]), so the forecasting target is
byte-identical to the temporal-only experiments and the comparison is exact.

Other regions are aligned onto those timestamps and their prices are *clipped*
to the same range rather than filtered -- dropping their outlier rows would
misalign the panel, and leaving raw values (NEM prices run to +/-$15,000) would
wreck normalisation for every region at once.  Clipping bounds the influence of
spikes in the covariates while keeping one row per target timestamp.

    python build_panel.py --out panel_2018_2022.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

REGIONS = ["SA1", "NSW1", "VIC1", "QLD1", "TAS1"]
LO, HI = 1.0, 981.65


def load_region(region: str, src_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(src_dir, f"*_{region}.csv")))
    if not files:
        raise FileNotFoundError(f"no files for {region} in {src_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df[["SETTLEMENTDATE", "RRP"]].sort_values("SETTLEMENTDATE")
    return df.drop_duplicates("SETTLEMENTDATE").rename(columns={"RRP": region})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-csv", default="SA_filtered_2018_2022.csv",
                    help="already-filtered target region series")
    ap.add_argument("--src-dir", default="nem_regions")
    ap.add_argument("--out", default="panel_2018_2022.csv")
    args = ap.parse_args()

    tgt = pd.read_csv(args.target_csv, parse_dates=["SETTLEMENTDATE"])
    tgt = tgt.rename(columns={"RRP": "SA1"})[["SETTLEMENTDATE", "SA1"]]
    print(f"target SA1 (filtered): {len(tgt)} rows")

    panel = tgt
    for r in REGIONS[1:]:
        df = load_region(r, args.src_dir)
        before = len(panel)
        panel = panel.merge(df, on="SETTLEMENTDATE", how="left")
        miss = int(panel[r].isna().sum())
        print(f"  {r}: merged, {miss} missing of {before}")

    # Forward/backward fill the handful of timestamps a region is missing,
    # then clip covariates into the target's range.
    for r in REGIONS[1:]:
        panel[r] = panel[r].ffill().bfill().clip(LO, HI)
    panel = panel.dropna().reset_index(drop=True)

    panel.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(panel)} rows x {len(REGIONS)} regions")
    print(panel[REGIONS].describe().loc[["mean", "std", "min", "max"]].round(2))

    c = panel[REGIONS].corr()
    print("\ncross-region price correlation (raw):")
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
