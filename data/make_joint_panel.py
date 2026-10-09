#!/usr/bin/env python3
"""Add the four neighbouring regions' price *levels* to the compound panel.

Why derive them instead of reading `data/raw/panel_2018_2022.csv`
----------------------------------------------------------------
That file is the **filtered** panel: 77,397 rows against the compound panel's
87,600, minimum 1.00, maximum 981.65, and not one negative price. The rows it is
missing are 278 / 849 / 1,840 / 3,622 / 3,614 by year -- the same filter ROUND2
exists to get away from. Joining targets from it would put the filtering bias
back into the one experiment meant to be free of it.

`spread_SA1_X = SA1_price - X_price` holds exactly in the compound panel (checked
to 1.1e-13 against the filtered panel on all 77,397 overlapping rows), so every
neighbour's unfiltered level is already recoverable from channels the arms
already read. Deriving adds no information to the inputs: it only names what the
model can already see, which is what makes the joint objective a clean test.

The 12 rows where a derived level lands outside the NEM price band are clipped.
A 30-minute RRP is the mean of six 5-minute dispatch prices, each inside
[-1000, 15500], so a value outside that band is not a price -- it is the
`merge_asof(nearest, +/-60min)` tolerance biting on a fast-moving interval.
"""
import argparse
import os

import numpy as np
import pandas as pd

NEIGHBOURS = ["NSW1", "VIC1", "QLD1", "TAS1"]
FLOOR, CAP = -1000.0, 15500.0


def build(src, out, check=None):
    df = pd.read_csv(src, parse_dates=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    before = df.shape
    for r in NEIGHBOURS:
        col = f"spread_SA1_{r}"
        if col not in df.columns:
            raise SystemExit(f"{src} has no {col}; cannot derive {r}")
        df[f"{r}_price"] = df["SA1_price"] - df[col]

    lvl = ["SA1_price"] + [f"{r}_price" for r in NEIGHBOURS]
    out_of_band = ((df[lvl] < FLOOR - 1e-6) | (df[lvl] > CAP + 1e-6))
    n_rows = int(out_of_band.any(axis=1).sum())
    n_cells = int(out_of_band.to_numpy().sum())
    df[lvl] = df[lvl].clip(FLOOR, CAP)
    print(f"clipped {n_cells} value(s) on {n_rows} row(s) to the NEM band "
          f"[{FLOOR:g}, {CAP:g}]  ({n_rows / len(df) * 100:.3f}% of rows)")

    if check and os.path.exists(check):
        flt = pd.read_csv(check, parse_dates=["SETTLEMENTDATE"])
        m = df.merge(flt, on="SETTLEMENTDATE", how="inner")
        worst = 0.0
        for r in ["SA1"] + NEIGHBOURS:
            src_col = "SA1_price" if r == "SA1" else f"{r}_price"
            worst = max(worst, float((m[src_col] - m[r]).abs().max()))
        print(f"cross-checked against {os.path.basename(check)} on {len(m)} "
              f"overlapping rows: worst disagreement {worst:.2e}")
        if worst > 1e-6:
            raise SystemExit("derived levels disagree with the filtered panel")

    assert df[lvl].notna().all().all(), "derived levels contain NaN"
    df.to_csv(out, index=False)
    print(f"{before[1]} -> {df.shape[1]} columns, {len(df)} rows -> {out}")
    print("\nunfiltered levels, whole period:")
    print(df[lvl].describe().T[["mean", "std", "min", "max"]].round(2))
    print("\nnegative-price rows per region:")
    for c in lvl:
        print(f"  {c:<12} {int((df[c] < 0).sum()):>6}  "
              f"({(df[c] < 0).mean() * 100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/compound_unfiltered_2018_2022.csv")
    ap.add_argument("--out", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--check", default="data/raw/panel_2018_2022.csv",
                    help="filtered panel, used only to validate the derivation")
    a = ap.parse_args()
    build(a.src, a.out, a.check)
