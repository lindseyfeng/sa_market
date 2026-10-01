"""Rebuild the compound panel on the full half-hourly grid.

`compound_2018_2022.csv` is indexed on the *filtered* target, which drops every
negative price and everything above 981.65 -- 1.6% of 2018 rising to 20.7% of
2022. Those rows carry most of the forecast error, and removing a different
fraction of each year filters train and test to different distributions.

The price comes back from the raw AEMO series. The exogenous channels are
reindexed onto the full grid and interpolated in time: the gaps have a median
length of one hour and 88% are under six, over which demand and weather are
smooth. The calendar channels are recomputed exactly rather than interpolated.
"""
import argparse
import numpy as np
import pandas as pd

CAL = ["cal_day_sin", "cal_day_cos", "cal_week_sin", "cal_week_cos",
       "cal_year_sin", "cal_year_cos", "cal_weekend"]


def calendar(ts):
    tod = ts.dt.hour * 2 + ts.dt.minute // 30
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear
    return pd.DataFrame({
        "cal_day_sin": np.sin(2 * np.pi * tod / 48),
        "cal_day_cos": np.cos(2 * np.pi * tod / 48),
        "cal_week_sin": np.sin(2 * np.pi * dow / 7),
        "cal_week_cos": np.cos(2 * np.pi * dow / 7),
        "cal_year_sin": np.sin(2 * np.pi * doy / 365.25),
        "cal_year_cos": np.cos(2 * np.pi * doy / 365.25),
        "cal_weekend": (dow >= 5).astype(float),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", default="data/raw/SA_unfiltered_2018_2022.csv")
    ap.add_argument("--compound", default="data/raw/compound_2018_2022.csv")
    ap.add_argument("--out", default="data/raw/compound_unfiltered_2018_2022.csv")
    args = ap.parse_args()

    price = pd.read_csv(args.price, parse_dates=["SETTLEMENTDATE"])
    price = price.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE")
    comp = pd.read_csv(args.compound, parse_dates=["SETTLEMENTDATE"])

    exo = [c for c in comp.columns if c not in ["SETTLEMENTDATE", "SA1_price"] + CAL]
    out = price[["SETTLEMENTDATE", "SA1_price"]].reset_index(drop=True)
    merged = out.merge(comp[["SETTLEMENTDATE"] + exo], on="SETTLEMENTDATE", how="left")

    before = merged[exo].isna().sum().sum()
    merged[exo] = merged[exo].interpolate(limit_direction="both")
    print(f"exogenous channels: {len(exo)}, filled {before} missing cells by time interpolation")

    merged = pd.concat([merged, calendar(merged.SETTLEMENTDATE)], axis=1)
    merged = merged[["SETTLEMENTDATE", "SA1_price"] + exo + CAL]

    assert merged.isna().sum().sum() == 0, "panel still has NaN"
    step = merged.SETTLEMENTDATE.diff().dropna().unique()
    assert len(step) == 1, f"grid is not uniform: {step}"

    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(merged)} rows x {len(merged.columns) - 1} channels")
    p = merged.SA1_price
    print(f"price: min {p.min():.1f}  median {p.median():.1f}  max {p.max():.1f}  "
          f"negative {100 * (p < 0).mean():.1f}%")


if __name__ == "__main__":
    main()
