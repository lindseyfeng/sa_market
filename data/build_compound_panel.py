#!/usr/bin/env python3
"""
Build the compound multi-channel exogenous signal for spatio-temporal NVMD.

Why not raw neighbouring prices
-------------------------------
The first spatial attempt used the five regional price *levels* and produced
nothing (14.30 vs 14.31, incoherent coupling matrix).  The reason is economic:
interconnectors arbitrage price levels together whenever unconstrained
(SA1-VIC1 correlation 0.914), so a neighbour's price carries almost nothing
SA's own price does not already contain.

This panel uses only non-redundant channels:

  market    regional DEMAND (physically local, not arbitraged) and price
            SPREADS (the level is the arbitraged part; the spread sits near
            zero while a link is unconstrained and diverges when it binds,
            making it a congestion proxy rather than a redundant copy).
  weather   wind speed at 100 m (turbine hub height) at two SA wind regions,
            temperature at the SA and VIC demand centres, and solar radiation.
            Wind is SA's dominant price driver and is genuinely exogenous.
  calendar  sin/cos of daily, weekly and annual cycles plus a weekend flag.
            These are *known future inputs* -- available at prediction time
            for any horizon -- and are pure single-frequency waves, so they
            land in known bands of the decomposition.

Channel 0 is always the target (SA1 price) so the decomposer still emits the
target's own modes and the coupling ablation stays exact.

CAVEAT: weather here is reanalysis (what actually happened), not forecast.
Treat weather-inclusive results as an upper bound unless re-run with forecast
weather.

    python build_compound_panel.py --out compound_2018_2022.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

REGIONS = ["SA1", "NSW1", "VIC1", "QLD1", "TAS1"]
WX_SITES = ["adelaide", "nsa_wind", "sesa_wind", "melbourne"]
LO, HI = 1.0, 981.65


def load_region(region, src_dir):
    files = sorted(glob.glob(os.path.join(src_dir, f"*_{region}.csv")))
    if not files:
        raise FileNotFoundError(f"no files for {region}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df[["SETTLEMENTDATE", "RRP", "TOTALDEMAND"]].sort_values("SETTLEMENTDATE")
    df = df.drop_duplicates("SETTLEMENTDATE")
    return df.rename(columns={"RRP": f"{region}_p", "TOTALDEMAND": f"{region}_d"})


def load_weather(path):
    raw = json.load(open(path))
    locs = raw if isinstance(raw, list) else [raw]
    out = None
    for site, loc in zip(WX_SITES, locs):
        h = loc["hourly"]
        d = pd.DataFrame({
            "SETTLEMENTDATE": pd.to_datetime(h["time"]),
            f"temp_{site}": h["temperature_2m"],
            f"wind100_{site}": h["wind_speed_100m"],
            f"solar_{site}": h["shortwave_radiation"],
        })
        out = d if out is None else out.merge(d, on="SETTLEMENTDATE", how="outer")
    return out.sort_values("SETTLEMENTDATE").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-csv", default="SA_filtered_2018_2022.csv")
    ap.add_argument("--src-dir", default="nem_regions")
    ap.add_argument("--weather", default="wx_raw.json")
    ap.add_argument("--out", default="compound_2018_2022.csv")
    args = ap.parse_args()

    tgt = pd.read_csv(args.target_csv, parse_dates=["SETTLEMENTDATE"])
    tgt = tgt.rename(columns={"RRP": "SA1_price"})[["SETTLEMENTDATE", "SA1_price"]]
    print(f"target SA1 price (filtered): {len(tgt)} rows")

    panel = tgt
    for r in REGIONS:
        panel = panel.merge(load_region(r, args.src_dir), on="SETTLEMENTDATE", how="left")
    for r in REGIONS:
        panel[f"{r}_p"] = panel[f"{r}_p"].ffill().bfill().clip(LO, HI)
        panel[f"{r}_d"] = panel[f"{r}_d"].ffill().bfill()

    out = pd.DataFrame({"SETTLEMENTDATE": panel["SETTLEMENTDATE"]})
    out["SA1_price"] = panel["SA1_price"].values                 # channel 0: target

    # --- market: un-arbitraged local demand -------------------------------
    for r in REGIONS:
        out[f"demand_{r}"] = panel[f"{r}_d"].values
    out["demand_NEM"] = panel[[f"{r}_d" for r in REGIONS]].sum(axis=1).values

    # --- market: congestion proxies ---------------------------------------
    for r in REGIONS[1:]:
        out[f"spread_SA1_{r}"] = (panel["SA1_p"] - panel[f"{r}_p"]).values

    # --- market: dispatch pressure ----------------------------------------
    out["ramp_SA1"] = panel["SA1_d"].diff().fillna(0.0).values
    out["ramp_VIC1"] = panel["VIC1_d"].diff().fillna(0.0).values
    roll = panel["SA1_d"].rolling(336, min_periods=1).max()      # 7d of 30-min
    out["scarcity_SA1"] = (panel["SA1_d"] / roll).values          # -> 1 at peak

    # --- weather (hourly -> 30 min by time interpolation) ------------------
    if os.path.exists(args.weather):
        wx = load_weather(args.weather)
        merged = pd.merge_asof(
            out[["SETTLEMENTDATE"]].sort_values("SETTLEMENTDATE"),
            wx.sort_values("SETTLEMENTDATE"),
            on="SETTLEMENTDATE", direction="nearest",
            tolerance=pd.Timedelta("60min"),
        )
        for c in merged.columns:
            if c != "SETTLEMENTDATE":
                out[c] = merged[c].interpolate().ffill().bfill().values
        print(f"weather merged: {len([c for c in merged.columns if c != 'SETTLEMENTDATE'])} channels")
    else:
        print(f"WARNING: {args.weather} missing -- no weather channels")

    # --- calendar: known future inputs, pure single-frequency waves --------
    t = out["SETTLEMENTDATE"]
    tod = t.dt.hour + t.dt.minute / 60.0
    out["cal_day_sin"] = np.sin(2 * np.pi * tod / 24)
    out["cal_day_cos"] = np.cos(2 * np.pi * tod / 24)
    out["cal_week_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    out["cal_week_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    doy = t.dt.dayofyear
    out["cal_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["cal_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["cal_weekend"] = (t.dt.dayofweek >= 5).astype(float)

    out = out.dropna().reset_index(drop=True)
    out.to_csv(args.out, index=False)

    chans = [c for c in out.columns if c != "SETTLEMENTDATE"]
    print(f"\nwrote {args.out}: {len(out)} rows x {len(chans)} channels")
    print("channel 0 =", chans[0], "(target)")

    c = out[chans].corr()["SA1_price"].drop("SA1_price")
    print("\ncorrelation with target, strongest first:")
    for k, v in c.reindex(c.abs().sort_values(ascending=False).index).items():
        print(f"  {k:<22s} {v:+.3f}")


if __name__ == "__main__":
    main()
