#!/usr/bin/env python3
"""Join the scarcity and interconnector channels onto the joint panel.

What goes in, and the honest expectation
----------------------------------------
Two blocks the 37-channel panel does not carry:

  scarcity   18 channels from DISPATCHREGIONSUM -- per-region reserve margin
             and its within-half-hour minimum, the forecast intermittent
             share, FCAS headroom and normalised demand
  weather    temperature, hub-height wind and irradiance at a demand centre
             and a renewable zone for NSW1, QLD1, TAS1 and VIC1. Scored on all
             five regions, the band architecture's gain tracks exactly whether
             a region has weather of its own -- SA1 -3.2% and VIC1 -2.9% have
             it, NSW1 -0.6%, QLD1 -1.3% and TAS1 -0.9% do not -- so this is a
             prediction that those three rows move, and a falsification of the
             mechanism if they do not
  ic         the 5 interconnector channels the compound panel DROPPED. The
             builder produced 12, of which `icbind_*`, `icbindrate_*` and
             `ic_any_bind` -- the constraint shadow prices, the only direct
             measurement of a binding link -- never made it into any panel.

The expectation is low and it is written down before the run. A single-feature
test at the forecast origin on 2021 says the panel already contains comparable
information: `wind100_nsa_wind`, already a channel, separates spikes at
AUC 0.748, while the best new feature, `uigfshare_SA1`, manages 0.720 and is
74% explained by a linear fit on nine existing channels -- its residual
separates spikes at 0.511, which is chance.

So this is not expected to unlock spikes. The reason to run it anyway is that
the test was single-feature and linear, and reserve margin is the *mechanism*
even where it is a weak marginal signal. If it does nothing, that is a result:
it says the 23.5% of error sitting on 115 rows is a base-rate problem, not a
feature-availability problem, and no amount of this kind of data fixes it.

    python -m data.add_scarcity_channels
"""
import argparse
import json
import os

import pandas as pd

WX_SITES = ["sydney", "nsw_rez", "brisbane", "qld_rez", "hobart",
            "tas_wind", "vic_wind"]

IC_KEEP = ["icbind_hey", "icbindrate_hey", "icbind_mur", "icbindrate_mur",
           "ic_any_bind", "icflow_hey", "ichead_hey", "icflow_mur",
           "ichead_mur", "ic_total_import"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--scarcity", default="data/raw/scarcity_2018_2022.csv")
    ap.add_argument("--ic", default="data/raw/ic_features_2018_2022.csv")
    ap.add_argument("--wx", default="data/raw/wx_allregions.json")
    ap.add_argument("--out", default="data/raw/compound_joint_sc_2018_2022.csv")
    a = ap.parse_args()

    p = pd.read_csv(a.panel, parse_dates=["SETTLEMENTDATE"])
    n0, c0 = p.shape

    # Weather for the four regions that had none. Hourly -> 30 min by time
    # interpolation, the same way build_compound_panel.py handles the four
    # sites already in the panel, so the new columns are built identically to
    # the old ones and a per-region comparison is not confounded by how the
    # feature was made.
    if os.path.exists(a.wx):
        raw = json.load(open(a.wx))
        locs = raw if isinstance(raw, list) else [raw]
        wx = None
        for site, loc in zip(WX_SITES, locs):
            h = loc["hourly"]
            d = pd.DataFrame({
                "SETTLEMENTDATE": pd.to_datetime(h["time"]),
                f"temp_{site}": h["temperature_2m"],
                f"wind100_{site}": h["wind_speed_100m"],
                f"solar_{site}": h["shortwave_radiation"]})
            wx = d if wx is None else wx.merge(d, on="SETTLEMENTDATE")
        wx = (wx.set_index("SETTLEMENTDATE").resample("30min").interpolate()
              .reset_index())
        before = len(p)
        p = p.merge(wx, on="SETTLEMENTDATE", how="left")
        assert len(p) == before, "weather merge changed the row count"
        cols = [c for c in wx.columns if c != "SETTLEMENTDATE"]
        gap = p[cols[0]].isna()
        yrs = sorted(p.loc[gap, "SETTLEMENTDATE"].dt.year.unique().tolist())
        p[cols] = p[cols].ffill().bfill()
        print(f"  + {len(cols):2d} from {os.path.basename(a.wx)}"
              f"   ({int(gap.sum())} rows filled"
              + (f", in {yrs}" if yrs else "") + ")")
    for path, keep in ((a.scarcity, None), (a.ic, IC_KEEP)):
        if not os.path.exists(path):
            print(f"  skip {path}: not present")
            continue
        d = pd.read_csv(path, parse_dates=["SETTLEMENTDATE"])
        cols = [c for c in (keep or d.columns) if c in d.columns
                and c != "SETTLEMENTDATE"]
        before = len(p)
        p = p.merge(d[["SETTLEMENTDATE"] + cols], on="SETTLEMENTDATE",
                    how="left")
        assert len(p) == before, "merge changed the row count"
        gap = p[cols[0]].isna()
        miss = int(p[cols].isna().sum().sum())
        # The panel's row set is the thing every arm is scored on, so a join
        # must never drop or duplicate a row. Gaps are carried forward in time,
        # which is what a forecaster at the origin would have -- but a filled
        # row inside a training or test year is a fabricated feature, so say
        # which years were touched rather than only how many cells.
        years = sorted(p.loc[gap, "SETTLEMENTDATE"].dt.year.unique().tolist())
        p[cols] = p[cols].ffill().bfill()
        print(f"  + {len(cols):2d} from {os.path.basename(path)}"
              f"   ({miss} cells filled"
              + (f", in {years}" if years else "") + ")")
        if set(years) & {2018, 2019, 2020, 2021}:
            print("    WARNING: filled rows fall inside the train/test years")

    assert len(p) == n0, "row count changed"
    p.to_csv(a.out, index=False)
    print(f"\n-> {a.out}\n   {n0:,} rows, {c0 - 1} -> {p.shape[1] - 1} channels")


if __name__ == "__main__":
    main()
