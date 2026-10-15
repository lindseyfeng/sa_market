#!/usr/bin/env python3
"""Scarcity features from AEMO DISPATCHREGIONSUM, for all five NEM regions.

Why this is the missing input rather than another architecture
--------------------------------------------------------------
115 spike rows are 0.7% of the 2021 test set and carry 23.5% of the total
absolute error, and on those rows persistence beats every trained model. A
per-regime oracle over the four best models buys 3.3%. That pattern -- nothing
learns it, the naive method wins, routing cannot reach it -- is what missing
information looks like, not what an underpowered model looks like. The 37
channels carry price, demand, weather and calendar; none of them says how much
headroom the region has.

  reserve      AVAILABLEGENERATION - TOTALDEMAND, the mechanism itself: a
               spike is what happens when the offer stack runs out. Normalised
               by a CONSTANT -- the region's median demand over the training
               years -- and not by instantaneous demand. SA1 operational
               demand genuinely approaches zero on mild high-solar days
               (rooftop PV is netted off it), so a ratio to live demand
               explodes: reserve/demand reached 1,836 on 2021-12-27. That is
               real market behaviour mis-parameterised, not a bad row
  reserve_min  the tightest five minutes inside the half hour, because a
               30-minute price is set by the worst dispatch interval in it
  uigf_share   UIGF / demand. UIGF is AEMO's *forecast* of intermittent
               output, so unlike the panel's reanalysis weather it is
               available at the forecast origin and is not an upper bound
  raisereg     regulation-raise FCAS availability, which tightens before
               energy does

Causality: everything here is read inside the input window, which ends at the
forecast origin, so realised quantities are legitimate. UIGF and DEMANDFORECAST
are forecasts made *for* their interval and would be legitimate past the origin
too; nothing here uses them that way.

    python -m data.build_scarcity --out data/raw/scarcity_2018_2022.csv
"""
import argparse
import csv
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

REGIONS = ["SA1", "NSW1", "VIC1", "QLD1", "TAS1"]
# LORSURPLUS and LRCSURPLUS are in the schema but null for the whole period --
# they are legacy reserve-condition fields AEMO stopped populating -- so they
# are not read. A column of NaN that gets forward-filled is worse than no
# column: it looks like a feature.
WANT = ["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "TOTALDEMAND",
        "AVAILABLEGENERATION", "DEMANDFORECAST", "UIGF",
        "RAISEREGACTUALAVAILABILITY"]


def read_zip(path):
    zf = zipfile.ZipFile(path)
    rows, hdr = [], None
    with zf.open(zf.namelist()[0]) as fh:
        for row in csv.reader(io.TextIOWrapper(fh, errors="replace")):
            if not row:
                continue
            if row[0] == "I":
                hdr = row
                idx = {w: hdr.index(w) for w in WANT if w in hdr}
            elif row[0] == "D" and hdr is not None:
                # INTERVENTION=1 rows are the re-run after a manual
                # intervention; mixing them in double-counts intervals.
                if row[idx["INTERVENTION"]] != "0":
                    continue
                if row[idx["REGIONID"]] not in REGIONS:
                    continue
                rows.append([row[idx[w]] if w in idx else "" for w in WANT])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/mmsdm/DISPATCHREGIONSUM")
    ap.add_argument("--out", default="data/raw/scarcity_2018_2022.csv")
    ap.add_argument("--train-end", default="2021-01-01",
                    help="normalising constants use data strictly before this")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*.zip")))
    if not files:
        raise SystemExit(f"no zips under {a.src}")
    allrows = []
    for i, f in enumerate(files, 1):
        allrows.extend(read_zip(f))
        print(f"  [{i}/{len(files)}] {os.path.basename(f)}  "
              f"{len(allrows):,} rows", flush=True)

    df = pd.DataFrame(allrows, columns=WANT)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    for c in WANT[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 5-min -> the 30-minute trading interval ENDING at T, the same convention
    # build_ic_features.py uses, so the two join on one index.
    df["TI"] = df["SETTLEMENTDATE"].dt.ceil("30min")

    # One constant scale per region, from the training years only, so nothing
    # about 2021 leaks into how 2018-2020 is represented.
    train = df[df.SETTLEMENTDATE < pd.Timestamp(a.train_end)]
    scale = (train.groupby("REGIONID")["TOTALDEMAND"].median()
             .clip(lower=100.0).to_dict())
    print("\nnormalising scale (median train demand, MW): "
          + "  ".join(f"{k} {v:.0f}" for k, v in sorted(scale.items())))

    out = None
    for r in REGIONS:
        g = df[df.REGIONID == r].copy()
        sc = scale[r]
        g["reserve"] = (g["AVAILABLEGENERATION"] - g["TOTALDEMAND"]) / sc
        g["reserve_frac"] = g["reserve"]
        g["uigf_share"] = g["UIGF"] / sc
        agg = g.groupby("TI").agg(
            **{f"reserve_{r}": ("reserve_frac", "mean"),
               f"reservemin_{r}": ("reserve_frac", "min"),
               f"uigfshare_{r}": ("uigf_share", "mean")})
        if r == "SA1":
            extra = g.groupby("TI").agg(
                raisereg_SA1=("RAISEREGACTUALAVAILABILITY", "min"),
                demandfc_SA1=("DEMANDFORECAST", "mean"),
                demand_norm_SA1=("TOTALDEMAND", "mean"))
            extra["demand_norm_SA1"] /= sc
            agg = agg.join(extra)
        out = agg if out is None else out.join(agg, how="outer")

    out.index.name = "SETTLEMENTDATE"
    out = out.sort_index()
    n_nan = int(out.isna().sum().sum())
    out = out.ffill().bfill()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out)
    print(f"\n-> {a.out}  {len(out):,} intervals x {out.shape[1]} channels "
          f"({n_nan} missing cells filled)")
    print(f"   {out.index.min()} .. {out.index.max()}")
    print(out.describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
