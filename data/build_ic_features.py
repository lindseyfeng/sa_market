#!/usr/bin/env python3
"""Interconnector features for SA from AEMO DISPATCHINTERCONNECTORRES.

Why this beats the price-spread proxy
-------------------------------------
The compound panel proxies congestion with SA1-X price spreads: near zero while a
link is unconstrained, diverging when it binds.  That is an *inference*.  This is
the measurement:

  MARGINALVALUE  shadow price of the interconnector constraint -- non-zero exactly
                 when the link binds.  Direct congestion, no inference.
  MWFLOW         signed flow (positive = into SA)
  headroom       (EXPORTLIMIT - flow) etc: how close to the limit the link is
                 running, which is what predicts a bind before it happens.

SA's links are V-SA (Heywood) and V-S-MNSP1 (Murraylink).
5-minute dispatch is averaged onto the 30-minute trading interval ending at T.
"""
import glob, io, csv, zipfile, os
import numpy as np, pandas as pd

SA_LINKS = ["V-SA", "V-S-MNSP1"]
WANT = ["SETTLEMENTDATE","INTERCONNECTORID","INTERVENTION","MWFLOW",
        "MARGINALVALUE","EXPORTLIMIT","IMPORTLIMIT","MWLOSSES"]

def read_zip(z):
    zf = zipfile.ZipFile(z)
    out = []
    with zf.open(zf.namelist()[0]) as fh:
        cols = None
        for line in io.TextIOWrapper(fh, "utf-8"):
            r = next(csv.reader([line]))
            if r[0] == "I":
                cols = r[4:]
            elif r[0] == "D" and cols:
                d = dict(zip(cols, r[4:]))
                if d.get("INTERCONNECTORID") in SA_LINKS and d.get("INTERVENTION") == "0":
                    out.append([d.get(k) for k in WANT])
    return out

rows = []
files = sorted(glob.glob("nem_ic/*.zip"))
for i, z in enumerate(files):
    rows += read_zip(z)
    if (i + 1) % 12 == 0:
        print(f"  {i+1}/{len(files)} zips, {len(rows):,} rows", flush=True)

df = pd.DataFrame(rows, columns=WANT)
df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
for c in ["MWFLOW","MARGINALVALUE","EXPORTLIMIT","IMPORTLIMIT","MWLOSSES"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"parsed {len(df):,} 5-min rows, {df.SETTLEMENTDATE.min()} -> {df.SETTLEMENTDATE.max()}")

# 5-min -> 30-min trading interval ENDING at T (ceil to next half hour)
df["TI"] = df["SETTLEMENTDATE"].dt.ceil("30min")

out = pd.DataFrame({"SETTLEMENTDATE": sorted(df.TI.unique())}).set_index("SETTLEMENTDATE")
for link in SA_LINKS:
    g = df[df.INTERCONNECTORID == link].groupby("TI")
    tag = "hey" if link == "V-SA" else "mur"
    out[f"icflow_{tag}"] = g["MWFLOW"].mean()
    # binding: shadow price non-zero.  Keep both the magnitude and the rate.
    out[f"icbind_{tag}"] = g["MARGINALVALUE"].apply(lambda s: s.abs().mean())
    out[f"icbindrate_{tag}"] = g["MARGINALVALUE"].apply(lambda s: (s.abs() > 1e-6).mean())
    # headroom to the export limit, the leading indicator of a bind
    out[f"ichead_{tag}"] = g.apply(
        lambda x: (x["EXPORTLIMIT"] - x["MWFLOW"]).mean(), include_groups=False)
    out[f"icloss_{tag}"] = g["MWLOSSES"].mean()

out["ic_total_import"] = out["icflow_hey"] + out["icflow_mur"]
out["ic_any_bind"] = out[["icbindrate_hey","icbindrate_mur"]].max(axis=1)
out = out.sort_index().ffill().bfill()
out.to_csv("ic_features_2018_2022.csv")
print(f"\nwrote ic_features_2018_2022.csv: {len(out)} rows x {out.shape[1]} cols")
print(out.describe().T[["mean","std","min","max"]].round(2).to_string())
