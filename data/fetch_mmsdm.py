#!/usr/bin/env python3
"""Fetch AEMO MMSDM monthly tables for the years the panel covers.

Why DISPATCHREGIONSUM and not DISPATCHLOAD
------------------------------------------
The thing missing from the 37-channel panel is *scarcity*. 115 spike rows carry
23.5% of the total error and persistence beats every trained model on them,
which says the driver is not in the inputs rather than that the model is weak.
The physical driver is the gap between what is available and what is demanded,
and DISPATCHREGIONSUM carries it per region per five minutes:

  AVAILABLEGENERATION - TOTALDEMAND   reserve margin
  LORSURPLUS, LRCSURPLUS              surplus against the reliability standards
  RAISE*ACTUALAVAILABILITY            FCAS headroom, which tightens first
  UIGF                                the *forecast* intermittent generation,
                                      not the realised weather the panel uses
  DEMANDFORECAST                      AEMO's own short-term demand forecast

DISPATCHLOAD would give the same thing per unit at 67 MB a month against 5.2,
and needs a DUID-to-region mapping to be useful. Region level is enough.

A caveat that has to survive into the feature builder: UIGF and DEMANDFORECAST
are forecasts *made for* the interval, so they are legitimate at the forecast
origin. AVAILABLEGENERATION and TOTALDEMAND are the realised dispatch outcome
and are only legitimate for intervals at or before the origin. Lagging is the
feature builder's job, not this script's.

    python -m data.fetch_mmsdm --start 2018-01 --end 2021-12
"""
import argparse
import os
import subprocess
import time

BASE = ("https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        "{y}/MMSDM_{y}_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
        "PUBLIC_DVD_{t}_{y}{m:02d}010000.zip")
TABLES = ["DISPATCHREGIONSUM", "PREDISPATCHREGIONSUM_D", "PREDISPATCHPRICE_D"]


def months(start, end):
    y0, m0 = (int(v) for v in start.split("-"))
    y1, m1 = (int(v) for v in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2021-12")
    ap.add_argument("--out", default="data/raw/mmsdm")
    ap.add_argument("--tables", default=",".join(TABLES))
    a = ap.parse_args()
    tabs = [t for t in a.tables.split(",") if t]
    got = skipped = failed = 0
    for t in tabs:
        d = os.path.join(a.out, t)
        os.makedirs(d, exist_ok=True)
        for y, m in months(a.start, a.end):
            dst = os.path.join(d, f"{y}{m:02d}.zip")
            if os.path.exists(dst) and os.path.getsize(dst) > 1024:
                skipped += 1
                continue
            url = BASE.format(y=y, m=m, t=t)
            # curl, not urllib. On a machine behind the GT VPN the TLS chain is
            # intercepted and re-signed by a CA that lives in the system
            # keychain but not in certifi, so every urllib fetch dies on
            # CERTIFICATE_VERIFY_FAILED while curl succeeds. Shelling out uses
            # the system trust store; turning verification off instead would
            # trade a real security property for a convenience.
            tmp = dst + ".part"          # atomic: a half-written zip that
            try:                         # passes an existence check is the
                r = subprocess.run(      # resume trap this repo already paid
                    ["curl", "-sSL", "--fail", "--max-time", "300",
                     "-o", tmp, url],    # for once.
                    capture_output=True, text=True)
                if r.returncode != 0 or not os.path.exists(tmp) \
                        or os.path.getsize(tmp) < 1024:
                    raise RuntimeError(r.stderr.strip() or f"curl {r.returncode}")
                os.replace(tmp, dst)
                got += 1
                print(f"  {t} {y}-{m:02d}  {os.path.getsize(dst)/1e6:.1f} MB",
                      flush=True)
            except Exception as e:
                failed += 1
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"  {t} {y}-{m:02d}  FAILED {e}", flush=True)
            time.sleep(0.4)          # be a polite guest on a public archive
    print(f"\nfetched {got}, already present {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
