#!/usr/bin/env python3
"""Weather for the four NEM regions that have none.

Why this and not another architecture
-------------------------------------
Scored on all five regions rather than SA1, the band architecture's gain
tracks one thing almost exactly:

    SA1  -3.2%   Adelaide + two SA wind zones
    VIC1 -2.9%   Melbourne
    QLD1 -1.3%   no weather
    TAS1 -0.9%   no weather
    NSW1 -0.6%   no weather

The exogenous panel was built for an SA1 forecast and reused unchanged when the
model went joint, so four of the five targets are predicted with no weather of
their own. That is a prediction, not a hope: give NSW1, QLD1 and TAS1 their own
temperature, hub-height wind and irradiance and those three rows should move.
If they do not, the mechanism above is wrong and the per-region pattern was a
coincidence.

Sites are a demand centre plus a renewable zone per region, because the two do
different jobs: temperature at the load centre drives demand, wind and
irradiance in the generation zone drive supply.

Matches the existing fetch exactly -- same three variables, same
Australia/Adelaide timezone, same span -- so the new columns align with the
four that are already in the panel.

    python -m data.fetch_weather_allregions
"""
import argparse
import json
import os
import subprocess

# (name, lat, lon, what it is for)
SITES = [
    ("sydney",      -33.868, 151.209, "NSW1 demand centre"),
    ("nsw_rez",     -32.250, 148.601, "NSW1 Central-West Orana renewable zone"),
    ("brisbane",    -27.470, 153.026, "QLD1 demand centre"),
    ("qld_rez",     -27.000, 150.500, "QLD1 Darling Downs solar/wind"),
    ("hobart",      -42.881, 147.325, "TAS1 demand centre"),
    ("tas_wind",    -40.900, 145.000, "TAS1 Woolnorth wind"),
    ("vic_wind",    -38.300, 142.900, "VIC1 Macarthur wind zone"),
]
VARS = "temperature_2m,wind_speed_100m,shortwave_radiation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--out", default="data/raw/wx_allregions.json")
    a = ap.parse_args()

    lat = ",".join(f"{s[1]}" for s in SITES)
    lon = ",".join(f"{s[2]}" for s in SITES)
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={a.start}&end_date={a.end}&hourly={VARS}"
           "&timezone=Australia%2FAdelaide")
    # curl, not urllib: behind the GT VPN the TLS chain is re-signed by a CA
    # that is in the system keychain but not in certifi, so urllib dies on
    # CERTIFICATE_VERIFY_FAILED while curl succeeds.
    r = subprocess.run(["curl", "-sSL", "--fail", "--max-time", "600",
                        "-o", a.out, url], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"fetch failed: {r.stderr.strip()}")
    d = json.load(open(a.out))
    locs = d if isinstance(d, list) else [d]
    print(f"-> {a.out}   {len(locs)} sites")
    for (nm, la, lo, why), loc in zip(SITES, locs):
        h = loc["hourly"]
        print(f"  {nm:10s} ({loc['latitude']:.3f}, {loc['longitude']:.3f})  "
              f"{len(h['time']):,} hours  {h['time'][0]} .. {h['time'][-1]}"
              f"   {why}")


if __name__ == "__main__":
    main()
