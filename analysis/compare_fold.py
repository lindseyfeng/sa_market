"""Compare saved predictions against persistence on identical rows.

Every arm is scored on the same window starts, so the comparison is paired:
the spike split and the per-month curve below are differences on the same
timestamps, not two independently computed averages.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

L, H = 96, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--price", default="data/raw/SA_unfiltered_2018_2022.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.price, parse_dates=["SETTLEMENTDATE"])
    col = [c for c in d.columns if c != "SETTLEMENTDATE"][0]
    te = d[d.SETTLEMENTDATE.dt.year == args.year].reset_index(drop=True)
    p = te[col].to_numpy(float)
    ts = te.SETTLEMENTDATE

    rows = []
    for f in sorted(glob.glob(os.path.join(args.preds, "*.npz"))):
        z = np.load(f)
        idx = z["start"] + L + H - 1
        prev = p[idx - 1]
        truth = p[idx]
        e_m = np.abs(z["pred"] - truth)
        e_p = np.abs(prev - truth)
        hot = e_p >= np.percentile(e_p, 99)
        rows.append(dict(arm=os.path.basename(f)[:-4],
                         mae=e_m.mean(), persistence=e_p.mean(),
                         gain=e_p.mean() - e_m.mean(),
                         spike=e_m[hot].mean(), spike_pers=e_p[hot].mean(),
                         calm=e_m[~hot].mean(), calm_pers=e_p[~hot].mean()))
    out = pd.DataFrame(rows).sort_values("mae")
    pd.set_option("display.width", 160)
    print(out.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    print("\nper-month MAE")
    month = ts.iloc[0:0]
    hdr = None
    for f in sorted(glob.glob(os.path.join(args.preds, "*.npz"))):
        z = np.load(f)
        idx = z["start"] + L + H - 1
        m = ts.iloc[idx].dt.to_period("M").to_numpy()
        e = np.abs(z["pred"] - p[idx])
        s = pd.Series(e).groupby(m).mean()
        if hdr is None:
            pers = pd.Series(np.abs(p[idx - 1] - p[idx])).groupby(m).mean()
            print("  " + "persistence".ljust(18) +
                  " ".join(f"{v:7.1f}" for v in pers.values))
            hdr = True
        print("  " + os.path.basename(f)[:-4].ljust(18) +
              " ".join(f"{v:7.1f}" for v in s.values))


if __name__ == "__main__":
    main()
