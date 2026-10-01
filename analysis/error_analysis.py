"""Where each model's error lives, on identical rows.

MAE on this series is an average over a distribution that is 37% one percent of
its own points, so a single number says very little. Every split below is on
the same timestamps for every model, so the columns are directly comparable.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

L = 96


def ar_asinh(tr, te, idx, n_lags, h):
    med = np.median(tr)
    iqr = np.percentile(tr, 75) - np.percentile(tr, 25)
    f = lambda x: np.arcsinh((x - med) / iqr)
    inv = lambda z: np.sinh(z) * iqr + med
    A = np.lib.stride_tricks.sliding_window_view(f(tr), n_lags)[:-h]
    y = f(tr)[n_lags + h - 1:]
    m = min(len(A), len(y))
    w, *_ = np.linalg.lstsq(np.hstack([A[:m], np.ones((m, 1))]), y[:m], rcond=None)
    end = idx - h + 1
    X = np.stack([f(te)[e - n_lags:e] for e in end])
    return inv(np.hstack([X, np.ones((len(X), 1))]) @ w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--train", default="2018")
    ap.add_argument("--test", type=int, default=2019)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    d = pd.read_csv("data/raw/SA_unfiltered_2018_2022.csv",
                    parse_dates=["SETTLEMENTDATE"])
    col = [c for c in d.columns if c != "SETTLEMENTDATE"][0]
    yrs = [int(v) for v in args.train.split(",")]
    tr = d[d.SETTLEMENTDATE.dt.year.isin(yrs)][col].to_numpy(float)
    sub = d[d.SETTLEMENTDATE.dt.year == args.test].reset_index(drop=True)
    te, ts = sub[col].to_numpy(float), sub.SETTLEMENTDATE

    files = sorted(glob.glob(os.path.join(args.preds, "*.npz")))
    if not files:
        raise SystemExit(f"no predictions in {args.preds}")
    idx = np.load(files[0])["start"] + L + args.horizon - 1
    truth = te[idx]
    prev = te[idx - args.horizon]

    models = {"persistence": prev, "AR(96) asinh": ar_asinh(tr, te, idx, 96, args.horizon)}
    for f in files:
        z = np.load(f)
        assert (z["start"] + L + args.horizon - 1 == idx).all()
        models[os.path.basename(f)[:-4].replace("_s1", "")] = z["pred"]

    move = np.abs(prev - truth)
    seg = {"spike 1%": move >= np.percentile(move, 99),
           "next 9%": (move >= np.percentile(move, 90)) & (move < np.percentile(move, 99)),
           "calm 90%": move < np.percentile(move, 90)}
    neg = truth < 0

    print(f"test {args.test}, h={args.horizon}, {len(idx)} rows "
          f"({100 * neg.mean():.1f}% negative prices)\n")
    hdr = f"{'':20}{'MAE':>8}{'RMSE':>9}" + "".join(f"{k:>11}" for k in seg) + \
          f"{'negative':>10}{'bias':>8}{'pred sd':>9}"
    print(hdr)
    for name, q in sorted(models.items(), key=lambda kv: np.abs(kv[1] - truth).mean()):
        e = np.abs(q - truth)
        row = f"{name:20}{e.mean():8.2f}{np.sqrt(((q-truth)**2).mean()):9.1f}"
        row += "".join(f"{e[m].mean():11.2f}" for m in seg.values())
        row += f"{e[neg].mean():10.2f}{(q-truth).mean():8.2f}{q.std():9.1f}"
        print(row)

    print(f"\ntruth sd {truth.std():.1f}   "
          f"share of total error in the top 1%: "
          f"{100 * np.abs(prev-truth)[seg['spike 1%']].sum() / np.abs(prev-truth).sum():.1f}% (persistence)")

    print("\nper-month MAE")
    mo = ts.iloc[idx].dt.month.to_numpy()
    print(f"{'':20}" + "".join(f"{m:>6}" for m in range(1, 13)))
    for name, q in models.items():
        s = pd.Series(np.abs(q - truth)).groupby(mo).mean()
        print(f"{name:20}" + "".join(f"{v:6.1f}" for v in s.values))


if __name__ == "__main__":
    main()
