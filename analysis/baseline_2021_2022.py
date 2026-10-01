"""Baselines on the unfiltered SA1 series, train 2018-2020 / test 2021-2022.

The target is the raw settlement price. The filter that produced
`SA_filtered_2018_2022.csv` drops every negative price and everything above
981.65, which is 1.6% of 2018 and 20.7% of 2022 -- it removes the part of the
series that carries most of the error, and it removes a different fraction of
each year, so train and test are filtered to different distributions.

Reported per test year, because 2022 contains the June market suspension and
scores 2.4x the persistence error of 2021.
"""
import argparse
import numpy as np
import pandas as pd

LAGS = 48


def load(path):
    d = pd.read_csv(path, usecols=["SETTLEMENTDATE", "RRP"],
                    parse_dates=["SETTLEMENTDATE"])
    d = d.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE")
    return d.reset_index(drop=True)


def design(x, lags):
    n = len(x) - lags
    return np.lib.stride_tricks.sliding_window_view(x, lags)[:n], x[lags:]


def fit(xtr, lags):
    X, y = design(xtr, lags)
    A = np.hstack([X, np.ones((len(X), 1))])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def predict(w, xte, lags):
    X, y = design(xte, lags)
    return np.hstack([X, np.ones((len(X), 1))]) @ w, y


def metrics(pred, y):
    e = np.abs(pred - y)
    return e.mean(), np.sqrt(((pred - y) ** 2).mean())


def spike_split(y_prev, y, pred, q=99.0):
    move = np.abs(y - y_prev)
    thr = np.percentile(move, q)
    hot = move >= thr
    e = np.abs(pred - y)
    return e[hot].mean(), e[~hot].mean(), hot.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/SA_prices_combined_2018_2022.csv")
    ap.add_argument("--train", default="2018,2019,2020")
    ap.add_argument("--test", default="2021,2022")
    args = ap.parse_args()

    d = load(args.csv)
    d["yr"] = d.SETTLEMENTDATE.dt.year
    tr_years = [int(y) for y in args.train.split(",")]
    te_years = [int(y) for y in args.test.split(",")]

    xtr = d[d.yr.isin(tr_years)].RRP.to_numpy(float)
    print(f"train {tr_years}  n={len(xtr)}")

    # asinh around the training median, scaled by the training IQR, which is
    # the variance-stabilising transform EPF uses for heavy-tailed prices.
    med = np.median(xtr)
    iqr = np.percentile(xtr, 75) - np.percentile(xtr, 25)
    fwd = lambda z: np.arcsinh((z - med) / iqr)
    inv = lambda z: np.sinh(z) * iqr + med

    w_raw = fit(xtr, LAGS)
    w_asinh = fit(fwd(xtr), LAGS)

    rows = []
    for label, years in [(str(y), [y]) for y in te_years] + [("+".join(map(str, te_years)), te_years)]:
        xte = d[d.yr.isin(years)].RRP.to_numpy(float)
        _, y = design(xte, LAGS)
        y_prev = xte[LAGS - 1:-1]

        preds = {
            "persistence": y_prev,
            "AR(48) raw": predict(w_raw, xte, LAGS)[0],
            "AR(48) asinh": inv(predict(w_asinh, fwd(xte), LAGS)[0]),
        }
        for name, p in preds.items():
            mae, rmse = metrics(p, y)
            hot, cold, frac = spike_split(y_prev, y, p)
            rows.append(dict(test=label, model=name, n=len(y), mae=mae, rmse=rmse,
                             mae_top1pct=hot, mae_rest=cold))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))

    print("\nasinh vs raw, MAE:")
    for label in out.test.unique():
        s = out[out.test == label].set_index("model").mae
        print(f"  {label:>12}  {s['AR(48) raw'] - s['AR(48) asinh']:+7.3f}"
              f"   ({(s['AR(48) raw'] - s['AR(48) asinh']) / s['AR(48) raw'] * 100:+5.1f}%)")


if __name__ == "__main__":
    main()
