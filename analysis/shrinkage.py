"""Scale the predicted change by a factor that depends on the regime.

At h=1 both AR(48) and nvmd_st get the direction of the next change right
58-62% of the time, which is real but thin: under MAE you win on the 60% and
pay double on the 40%, so the magnitude matters more than the direction. A
global factor already recovers most of nvmd_st's deficit (27.34 -> 26.79 at
0.8). The factor should not be global, though: the models beat persistence by
27% on the top 1% of moves and lose to it on the calm 90%, so confidence is
regime-dependent.

The factor is fitted on the first half of the test year and applied to the
second, so nothing here is fitted on what it is scored against.
"""
import argparse
import numpy as np
import pandas as pd

L, H, LAGS = 96, 1, 48
GRID = np.linspace(0.0, 1.2, 25)


def asinh_ar(tr, p, idx):
    med = np.median(tr)
    iqr = np.percentile(tr, 75) - np.percentile(tr, 25)
    f = lambda x: np.arcsinh((x - med) / iqr)
    inv = lambda z: np.sinh(z) * iqr + med
    A = np.lib.stride_tricks.sliding_window_view(f(tr), LAGS)[:-1]
    w, *_ = np.linalg.lstsq(np.hstack([A, np.ones((len(A), 1))]), f(tr)[LAGS:],
                            rcond=None)
    X = np.stack([f(p)[i - LAGS:i] for i in idx])
    return inv(np.hstack([X, np.ones((len(X), 1))]) @ w)


def best_alpha(dhat, dtrue):
    return GRID[np.argmin([np.abs(g * dhat - dtrue).mean() for g in GRID])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds/fold2021/nvmd_st_s1.npz")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    d = pd.read_csv("data/raw/SA_unfiltered_2018_2022.csv",
                    parse_dates=["SETTLEMENTDATE"])
    col = [c for c in d.columns if c != "SETTLEMENTDATE"][0]
    tr = d[d.SETTLEMENTDATE.dt.year.isin([2018, 2019, 2020])][col].to_numpy(float)
    p = d[d.SETTLEMENTDATE.dt.year == args.year][col].to_numpy(float)

    z = np.load(args.pred)
    idx = z["start"] + L + H - 1
    truth, prev = p[idx], p[idx - 1]
    dtrue = truth - prev

    models = {"AR(48)": asinh_ar(tr, p, idx) - prev,
              "nvmd_st": z["pred"] - prev}

    # regime = how much the price has been moving lately, known at prediction time
    vol = np.array([np.abs(np.diff(p[i - LAGS:i])).mean() for i in idx])
    half = len(idx) // 2
    edges = np.quantile(vol[:half], np.linspace(0, 1, args.bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(vol, edges[1:-1])

    pers = np.abs(dtrue[half:]).mean()
    print(f"second half of {args.year}: persistence {pers:.3f}\n")
    for name, dhat in models.items():
        raw = np.abs(dhat[half:] - dtrue[half:]).mean()
        g = best_alpha(dhat[:half], dtrue[:half])
        flat = np.abs(g * dhat[half:] - dtrue[half:]).mean()
        adj = dhat.copy()
        alphas = []
        for k in range(args.bins):
            m = b == k
            a = best_alpha(dhat[:half][m[:half]], dtrue[:half][m[:half]])
            alphas.append(a)
            adj[m] = a * dhat[m]
        binned = np.abs(adj[half:] - dtrue[half:]).mean()
        print(f"{name:9} raw {raw:7.3f}   global a={g:.2f} {flat:7.3f}   "
              f"per-regime {binned:7.3f}   alphas {np.round(alphas, 2)}")


if __name__ == "__main__":
    main()
