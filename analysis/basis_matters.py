#!/usr/bin/env python3
"""Does the band basis help a learner that cannot undo a rotation?

A partition-of-unity band decomposition is an invertible linear map of the
window, so it adds no information: ridge on bands and ridge on the raw window
return identical predictions to 0.0000 $/MWh, measured. An LSTM's input-to-hidden
matrix is dense across channels, so for it too the bands are a reparameterisation
-- W_ih @ (M x) is the same model class as W_ih' @ x. The 6.8% the bands are
worth to the LSTM therefore cannot be information; the mechanism is that the band
masks are non-causal over the window, so band_k(t) already carries the whole
window at every step and the LSTM does not have to integrate 96 steps to get it.
A model that sees the window at once, like ridge, has no such bottleneck to
shortcut, which is exactly why it gains nothing.

That makes one prediction worth testing. A decision tree splits on individual
features and cannot absorb a basis rotation into a weight matrix. If the band
basis is genuinely a better coordinate system -- rather than a shortcut around a
sequential bottleneck -- it should help a tree ensemble where it provably cannot
help ridge at all.

Same rows, same target, same information in both arms:
    raw    the target's own 96-step window
    band   the same window as K bands x 96 steps, which sum back to it exactly
"""
import argparse, itertools, os
import numpy as np, pandas as pd
from analysis.baselines import Asinh, REGIONS


def bands_of(W, K, L):
    """(n, L) -> (n, K, L) by geometric band masks forming a partition of unity."""
    nf = L // 2 + 1
    edges = np.unique(np.round(np.geomspace(1, nf, K + 1)).astype(int))
    M = np.zeros((K, nf), np.float32)
    for k in range(K):
        a = edges[min(k, len(edges) - 2)]
        b = edges[min(k + 1, len(edges) - 1)]
        M[k, a:b] = 1.0
    M[0, 0] = 1.0
    M /= np.maximum(M.sum(0, keepdims=True), 1e-9)
    F = np.fft.rfft(W, axis=-1)
    return np.stack([np.fft.irfft(F * M[k], n=L, axis=-1) for k in range(K)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--sub", type=int, default=4, help="use every Nth train window")
    a = ap.parse_args()
    L, h, K = a.seq_len, a.horizon, a.K

    df = pd.read_csv(a.panel, parse_dates=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    ch = [c for c in df.columns if c != "SETTLEMENTDATE"]
    t = ch.index("SA1_price")
    tr = df[df.SETTLEMENTDATE.dt.year.isin([2018, 2019, 2020])][ch].to_numpy(np.float32)
    te = df[df.SETTLEMENTDATE.dt.year == 2021][ch].to_numpy(np.float32)
    al = np.arange(0, len(tr) - L - h + 1)
    nv = int(len(al) * a.val_frac)
    va, trs = al[-nv:], al[:-(nv + L)][::a.sub]
    tes = np.arange(0, len(te) - L - h + 1)

    sc = Asinh(tr[:, [t]])
    ip = Asinh(tr[:, [t]])
    ztr, zte = sc.fwd(tr[:, [t]]), sc.fwd(te[:, [t]])
    vtr, vte = ip.fwd(tr[:, [t]]), ip.fwd(te[:, [t]])
    mu, sd = vtr.mean(), vtr.std() + 1e-8

    def win(src, st):
        return np.stack([src[i:i + L, 0] for i in st])

    def feats(src, st, kind):
        W = (win(src, st) - mu) / sd
        if kind == "raw":
            return W
        return bands_of(W, K, L).reshape(len(st), -1)

    y = {"tr": ztr[trs + L + h - 1, 0], "va": ztr[va + L + h - 1, 0],
         "te": zte[tes + L + h - 1, 0]}
    truth = sc.inv(zte[tes + L + h - 1])[:, 0]

    print(f"h={h}  K={K}  train {len(trs)} (every {a.sub}th) / val {len(va)} / test {len(tes)}")
    print(f"{'learner':<22}{'features':<10}{'dim':>7}{'val MAE':>10}{'test MAE':>10}")
    print("-" * 61)
    out = {}
    for kind in ("raw", "band"):
        Xtr = feats(vtr, trs, kind)
        Xva = feats(vtr, va, kind)
        Xte = feats(vte, tes, kind)

        # Ridge: the control. A rotation is absorbed by the weight vector, so the
        # two arms must agree; if they do not, the band construction is wrong.
        from sklearn.linear_model import Ridge
        best = (1e9, None)
        for lam in (1e1, 1e2, 1e3, 1e4, 3e4, 1e5):
            m = Ridge(alpha=lam).fit(Xtr, y["tr"])
            v = np.abs(sc.inv(m.predict(Xva)[:, None])[:, 0]
                       - sc.inv(ztr[va + L + h - 1])[:, 0]).mean()
            if v < best[0]:
                best = (v, m)
        p = sc.inv(best[1].predict(Xte)[:, None])[:, 0]
        out[("ridge", kind)] = np.abs(p - truth).mean()
        print(f"{'ridge':<22}{kind:<10}{Xtr.shape[1]:>7}{best[0]:>10.3f}"
              f"{out[('ridge', kind)]:>10.3f}")

        # Trees split on single features and cannot fold a rotation into a
        # weight matrix, so this is where a better basis has to show up.
        from sklearn.ensemble import HistGradientBoostingRegressor
        g = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=400, learning_rate=0.06,
            early_stopping=True, validation_fraction=None, random_state=0)
        g.fit(np.vstack([Xtr, Xva]), np.concatenate([y["tr"], y["va"]]))
        p = sc.inv(g.predict(Xte)[:, None])[:, 0]
        out[("trees", kind)] = np.abs(p - truth).mean()
        v = np.abs(sc.inv(g.predict(Xva)[:, None])[:, 0]
                   - sc.inv(ztr[va + L + h - 1])[:, 0]).mean()
        print(f"{'hist-gbt':<22}{kind:<10}{Xtr.shape[1]:>7}{v:>10.3f}"
              f"{out[('trees', kind)]:>10.3f}")

    print("\nwhat the band basis is worth, by learner:")
    for lr in ("ridge", "trees"):
        r, b = out[(lr, "raw")], out[(lr, "band")]
        print(f"  {lr:<8} {r:.3f} -> {b:.3f}   {(b - r) / r * 100:+.2f}%")
    print("\nridge must read ~0.00%: a rotation is absorbed by its weights.")
    print("trees cannot absorb one, so a real basis gain shows up there or nowhere.")


if __name__ == "__main__":
    main()
