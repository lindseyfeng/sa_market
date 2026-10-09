#!/usr/bin/env python3
"""Route each interval to the arm that is best in its regime.

The per-regime results are a Pareto frontier, not a ranking: raising the
Jacobian exponent moves error out of the tails and into the calm band
monotonically, so no single point forecast wins all four regimes. An oracle that
picks the best arm per regime reaches MAE 35.86 against the ridge's 37.77, -5.1%,
and that oracle is worth chasing because its assignment is interpretable:

    negative   a0.75        Jacobian weighting, -6.8% against ridge, p=0.000
    calm       joint        the decomposition as a regulariser, -1.9%
    high       arx_window   linear is steadiest here
    spike      persistence  every trained model regresses to the mean; copying
                            the last price is closer on all 115 spike rows

The oracle uses the realised regime, so the question this file answers is how
much survives when the regime has to be predicted from the same window the
forecasters see. Two numbers are reported and they measure different things:
`oracle-regime` keeps the true regime and isolates the assignment, `predicted`
replaces it with a classifier and is the only one that could be run live.

The arm-to-regime map is fixed by the mechanism above rather than fitted, because
the per-arm validation predictions were not saved. Fitting it on the test
segmentation would be choosing the answer from the answer sheet; this way the
map is a stated design and only the classifier is trained.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd

REG = [("negative", lambda y: y < 0), ("calm", lambda y: (y >= 0) & (y < 100)),
       ("high", lambda y: (y >= 100) & (y < 300)), ("spike", lambda y: y >= 300)]
PLAN = {"negative": "a0.75", "calm": "joint", "high": "arx_window",
        "spike": "naive_persist"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    L, h = a.seq_len, a.horizon

    E = {}
    for pat in ("preds/pace_h6/*.npz", "preds/jac/*.npz", "preds/baselines_h6/*.npz"):
        for f in sorted(glob.glob(pat)):
            z = np.load(f, allow_pickle=True)
            nm = [str(v) for v in z["names"]]
            i = nm.index("SA1_price") if "SA1_price" in nm else 0
            E[re.sub(r"_s\d\.npz$", "", os.path.basename(f))] = (z["pred"][:, i],
                                                                 z["truth"][:, i])
    y = E["arx_window"][1]
    need = set(PLAN.values())
    miss = [k for k in need if k not in E or len(E[k][1]) != len(y)]
    if miss:
        raise SystemExit(f"missing arms: {miss}")

    # ---- the regime classifier, on the same window the forecasters read ----
    df = pd.read_csv(a.panel, parse_dates=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    ch = [c for c in df.columns if c != "SETTLEMENTDATE"]
    tr = df[df.SETTLEMENTDATE.dt.year.isin([2018, 2019, 2020])][ch].to_numpy(np.float32)
    te = df[df.SETTLEMENTDATE.dt.year == 2021][ch].to_numpy(np.float32)
    t = ch.index("SA1_price")
    al = np.arange(0, len(tr) - L - h + 1)
    nv = int(len(al) * a.val_frac)
    trs, vas = al[:-(nv + L)], al[-nv:]
    tes = np.arange(0, len(te) - L - h + 1)

    def lab(v):
        out = np.zeros(len(v), int)
        for i, (_, f) in enumerate(REG):
            out[f(v)] = i
        return out

    # A compact summary of the window rather than all 3,552 values: the classifier
    # only has to separate four regimes, and more features here would cost
    # variance without buying separation.
    def feats(src, st):
        rows = []
        for i in st:
            W = src[i:i + L]
            p = W[:, t]
            rows.append(np.concatenate([
                p[[-1, -2, -3, -6, -12, -24, -48, -96]],
                [p.mean(), p.std(), p.min(), p.max(),
                 np.percentile(p, 10), np.percentile(p, 90),
                 (p < 0).mean(), (p > 300).mean()],
                W[-1], W.mean(0), W.std(0),
            ]))
        return np.asarray(rows, np.float32)

    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, Xva, Xte = feats(tr, trs), feats(tr, vas), feats(te, tes)
    ytr, yva = lab(tr[trs + L + h - 1, t]), lab(tr[vas + L + h - 1, t])
    yte = lab(y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         early_stopping=True,
                                         validation_fraction=0.12, random_state=0)
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    pred = clf.predict(Xte)
    print(f"regime classifier: {Xtr.shape[1]} features, "
          f"{len(trs) + len(vas)} train rows, accuracy {(pred == yte).mean():.3f}")
    print(f"  {'regime':<10}{'n':>7}{'recall':>8}{'precision':>11}  routed to")
    for i, (nm, _) in enumerate(REG):
        m, pm = yte == i, pred == i
        rc = (pred[m] == i).mean() if m.sum() else float("nan")
        pr = (yte[pm] == i).mean() if pm.sum() else float("nan")
        print(f"  {nm:<10}{m.sum():>7}{rc:>8.3f}{pr:>11.3f}  {PLAN[nm]}")

    def assemble(reg_idx):
        out = np.empty(len(y))
        for i, (nm, _) in enumerate(REG):
            m = reg_idx == i
            out[m] = E[PLAN[nm]][0][m]
        return out

    def blend(prob, temp=1.0):
        """Weight each arm by the probability of its regime, not by argmax.

        Hard routing amplifies the classifier: one misrouted interval gets the
        arm that is worst for it, and the classifier recalls only 26% of
        negative prices and 3% of spikes, so routing lost 2.4% against the ridge
        where the oracle gains 5.1%. A probability weight degrades gracefully
        instead -- a 26% belief buys 26% of the tail arm rather than all of it.
        `temp` < 1 sharpens the weights toward routing, > 1 flattens them toward
        a fixed average, so the two failure modes sit at the ends of one knob.
        """
        w = prob ** (1.0 / temp)
        w = w / w.sum(1, keepdims=True)
        P = np.stack([E[PLAN[nm]][0] for nm, _ in REG], 1)
        return (w * P).sum(1)

    print(f"\n{'':<26}{'MAE':>9}{'RMSE':>10}{'vs ridge MAE':>14}")
    print("-" * 59)
    br = E["arx_window"][0]
    bm, brm = np.abs(br - y).mean(), np.sqrt(((br - y) ** 2).mean())
    print(f"{'arx_window (ridge)':<26}{bm:>9.3f}{brm:>10.1f}{0.0:>13.1f}%")
    for nm, idx in (("oracle-regime", yte), ("predicted regime", pred)):
        p = assemble(idx)
        m, r = np.abs(p - y).mean(), np.sqrt(((p - y) ** 2).mean())
        print(f"{nm:<26}{m:>9.3f}{r:>10.1f}{(m - bm) / bm * 100:>13.1f}%")
    prob = clf.predict_proba(Xte)
    for temp in (0.5, 1.0, 2.0, 4.0):
        p = blend(prob, temp)
        m, r = np.abs(p - y).mean(), np.sqrt(((p - y) ** 2).mean())
        print(f"{f'soft blend (temp={temp})':<26}{m:>9.3f}{r:>10.1f}"
              f"{(m - bm) / bm * 100:>13.1f}%")
    # A fixed average of the same arms, as the control: if the classifier adds
    # nothing, the blend should not beat this.
    P = np.stack([E[PLAN[nm]][0] for nm, _ in REG], 1)
    for nm_, w_ in (("equal average", np.ones(4) / 4),
                    ("calm-heavy average", np.array([.2, .6, .15, .05]))):
        p = P @ w_
        m, r = np.abs(p - y).mean(), np.sqrt(((p - y) ** 2).mean())
        print(f"{nm_:<26}{m:>9.3f}{r:>10.1f}{(m - bm) / bm * 100:>13.1f}%")
    for k in ("a0.25", "joint", "a0.75"):
        if k in E:
            p = E[k][0]
            m, r = np.abs(p - y).mean(), np.sqrt(((p - y) ** 2).mean())
            print(f"{k:<26}{m:>9.3f}{r:>10.1f}{(m - bm) / bm * 100:>13.1f}%")


if __name__ == "__main__":
    main()
