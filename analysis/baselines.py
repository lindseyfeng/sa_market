#!/usr/bin/env python3
"""Established baselines for the joint multi-region forecasts, on identical rows.

Comparing a new model only against one's own single-task version is an internal
ablation, not a comparison. Lago et al.'s first best practice is that a new
method is compared against well-established methods with a statistical test, so
this file supplies the ones the field actually uses, fitted on the same training
rows and scored on the same test rows as the networks -- which is what lets the
Diebold-Mariano test in `eval_joint.py` run on the pair.

    naive_week     the same half-hour one week earlier; the rMAE denominator
    naive_persist  the last observed price; the h=1 benchmark that is hard to beat
    ar_window      ridge on the target's own 96-step window
    arx_window     ridge on the whole 96 x 37 window -- THE control. Same
                   information as the network, same rows, linear function class.
                   A decomposition plus a coupling tensor has to beat this or it
                   is not paying for itself.
    var            VAR on the five regional price windows: the classic joint
                   linear specification in multivariate EPF
    lear           Lasso-estimated AR, per region, on the standard EPF lag set.
                   Lago et al. call it the most accurate linear model in EPF.
    mtlasso        multi-task (group) Lasso across the five regions: one shared
                   support, which is what CING-LEAR does to make LEAR multi-output
    global_linear  one pooled linear model across all five regions, the global
                   method of Montero-Manso and Hyndman
    factor         the cross-region mean and each region's deviation from it,
                   modelled separately. 96.4% of the variance across regions is
                   the common mode, so this is the baseline that asks whether a
                   joint model is doing anything beyond "common factor plus
                   residual" -- the most dangerous one on this panel.

Everything is fitted on asinh-transformed prices, as the EPF toolbox does, and
every metric is taken in $/MWh after inverting.
"""
import argparse
import os

import numpy as np
import pandas as pd

REGIONS = ["SA1_price", "NSW1_price", "VIC1_price", "QLD1_price", "TAS1_price"]
DAY, WEEK = 48, 7 * 48


def gram(make_row, n, p, chunk=2048):
    """Accumulate X'X and X'y in chunks: the full design matrix is 44,535 x 3,552
    and this box is already deep in swap."""
    XtX = np.zeros((p + 1, p + 1))
    Xty = None
    for a in range(0, n, chunk):
        X, y = make_row(a, min(a + chunk, n))
        X = np.hstack([X, np.ones((len(X), 1), np.float32)])
        XtX += X.T.astype(np.float64) @ X.astype(np.float64)
        t = X.T.astype(np.float64) @ y.astype(np.float64)
        Xty = t if Xty is None else Xty + t
    return XtX, Xty


def ridge_fit(XtX, Xty, lam):
    p = XtX.shape[0]
    A = XtX + lam * np.eye(p)
    A[-1, -1] -= lam                       # never penalise the intercept
    return np.linalg.solve(A, Xty)


# Half-decade steps, not decades. A decade grid put the 96-step window's
# optimum between its points: validation prefers 3e4 (test 37.037) but a grid of
# 1e4/1e5 selects 1e5 and lands on 37.772, so the baseline looked 0.7 MAE worse
# than it is for no reason but grid resolution. On the finer grid the
# validation optimum and the test optimum coincide at 3e4 and the selection
# effect is 0.000 -- which is the number any claim of beating this ridge has to
# clear.
LAMS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]


def ridge_select(XtX, Xty, f_val, n_val, y_val, sc, cols_out=None):
    """Pick the ridge penalty on the validation tail, in $/MWh.

    The networks select an epoch on this same tail. Leaving a baseline's one
    hyperparameter at a hand-picked value while the network gets to choose would
    be exactly the asymmetry Lago et al. object to -- and selecting it on the
    test set would be the ex-post overfitting they object to more.
    """
    best, best_W, best_lam = float("inf"), None, None
    for lam in LAMS:
        W = ridge_fit(XtX, Xty, lam)
        P = sc.inv(predict(f_val, n_val, W) if cols_out is None
                   else cols_out(predict(f_val, n_val, W)))
        m = float(np.abs(P - y_val).mean())
        if m < best:
            best, best_W, best_lam = m, W, lam
    return best_W, best_lam, best


def predict(make_row, n, W, chunk=4096):
    out = []
    for a in range(0, n, chunk):
        X, _ = make_row(a, min(a + chunk, n))
        X = np.hstack([X, np.ones((len(X), 1), np.float32)])
        out.append(X.astype(np.float64) @ W)
    return np.vstack(out)


FLOOR, CAP = -1000.0, 15500.0


class Asinh:
    """The variance-stabilising transform the EPF toolbox uses, per column.

    The inverse clamps twice, and both clamps matter. Without the first, a ridge
    whose asinh-space prediction wanders a little comes back as a price of
    272,650 $/MWh -- measured -- and five such rows moved the mean absolute error
    from a median of 38.7 to 98.8. The networks already clamp to the training
    range plus a margin inside `run_joint.py`, so a baseline without it is not
    being compared on equal terms. The second clamp is the market: a 30-minute
    RRP is the mean of six 5-minute dispatch prices, each inside the NEM band, so
    a number outside it is not a price and no model should be credited or
    debited for producing one.
    """

    def __init__(self, y, margin=2.0):
        self.c = np.median(y, 0)
        self.w = (np.percentile(y, 75, 0) - np.percentile(y, 25, 0)) + 1e-8
        z = np.arcsinh((y - self.c) / self.w)
        self.lo, self.hi = z.min(0) - margin, z.max(0) + margin

    def fwd(self, y):
        return np.arcsinh((y - self.c) / self.w)

    def inv(self, z):
        z = np.clip(z, self.lo, self.hi)
        return np.clip(np.sinh(z) * self.w + self.c, FLOOR, CAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--train-year", default="2018,2019,2020")
    ap.add_argument("--test-year", default="2021")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    L, h = a.seq_len, a.horizon
    out_dir = a.out or f"preds/baselines_h{h}"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(a.panel, parse_dates=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
    tgt = [chans.index(r) for r in REGIONS]

    def span(spec):
        yr = [int(v) for v in str(spec).split(",")]
        return df[df.SETTLEMENTDATE.dt.year.isin(yr)][chans].to_numpy(np.float32)

    tr_raw, te_raw = span(a.train_year), span(a.test_year)
    # The same window list and the same validation embargo the networks use, so
    # the scored rows line up and a paired test is meaningful.
    tr_starts_all = np.arange(0, len(tr_raw) - L - h + 1)
    te_starts = np.arange(0, len(te_raw) - L - h + 1)
    n_val = int(len(tr_starts_all) * a.val_frac)
    va_starts = tr_starts_all[-n_val:]
    tr_starts = tr_starts_all[:-(n_val + L)]

    sc = Asinh(tr_raw[:, tgt])
    ztr, zte = sc.fwd(tr_raw[:, tgt]), sc.fwd(te_raw[:, tgt])

    # Inputs get the same variance-stabilising transform as the target, per
    # channel, before standardising. This is not cosmetic. With raw-price
    # features against an asinh target, the best linear map is forced to
    # approximate asinh itself, and it cannot: fitted on the lag-1 price alone
    # the optimal least-squares solution scores an MAE of 29,865 against
    # persistence's 29.5. In asinh space the same one-parameter fit scores 27.2
    # with a coefficient of 0.876 -- it beats persistence immediately, because
    # a coefficient of 1 *is* persistence once the transform is inverted. The
    # networks are unaffected either way, being nonlinear, so leaving the
    # baselines in raw space would have handicapped only them. LEAR applies the
    # transform to inputs and outputs alike for the same reason.
    inp = Asinh(tr_raw)
    tr_v, te_v = inp.fwd(tr_raw), inp.fwd(te_raw)
    mu, sd = tr_v.mean(0), tr_v.std(0) + 1e-8
    tr_n, te_n = (tr_v - mu) / sd, (te_v - mu) / sd

    def rows(raw_n, z, starts, cols, lags=None):
        """-> (X, y) for the given window start indices.

        `cols` selects channels; `lags` selects positions inside the window,
        counted back from its last row. None means the whole window.
        """
        def f(a_, b_):
            s = starts[a_:b_]
            if lags is None:
                X = np.stack([raw_n[i:i + L, cols].ravel() for i in s])
            else:
                X = np.stack([raw_n[i + L - 1 - np.asarray(lags)][:, cols].ravel()
                              for i in s])
            return X.astype(np.float32), z[s + L + h - 1]
        return f

    want = set(a.only.split(",")) if a.only else None
    def run(name):
        return want is None or name in want

    preds, truth = {}, sc.inv(zte[te_starts + L + h - 1])

    def save(name, p, note=""):
        np.savez_compressed(os.path.join(out_dir, f"{name}_s0.npz"),
                            pred=p, truth=truth, start=te_starts,
                            names=np.array(REGIONS))
        mae = np.abs(p - truth).mean(0)
        print(f"  {name:<16}" + "".join(f"{v:>9.3f}" for v in mae)
              + f"{mae.mean():>10.3f}   {note}")

    print(f"h={h}  train {len(tr_starts)} / test {len(te_starts)} windows, "
          f"{len(chans)} channels")
    print(f"  {'baseline':<16}" + "".join(f"{r.replace('_price',''):>9}" for r in REGIONS)
          + f"{'mean':>10}")

    # ---- naive ----
    if run("naive_week"):
        idx = te_starts + L + h - 1
        src = np.where(idx >= WEEK, idx - WEEK, idx)
        save("naive_week", te_raw[src][:, tgt])
    if run("naive_persist"):
        save("naive_persist", te_raw[te_starts + L - 1][:, tgt])

    # ---- ridge families, all on the normal equations ----
    specs = {
        "ar_window":   (tgt, None),
        "arx_window":  (list(range(len(chans))), None),
        "var":         (tgt, None),
    }
    for name, (cols, lags) in specs.items():
        if not run(name):
            continue
        if name == "ar_window":
            # each region on its own window only: a genuinely univariate AR
            P = np.zeros((len(te_starts), len(tgt)))
            lams = []
            for k, c in enumerate(tgt):
                sc_k = Asinh(tr_raw[:, [tgt[k]]])
                f_tr = rows(tr_n, ztr[:, [k]], tr_starts, [c], lags)
                XtX, Xty = gram(f_tr, len(tr_starts), L)
                f_va = rows(tr_n, ztr[:, [k]], va_starts, [c], lags)
                y_va = sc_k.inv(ztr[va_starts + L + h - 1][:, [k]])
                W, lam, _ = ridge_select(XtX, Xty, f_va, len(va_starts), y_va, sc_k)
                lams.append(lam)
                f_te = rows(te_n, zte[:, [k]], te_starts, [c], lags)
                P[:, [k]] = np.clip(predict(f_te, len(te_starts), W),
                                    sc.lo[k], sc.hi[k])
            save(name, sc.inv(P), f"lam {lams}")
            continue
        p = L * len(cols) if lags is None else len(lags) * len(cols)
        f_tr = rows(tr_n, ztr, tr_starts, cols, lags)
        XtX, Xty = gram(f_tr, len(tr_starts), p)
        f_va = rows(tr_n, ztr, va_starts, cols, lags)
        W, lam, _ = ridge_select(XtX, Xty, f_va, len(va_starts),
                                 sc.inv(ztr[va_starts + L + h - 1]), sc)
        f_te = rows(te_n, zte, te_starts, cols, lags)
        save(name, sc.inv(predict(f_te, len(te_starts), W)), f"lam {lam:g}")

    # ---- common factor plus deviation ----
    if run("factor"):
        cols = list(range(len(chans)))
        zc = ztr.mean(1, keepdims=True)
        zd = ztr - zc
        f_tr = rows(tr_n, np.hstack([zc, zd]), tr_starts, cols, None)
        XtX, Xty = gram(f_tr, len(tr_starts), L * len(cols))
        f_va = rows(tr_n, np.hstack([zc, zd]), va_starts, cols, None)
        recombine = lambda Z: Z[:, [0]] + Z[:, 1:]
        W, lam, _ = ridge_select(XtX, Xty, f_va, len(va_starts),
                                 sc.inv(ztr[va_starts + L + h - 1]), sc,
                                 cols_out=recombine)
        f_te = rows(te_n, np.hstack([zte.mean(1, keepdims=True), zte]),
                    te_starts, cols, None)
        save("factor", sc.inv(recombine(predict(f_te, len(te_starts), W))),
             f"lam {lam:g}")

    # ---- pooled across regions: one function for all five ----
    if run("global_linear"):
        P = np.zeros((len(te_starts), len(tgt)))
        XtX = np.zeros((L + 1, L + 1)); Xty = np.zeros((L + 1, 1))
        for k, c in enumerate(tgt):
            f = rows(tr_n, ztr[:, [k]], tr_starts, [c], None)
            A, b = gram(f, len(tr_starts), L)
            XtX += A; Xty += b
        bestm, bestW, bestlam = float("inf"), None, None
        for lam in LAMS:
            W = ridge_fit(XtX, Xty, lam)
            Pv = np.zeros((len(va_starts), len(tgt)))
            for k, c in enumerate(tgt):
                f = rows(tr_n, ztr[:, [k]], va_starts, [c], None)
                Pv[:, [k]] = predict(f, len(va_starts), W)
            m = float(np.abs(sc.inv(Pv) - sc.inv(ztr[va_starts + L + h - 1])).mean())
            if m < bestm:
                bestm, bestW, bestlam = m, W, lam
        for k, c in enumerate(tgt):
            f = rows(te_n, zte[:, [k]], te_starts, [c], None)
            P[:, [k]] = predict(f, len(te_starts), bestW)
        save("global_linear", sc.inv(P), f"lam {bestlam:g}")

    # ---- LEAR and its multi-output form, on the standard EPF lag set ----
    # The literature's LEAR selects lags rather than using a whole window, so a
    # Lasso is tractable; the day and week lags it needs reach past the 96-step
    # window the networks see, which is noted in the write-up as extra
    # information rather than hidden.
    if run("lear") or run("mtlasso"):
        from sklearn.linear_model import MultiTaskLassoCV, LassoCV
        lags = [0, 1, 2, 3, DAY - 1, DAY, 2 * DAY, WEEK - 1, WEEK]
        back = max(lags)
        keep = tr_starts[tr_starts + L - 1 - back >= 0]
        kte = te_starts[te_starts + L - 1 - back >= 0]
        sub = keep[::4]                       # Lasso needs the real matrix; thin it
        cols = list(range(len(chans)))
        f = rows(tr_n, ztr, sub, cols, lags)
        X, y = f(0, len(sub))
        fte = rows(te_n, zte, kte, cols, lags)
        Xe, ye = fte(0, len(kte))
        print(f"  (lasso on {X.shape[0]} x {X.shape[1]}, every 4th train window)")
        if run("mtlasso"):
            m = MultiTaskLassoCV(cv=3, n_alphas=12, max_iter=2000,
                                 n_jobs=1).fit(X, y)
            P = m.predict(Xe)
            np.savez_compressed(os.path.join(out_dir, "mtlasso_s0.npz"),
                                pred=sc.inv(P), truth=sc.inv(ye), start=kte,
                                names=np.array(REGIONS))
            mae = np.abs(sc.inv(P) - sc.inv(ye)).mean(0)
            print(f"  {'mtlasso':<16}" + "".join(f"{v:>9.3f}" for v in mae)
                  + f"{mae.mean():>10.3f}")
        if run("lear"):
            P = np.zeros_like(ye)
            for k in range(len(tgt)):
                P[:, k] = LassoCV(cv=3, n_alphas=12, max_iter=2000,
                                  n_jobs=1).fit(X, y[:, k]).predict(Xe)
            np.savez_compressed(os.path.join(out_dir, "lear_s0.npz"),
                                pred=sc.inv(P), truth=sc.inv(ye), start=kte,
                                names=np.array(REGIONS))
            mae = np.abs(sc.inv(P) - sc.inv(ye)).mean(0)
            print(f"  {'lear':<16}" + "".join(f"{v:>9.3f}" for v in mae)
                  + f"{mae.mean():>10.3f}")

    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
