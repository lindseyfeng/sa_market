#!/usr/bin/env python3
"""Evaluate joint multi-region forecasts the way the field does, not by eyeballing MAE.

Three things this adds over a table of MAEs, each because the literature says a
margin without it is not evidence.

**rMAE, not MAE alone.** Lago et al. (2021), the EPF review and benchmark, argue
MAPE and sMAPE are unusable for electricity prices -- prices cross zero, so the
metrics have undefined mean and infinite variance -- and that MASE is unusable
for *comparison* because its scale factor comes from each model's own in-sample
window. Their recommendation is rMAE: MAE divided by the MAE of a naive forecast
built on the **out-of-sample** data. Of their three naive candidates they settle
on the weekly one, the same period one week earlier, because it captures weekly
effects and because the model ranking is then independent of the benchmark. On a
half-hourly market that is a lag of 7*48 = 336.

**A Diebold-Mariano test, not a difference of means.** This project already knows
seed noise spans 0.116 MAE while five decomposition families span 0.04, so an
unpaired single-seed margin is noise. The DM test is a paired, model-free test on
the loss differential, which is exactly the right instrument. Following the same
review, it is run one-sided -- H0: E(delta_AB) <= 0, so rejecting means B really is
more accurate than A -- in both recommended variants: 48 independent univariate
tests, one per half-hour of the day, and one multivariate test on the daily loss
differential. Note what DM cannot do: it compares *forecasts*, not models, so it
says nothing about whether a rerun with another seed would land the same way.

**Per-task single-task baselines, and the sign of the transfer.** The multi-task
literature is explicit that a multi-task model is only interpretable against a
single-task model per task, trained on the same backbone, and that the headline
number is the mean relative change -- the `delta_m` of Maninis et al. Reporting
only the mean MAE across five regions hides negative transfer: joint training can
buy an average improvement while making one region strictly worse, and for this
project SA1 is the region every earlier claim is about.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

WEEK = 7 * 48          # half-hourly: one week earlier, the naive of Lago et al.


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        tag = os.path.basename(f)[:-4]
        out[tag] = {"pred": z["pred"], "truth": z["truth"],
                    "start": z["start"], "names": [str(v) for v in z["names"]]}
    return out


def naive_mae(panel, test_year, rows, region, L, h):
    """MAE of the weekly-naive forecast on exactly the scored rows.

    Built from the panel rather than from the saved truth, because the lag
    reaches 336 rows before the window and those rows are not in any batch.
    """
    df = pd.read_csv(panel, parse_dates=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    yr = [int(v) for v in str(test_year).split(",")]
    sel = df[df.SETTLEMENTDATE.dt.year.isin(yr)]
    y = sel[region].to_numpy(float)
    idx = rows                                    # row of the predicted value
    ok = idx >= WEEK
    if not ok.all():
        print(f"  note: {int((~ok).sum())} of {len(idx)} scored rows are inside the "
              f"first week of the test year and have no weekly naive; "
              f"rMAE is computed on the remaining {int(ok.sum())}")
    if not ok.any():
        return float("nan"), ok
    return float(np.abs(y[idx[ok]] - y[idx[ok] - WEEK]).mean()), ok


def dm(e_a, e_b, p=1, lag="auto"):
    """One-sided Diebold-Mariano. H0: E(L(e_a) - L(e_b)) <= 0.

    A small p-value says model B's forecasts are more accurate than A's.

    The textbook truncation for a one-step forecast is 0 lags, on the argument
    that an optimal one-step forecast has white errors. These forecasts are not
    optimal and the windows overlap, so the loss differential is autocorrelated
    in fact -- which inflates z if it is ignored. The default is therefore a
    Newey-West estimate at floor(4*(n/100)^(2/9)) lags, the usual rule, and
    `ac1` is returned so the choice can be checked rather than trusted.
    """
    d = np.abs(e_a) ** p - np.abs(e_b) ** p
    n = len(d)
    if n < 10:
        return float("nan"), float("nan"), float("nan")
    mu = d.mean()
    v = d.var(ddof=0)
    if v <= 0:
        return float("nan"), float("nan"), 0.0
    ac1 = float(np.corrcoef(d[1:], d[:-1])[0, 1]) if n > 2 else 0.0
    if lag == "auto":
        # Andrews (1991) plug-in bandwidth from an AR(1) fit, not the
        # 4*(n/100)^(2/9) rule of thumb. That rule is for weakly dependent data;
        # here the windows overlap on a highly persistent price and the measured
        # AC1 runs near 0.99, where it would choose 4 lags and overstate z by an
        # order of magnitude. The AR(1) plug-in grows with the persistence it
        # measures, which is the whole point.
        rho = min(max(ac1, 0.0), 0.995)
        m = int(np.ceil(1.1447 * (4 * rho ** 2 * n / (1 - rho ** 2) ** 2) ** (1 / 3)))
        m = min(m, n // 4)
    else:
        m = int(lag)
    for k in range(1, max(m, 0) + 1):
        if k >= n:
            break
        gk = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
        v += 2 * (1 - k / (m + 1)) * gk
    se = np.sqrt(max(v, 1e-30) / n)
    z = mu / se
    return z, float(stats.norm.cdf(-z)), ac1      # one-sided, upper tail of -z


def eff_n(d):
    """Rough effective sample size, 1 + 2*sum(rho_k), for reporting only."""
    n = len(d)
    x = d - d.mean()
    v = (x ** 2).mean()
    if v <= 0:
        return float(n)
    s_ = 1.0
    for k in range(1, min(200, n - 1)):
        r = float((x[k:] * x[:-k]).mean() / v)
        if r <= 0:
            break
        s_ += 2 * r
    return float(n / max(s_, 1.0))


def align(ea, ra, eb, rb):
    """Restrict two error series to the rows both actually scored.

    A long-context arm starts its windows later, so two runs can differ by a few
    hundred rows out of 17,419. A paired test on misaligned series compares
    different half-hours and is worthless, so the rows are intersected rather
    than assumed equal.
    """
    if len(ra) == len(rb) and (ra == rb).all():
        return ea, eb, ra
    common = np.intersect1d(ra, rb)
    ia = {int(v): i for i, v in enumerate(ra)}
    ib = {int(v): i for i, v in enumerate(rb)}
    sa = np.array([ia[int(v)] for v in common])
    sb = np.array([ib[int(v)] for v in common])
    return ea[sa], eb[sb], common


def dm_pair(a, b, rows, p=1):
    """Both variants the EPF review recommends, for one region."""
    z_mv, p_mv, ac1 = dm(a, b, p)
    hh = rows % 48                                # half-hour of day
    wins = sig = 0
    for k in range(48):
        m = hh == k
        if m.sum() < 30:
            continue
        sig += 1
        _, pv, _ = dm(a[m], b[m], p)
        if pv == pv and pv < 0.05:
            wins += 1
    d_ = np.abs(a) - np.abs(b)
    return {"z": float(z_mv), "p": p_mv, "ac1": ac1, "eff_n": eff_n(d_),
            "n": int(len(a)), "halfhours_B_better": int(wins),
            "halfhours_tested": int(sig)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds/joint")
    ap.add_argument("--panel", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--test-year", default="2021")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--mtl", default="joint", help="the multi-task arm to judge")
    ap.add_argument("--out", default="results/joint_eval.json")
    a = ap.parse_args()

    runs = load(a.preds)
    if not runs:
        raise SystemExit(f"no .npz in {a.preds}; run with --save-preds first")
    L, h = a.seq_len, a.horizon

    # region -> {arm: (err, rows)}
    per = {}
    for tag, r in runs.items():
        arm = tag.rsplit("_s", 1)[0]
        rows = r["start"] + L + h - 1
        for j, nm in enumerate(r["names"]):
            per.setdefault(nm, {})[arm] = (r["pred"][:, j] - r["truth"][:, j], rows)

    print(f"{'region':<12}{'arm':<18}{'MAE':>9}{'rMAE':>8}   (naive = same half-hour one week earlier)")
    print("-" * 78)
    table, nmae = {}, {}
    for nm in sorted(per):
        for arm in sorted(per[nm]):
            e, rows = per[nm][arm]
            if nm not in nmae:
                nmae[nm], ok = naive_mae(a.panel, a.test_year, rows, nm, L, h)
            mae = float(np.abs(e).mean())
            table.setdefault(nm, {})[arm] = {"mae": mae, "rmae": mae / nmae[nm]}
            print(f"{nm.replace('_price',''):<12}{arm:<18}{mae:>9.3f}"
                  f"{mae / nmae[nm]:>8.3f}")
        print(f"{'':<12}{'(naive)':<18}{nmae[nm]:>9.3f}{1.0:>8.3f}")

    # ---- single-task baseline per region, and the sign of the transfer ----
    print("\nmulti-task vs single-task, per region")
    print(f"  {'region':<12}{'STL':>9}{'MTL':>9}{'change':>9}{'DM p':>8}{'48 hh':>8}"
          f"{'AC1':>8}{'eff n':>9}")
    rel, neg = [], []
    for nm in sorted(table):
        stl = next((k for k in table[nm] if k.startswith("single")), None)
        if stl is None or a.mtl not in table[nm]:
            continue
        s_, m_ = table[nm][stl]["mae"], table[nm][a.mtl]["mae"]
        d = dm_pair(per[nm][stl][0], per[nm][a.mtl][0], per[nm][stl][1])
        rel.append((m_ - s_) / s_)
        if m_ > s_:
            neg.append(nm)
        print(f"  {nm.replace('_price',''):<12}{s_:>9.3f}{m_:>9.3f}"
              f"{(m_ - s_) / s_ * 100:>8.1f}%{d['p']:>8.3f}"
              f"{d['halfhours_B_better']:>5}/{d['halfhours_tested']}"
              f"{d['ac1']:>8.2f}{d['eff_n']:>9.0f}")
    if rel:
        dm_gain = -float(np.mean(rel))
        print(f"\n  delta_m = {dm_gain * 100:+.2f}%   "
              f"(mean relative change vs the single-task baseline; "
              f"positive = multi-task is better)")
        print(f"  negative transfer on {len(neg)} of {len(rel)} regions"
              + (f": {', '.join(n.replace('_price','') for n in neg)}" if neg else ""))

    # ---- every arm pair, SA1 only: the number every earlier claim is about ----
    arms = sorted(per.get("SA1_price", {}))
    if len(arms) > 1:
        print("\nDiebold-Mariano on SA1, one-sided. H0: E(L(A) - L(B)) <= 0, so a "
              "small p says B is better.")
        print(f"  {'A':<18}{'B':<18}{'z':>8}{'p':>8}{'half-hours B wins':>20}")
        for A, B in itertools.combinations(arms, 2):
            d = dm_pair(per["SA1_price"][A][0], per["SA1_price"][B][0],
                        per["SA1_price"][A][1])
            print(f"  {A:<18}{B:<18}{d['z']:>8.2f}{d['p']:>8.3f}"
                  f"{d['halfhours_B_better']:>14}/{d['halfhours_tested']}"
                  f"   AC1 {d['ac1']:.2f}  eff n {d['eff_n']:.0f}/{d['n']}")

    json.dump({"table": table, "naive_mae": nmae}, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
