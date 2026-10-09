#!/usr/bin/env python3
"""Score by price regime, because the global mean is not where the models differ.

On 2021, 115 spike rows -- 0.7% of the test set -- carry 23% of the total
absolute error, and both the network and a ridge sit at ~1,345 $/MWh on them.
Negative prices are 19.6% of rows and another 29% of the error. So a global MAE
comparison is settled mostly on rows where no model has skill, and it hid the
one place the decomposition wins: in the calm band, 68.6% of rows, the joint arm
beats the ridge by 1.8% and beats the no-decomposition control by 4.4%
(DM p=0.004).

Diebold-Mariano is run within each regime. Conditioning the test on the realised
outcome is a choice, not a free lunch: the segments are defined by the truth, so
these are conditional comparisons and cannot be read as an unconditional claim.
"""
import argparse, glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.eval_joint import dm

SEGS = [("negative", lambda y: y < 0),
        ("calm 0-100", lambda y: (y >= 0) & (y < 100)),
        ("high 100-300", lambda y: (y >= 100) & (y < 300)),
        ("spike >=300", lambda y: y >= 300)]


def load(pats):
    out = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            z = np.load(f, allow_pickle=True)
            nm = [str(v) for v in z["names"]]
            i = nm.index("SA1_price") if "SA1_price" in nm else 0
            tag = re.sub(r"_s\d+\.npz$", "", os.path.basename(f))
            out[tag] = (z["pred"][:, i], z["truth"][:, i], z["start"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+",
                    default=["preds/pace_h6/*.npz", "preds/jac/*.npz",
                             "preds/baselines_h6/arx_window_s0.npz"])
    ap.add_argument("--ref", default="arx_window",
                    help="the model every other is compared against")
    a = ap.parse_args()
    E = load(a.preds)
    if a.ref not in E:
        raise SystemExit(f"{a.ref} not among {sorted(E)}")
    y = E[a.ref][1]

    print(f"{'':<26}" + "".join(f"{n:>16}" for n, _ in SEGS) + f"{'global':>10}")
    print(f"{'':<26}" + "".join(f"{f'n={f(y).sum()}':>16}" for _, f in SEGS)
          + f"{len(y):>10}")
    print("-" * (26 + 16 * len(SEGS) + 10))
    for k in sorted(E, key=lambda k: np.abs(E[k][0] - E[k][1]).mean()):
        p, t, _ = E[k]
        if len(t) != len(y):
            continue
        row = f"{k:<26}"
        for _, f in SEGS:
            m = f(y)
            row += f"{np.abs(p[m] - t[m]).mean():>16.2f}"
        print(row + f"{np.abs(p - t).mean():>10.2f}")

    print(f"\nagainst {a.ref}: effect, then DM p (small = the arm is better)")
    print(f"{'arm':<26}" + "".join(f"{n:>16}" for n, _ in SEGS))
    print("-" * (26 + 16 * len(SEGS)))
    er_all = E[a.ref][0] - E[a.ref][1]
    for k in sorted(E):
        if k == a.ref or len(E[k][1]) != len(y):
            continue
        ek_all = E[k][0] - E[k][1]
        row = f"{k:<26}"
        for _, f in SEGS:
            m = f(y)
            er, ek = er_all[m], ek_all[m]
            if m.sum() < 30:
                row += f"{'-':>16}"
                continue
            _, pv, _ = dm(er, ek)
            d = (np.abs(ek).mean() - np.abs(er).mean()) / np.abs(er).mean() * 100
            row += f"{d:>+9.1f}% {pv:>5.3f}"
        print(row)


if __name__ == "__main__":
    main()
