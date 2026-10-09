#!/usr/bin/env python3
"""The MAE-RMSE frontier across Jacobian exponents, against the neural control.

The reference is `panel_lstm`: the same LSTM, the same head, the same window
and the same objective, reading the raw 37 channels instead of bands. It is the
only comparison in this report where exactly one thing changes, which is why it
and not a linear baseline is what the alpha sweep is scored against.

    python report/plot_pareto.py --out assets/pareto_frontier.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = [(0.0, "preds/pace_h6/joint_s1.npz"), (0.25, "preds/jac/a0.25_s1.npz"),
        (0.5, "preds/jac/a0.5_s1.npz"), (0.75, "preds/jac/a0.75_s1.npz"),
        (1.0, "preds/jac/a1.0_s1.npz")]
REF = "preds/pace_h6/panel_lstm_s1.npz"
SEGS = [("Negative", lambda y: y < 0, "#1d4ed8"),
        ("Calm", lambda y: (y >= 0) & (y < 100), "#047857"),
        ("High", lambda y: (y >= 100) & (y < 300), "#b45309"),
        ("Spike", lambda y: y >= 300, "#6d28d9")]


def ld(f):
    z = np.load(f, allow_pickle=True)
    nm = [str(v) for v in z["names"]]
    i = nm.index("SA1_price") if "SA1_price" in nm else 0
    return z["pred"][:, i], z["truth"][:, i], z["start"]


def scored(f, ref):
    pr, yr, sr = ref
    p, y, st = ld(f)
    common = np.intersect1d(st, sr)
    a, b = np.isin(st, common), np.isin(sr, common)
    return p[a], y[a], pr[b], yr[b]


def nondominated(pts):
    keep = []
    for i, (x, y) in enumerate(pts):
        if not any(u <= x and v <= y and (u < x or v < y)
                   for j, (u, v) in enumerate(pts) if j != i):
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/pareto_frontier.png")
    a = ap.parse_args()
    ref = ld(REF)

    P, R, SEG = [], [], []
    for al, f in ARMS:
        p, y, pr, yr = scored(f, ref)
        P.append(np.abs(p - y).mean())
        R.append(np.sqrt(((p - y) ** 2).mean()))
        SEG.append([100 * (np.abs(p - y)[m(y)].mean()
                           - np.abs(pr - yr)[m(y)].mean())
                    / np.abs(pr - yr)[m(y)].mean() for _, m, _ in SEGS])
    p0, y0, pr0, yr0 = scored(REF, ref)
    ctl = (np.abs(p0 - y0).mean(), np.sqrt(((p0 - y0) ** 2).mean()))
    alphas = [al for al, _ in ARMS]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=190)

    pts = list(zip(P, R))
    front = nondominated(pts)
    order = sorted(front, key=lambda i: P[i])
    ax.plot([P[i] for i in order], [R[i] for i in order], "-", color="#0f766e",
            lw=2.0, zorder=2, label="Nondominated α variants")
    dom = [i for i in range(len(pts)) if i not in front]
    for i in dom:
        near = min(front, key=lambda j: abs(P[j] - P[i]) + abs(R[j] - R[i]))
        ax.plot([P[i], P[near]], [R[i], R[near]], ":", color="#a1a1aa", lw=1.2,
                zorder=1)
    cmap = {0.0: "#1d4ed8", 0.25: "#0f766e", 0.5: "#4d7c0f", 0.75: "#6d28d9",
            1.0: "#b45309"}
    for i, al in enumerate(alphas):
        ax.plot(P[i], R[i], "o", ms=10, color=cmap[al], zorder=4,
                mec="white", mew=1.2)
        off = {0.0: (10, 6), 0.25: (10, 6), 0.5: (10, 5), 0.75: (11, -5),
               1.0: (11, -5)}[al]
        ax.annotate(f"α = {al:g}", (P[i], R[i]), textcoords="offset points",
                    xytext=off, fontsize=9.6, color=cmap[al],
                    fontweight="bold", va="center")
    ax.plot(*ctl, marker="*", ms=19, color="#18181b", zorder=5, mec="white",
            mew=1.0, label="panel_lstm (raw-window control)", ls="none")
    ax.annotate("panel_lstm\n(no decomposition)", ctl,
                textcoords="offset points", xytext=(-12, -4), ha="right",
                fontsize=9.2, color="#18181b", fontweight="bold",
                linespacing=1.3)
    ax.set_xlabel("MAE ($/MWh)  ↓", fontsize=10)
    ax.set_ylabel("RMSE ($/MWh)  ↓", fontsize=10)
    ax.set_title("A. MAE–RMSE trade-off", fontsize=11, loc="left", pad=8)
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    lo, hi = min(P + [ctl[0]]), max(P + [ctl[0]])
    ax.set_xlim(lo - 0.6, hi + 0.85)
    rlo, rhi = min(R + [ctl[1]]), max(R + [ctl[1]])
    ax.set_ylim(rlo - 1.3, rhi + 1.7)
    ax.legend(loc="lower right", fontsize=8.6, framealpha=0.94)
    ax.text(0.015, 0.965, "Lower-left is better. Lines connect measured points "
            "only.", transform=ax.transAxes, fontsize=8, color="#64748b",
            va="top")

    for j, (nm, _, col) in enumerate(SEGS):
        bx.plot(alphas, [s[j] for s in SEG], "o-", color=col, lw=2.0, ms=7,
                label=nm, mec="white", mew=1.0)
    bx.axhline(0, color="#18181b", ls="--", lw=1.1)
    bx.text(1.0, 0.6, "panel_lstm", fontsize=8.4, color="#18181b", ha="right",
            va="bottom")
    bx.set_xlabel("Jacobian exponent α", fontsize=10)
    bx.set_ylabel("Regime MAE change vs panel_lstm (%)  ↓", fontsize=10)
    bx.set_title("B. Where the error moves", fontsize=11, loc="left", pad=8)
    bx.set_xticks(alphas)
    bx.grid(alpha=0.22, lw=0.7)
    bx.set_axisbelow(True)
    bx.legend(ncol=2, fontsize=9, framealpha=0.94)

    fig.suptitle("Increasing tail weight trades typical-price accuracy for "
                 "tail accuracy", fontsize=13.2, fontweight="bold",
                 color="#1e293b", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print("->", a.out)
    for al, p_, r_, s in zip(alphas, P, R, SEG):
        print(f"  a={al:<5g} MAE {p_:7.3f}  RMSE {r_:7.2f}  "
              + "  ".join(f"{n}{v:+6.1f}%" for (n, _, _), v in zip(SEGS, s)))
    print(f"  control  MAE {ctl[0]:7.3f}  RMSE {ctl[1]:7.2f}")
    print("  frontier:", [alphas[i] for i in order])


if __name__ == "__main__":
    main()
