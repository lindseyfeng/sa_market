#!/usr/bin/env python3
r"""MAE and RMSE as the objective moves from L1 to MSE.

The point of the figure: sliding the objective toward MSE costs MAE
monotonically and buys nothing on RMSE -- so the tail was already learned, and
further tail weighting only takes from the bulk.  A narrow arm, which has no
spare capacity to reallocate, barely moves.
"""
import json, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, GOLD, GREY = "#12314F", "#0F6A72", "#B5761A", "#5A6472"

def agg(paths):
    rows = []
    for p in paths:
        try: rows += json.load(open(p))
        except FileNotFoundError: pass
    if not rows: return None
    return (np.mean([r["test_mae"] for r in rows]),
            np.mean([r["test_rmse"] for r in rows]), len(rows))

# x is "how MSE-like": L1 = 0, MSE = 1, Huber placed by its beta on a log scale
SPEC = [("L1",        0.00, ["results/beta_l10.json"]),
        (r"$\beta$=0.5", 0.30, ["results/beta_huber0p5.json"]),
        (r"$\beta$=1",   0.45, ["results/huber_nvmd_st_s1.json",
                                "results/huber_nvmd_st_s2.json"]),
        (r"$\beta$=2",   0.62, ["results/beta_huber2p0.json"]),
        (r"$\beta$=4",   0.78, ["results/beta_huber4p0.json"]),
        ("MSE",       1.00, None)]

x, lab, mae, rmse, ns = [], [], [], [], []
for name, pos, paths in SPEC:
    if paths is None:                                   # MSE comes from the four-arm run
        d = [r for r in json.load(open("results/three_arms_results.json"))
             if r["arm"] == "nvmd_st"]
        v = (np.mean([r["test_mae"] for r in d]),
             np.mean([r["test_rmse"] for r in d]), len(d))
    else:
        v = agg(paths)
    if v is None: continue
    x.append(pos); lab.append(name); mae.append(v[0]); rmse.append(v[1]); ns.append(v[2])

# the narrow baseline, for contrast
base = {}
b_mse = [r for r in json.load(open("results/stability_results.json"))
         if r["arm"] == "vmd_price_res"]
base["MSE"] = (np.mean([r["test_mae"] for r in b_mse]),
               np.mean([r["test_rmse"] for r in b_mse]), len(b_mse))
bh = agg(sorted(glob.glob("results/base_huber_s*.json")))
if bh: base[r"$\beta$=1"] = bh
bl = agg(sorted(glob.glob("results/base_l1_s*.json")))
if bl: base["L1"] = bl

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.5), sharex=True)

for j, (vals, name, lo_is_good) in enumerate(
        [(mae, "test MAE", True), (rmse, "test RMSE", True)]):
    a = ax[j]
    a.plot(x, vals, "o-", color=NAVY, lw=2.0, ms=7, zorder=3,
           label=r"$\mathtt{nvmd\_st}$  (spatial, $2K$ head inputs)")
    for xi, v, n in zip(x, vals, ns):
        a.annotate(f"{v:.3f}" + ("" if n > 1 else "*"), (xi, v),
                   textcoords="offset points", xytext=(0, 9 if j == 0 else -16),
                   ha="center", fontsize=8, color=NAVY)
    # baseline points where we have them
    bx, bv = [], []
    for name_, pos, _ in SPEC:
        if name_ in base:
            bx.append(pos); bv.append(base[name_][j])
    if bx:
        a.plot(bx, bv, "s--", color=GOLD, lw=1.6, ms=6, zorder=3,
               label=r"$\mathtt{vmd\_price\_res}$  (baseline, $K{+}1$ inputs)")
        for xi, v in zip(bx, bv):
            a.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                       xytext=(0, -16 if j == 0 else 9), ha="center",
                       fontsize=8, color=GOLD)
    a.set_xticks(x); a.set_xticklabels(lab)
    a.set_xlabel("objective:  L1  " + r"$\longrightarrow$" + "  MSE", fontsize=9.5)
    a.set_ylabel(name, fontsize=10)
    a.grid(alpha=0.25, lw=0.6)
    a.set_axisbelow(True)

ax[0].set_title("MAE: the margin is a function of the objective",
                fontsize=10.5, color=NAVY)
ax[1].set_title("RMSE: a constant gap the objective does not touch",
                fontsize=10.5, color=TEAL)
ax[0].legend(fontsize=8.2, loc="lower right", framealpha=0.95)

# the two arms converge on MAE and never converge on RMSE -- say so on the axes
ax[0].annotate("", xy=(1.0, 14.480), xytext=(1.0, 14.372),
               arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
ax[0].annotate("behind by 0.108", xy=(1.0, 14.43),
               xytext=(-8, 0), textcoords="offset points",
               fontsize=7.6, color=GREY, style="italic", ha="right", va="center")
ax[0].annotate("", xy=(0.0, 13.744), xytext=(0.0, 14.091),
               arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
ax[0].annotate("at L1 it is ahead\nby 0.347", xy=(0.0, 13.92),
               xytext=(14, 0), textcoords="offset points",
               fontsize=7.6, color=GREY, style="italic", va="center")

fig.suptitle("The RMSE margin survives any objective; the MAE margin does not",
             fontsize=12, color=NAVY, y=1.00, weight="bold")
fig.text(0.5, -0.05,
         "Left: the lines cross. Under MSE the spatial arm is 0.108 behind the "
         "baseline; under L1 it is 0.347 ahead. Its slope is 2.6x the baseline's, "
         "which is the capacity argument made visible --\nit has $2K$ head inputs "
         "against $K{+}1$, so the objective has more to reallocate. "
         "Right: a constant ~1.4 RMSE gap, about 5%, that no objective touches. "
         "Points marked * are single-seed; the seed noise is 0.116.",
         ha="center", fontsize=8.2, color=GREY)
fig.tight_layout()
fig.savefig("figures/loss_sweep.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig("figures/loss_sweep.pdf", bbox_inches="tight", facecolor="white")
print("wrote figures/loss_sweep.{png,pdf}")
for n, m, r, k in zip(lab, mae, rmse, ns):
    print(f"  {n:10} MAE {m:7.3f}  RMSE {r:7.3f}  ({k} seed{'s' if k>1 else ''})")
print("  baseline:", {k: f"{v[0]:.3f}/{v[1]:.3f}" for k, v in base.items()})
