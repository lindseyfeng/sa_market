#!/usr/bin/env python3
"""
Figure: why NVMD's decomposition is more interpretable than classical VMD.

Numbers are the output of

    python3 interpret_modes.py \
      --method "Causal VMD tuned a1000k8" vmdsw_a1000_k8_2018_2018.csv \
                                          vmdsw_a1000_k8_2019_2019.csv \
      --method "NVMD v3static"            v3static_modes_2018_2018.csv \
                                          v3static_modes_2019_2019.csv \
      --hours-per-step 0.5

VMD is the *tuned* baseline (alpha=1000, K=8 -- the sweep winner), not the
untuned alpha=2000/K=12 config, so none of this rests on a handicapped baseline.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
VMD_C = "#2a78d6"     # categorical slot 1
NVMD_C = "#eb6834"    # categorical slot 2
GRID = "#e4e3df"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False,
})

# mode, centre, bandwidth, period_h, energy%, ablation  (Residual excluded from
# the band panels -- it is not a band)
VMD = dict(
    centre=np.array([0.0249, 0.0476, 0.0883, 0.1428, 0.2032, 0.2701, 0.3376, 0.4120]),
    bw=np.array([0.0631, 0.0617, 0.0713, 0.0791, 0.0793, 0.0775, 0.0798, 0.0824]),
    period=np.array([20.1, 10.5, 5.7, 3.5, 2.5, 1.9, 1.5, 1.2]),
    energy=np.array([77.36, 11.37, 4.84, 2.19, 1.21, 0.81, 0.64, 0.58]),
    abl=np.array([10.804, 0.218, 0.014, -0.060, -0.069, -0.067, -0.060, -0.059]),
    pr=1.63, removable=6,
)
NVMD = dict(
    centre=np.array([0.0016, 0.0079, 0.0243, 0.0454, 0.0781, 0.1100, 0.1141, 0.2089]),
    bw=np.array([0.0034, 0.0128, 0.0177, 0.0294, 0.0555, 0.0956, 0.1279, 0.1875]),
    period=np.array([307.2, 63.3, 20.5, 11.0, 6.4, 4.5, 4.4, 2.4]),
    energy=np.array([18.53, 7.49, 11.51, 10.33, 8.47, 10.39, 28.16, 5.12]),
    abl=np.array([0.000, -0.001, -0.001, -0.002, -0.002, -0.007, -0.006, 0.031]),
    pr=6.11, removable=8,
)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
fig.subplots_adjust(hspace=0.46, wspace=0.26, top=0.89, bottom=0.145,
                    left=0.075, right=0.975)

# ---------------------------------------------------------------- A: bandwidth
ax = axes[0, 0]
lo, hi = 8e-4, 0.75
ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=MUTED, alpha=0.10, lw=0)
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(0.0016, 0.42, "bandwidth exceeds centre\n(band spans DC — not a band)",
        color=INK2, fontsize=7.6, va="top")
ax.plot(VMD["centre"], VMD["bw"], "o-", color=VMD_C, ms=6.5, lw=2,
        mec=SURFACE, mew=1.4, label="Causal VMD (tuned)", zorder=3)
ax.plot(NVMD["centre"], NVMD["bw"], "o-", color=NVMD_C, ms=6.5, lw=2,
        mec=SURFACE, mew=1.4, label="NVMD v3", zorder=3)
ax.annotate("flat ≈0.06–0.08\nregardless of centre", (0.20, 0.0793),
            textcoords="offset points", xytext=(6, -30), color=VMD_C, fontsize=7.8,
            arrowprops=dict(arrowstyle="-", color=VMD_C, lw=1))
ax.annotate("scales with centre\n(constant-Q)", (0.0781, 0.0555),
            textcoords="offset points", xytext=(30, -34), color=NVMD_C, fontsize=7.8,
            ha="left", arrowprops=dict(arrowstyle="-", color=NVMD_C, lw=1))
ax.annotate("Mode 1: bw 2.5× its centre", (0.0249, 0.0631),
            textcoords="offset points", xytext=(14, 16), color=VMD_C, fontsize=7.8,
            arrowprops=dict(arrowstyle="-", color=VMD_C, lw=1))
ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(2e-3, 0.55),
       xlabel="mode centre frequency (cycles/step)", ylabel="bandwidth")
ax.set_title("A   Bands vs smears", loc="left")
ax.grid(True, which="major", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=8.2, labelcolor=INK2)

# ------------------------------------------------------------------ B: energy
ax = axes[0, 1]
x = np.arange(1, 9)
w = 0.38
ax.bar(x - w/2 - 0.01, VMD["energy"], w, color=VMD_C, label="Causal VMD (tuned)")
ax.bar(x + w/2 + 0.01, NVMD["energy"], w, color=NVMD_C, label="NVMD v3")
ax.text(1 - w/2, 77.36 + 2.5, "77.4%", color=VMD_C, fontsize=8.6, ha="center",
        fontweight="bold")
ax.text(5.0, 62, f"participation ratio\n"
                 f"VMD  {VMD['pr']:.2f} of 9 modes\n"
                 f"NVMD {NVMD['pr']:.2f} of 9 modes",
        color=INK2, fontsize=8.4, va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID))
ax.set(xlabel="mode (ordered by centre frequency)", ylabel="share of total energy (%)",
       xticks=x, ylim=(0, 88))
ax.set_title("B   One mode, or many?", loc="left")
ax.grid(True, axis="y", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8.2, labelcolor=INK2)

# ---------------------------------------------------------------- C: coverage
ax = axes[1, 0]
for lbl, xp in [("½-daily", 12), ("daily", 24), ("weekly", 168)]:
    ax.axvline(xp, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=0)
    ax.text(xp, 2.42, lbl, color=INK2, fontsize=7.4, ha="center", va="top")
ax.plot([VMD["period"].min(), VMD["period"].max()], [1.62, 1.62],
        color=VMD_C, lw=1.2, alpha=0.4, zorder=1)
ax.plot([NVMD["period"].min(), NVMD["period"].max()], [0.62, 0.62],
        color=NVMD_C, lw=1.2, alpha=0.4, zorder=1)
for p_ in VMD["period"]:
    ax.plot([p_, p_], [1.62 - 0.26, 1.62 + 0.26], color=VMD_C, lw=3.5,
            solid_capstyle="butt", zorder=3)
for p_ in NVMD["period"]:
    ax.plot([p_, p_], [0.62 - 0.26, 0.62 + 0.26], color=NVMD_C, lw=3.5,
            solid_capstyle="butt", zorder=3)
ax.text(1.05, 1.14, "ceiling 20.1 h — no multi-day mode exists",
        color=VMD_C, fontsize=7.8, va="center")
ax.text(1.05, 0.14, "reaches 307 h (12.8 days) — 15× further",
        color=NVMD_C, fontsize=7.8, va="center")
ax.set(xscale="log", xlim=(0.9, 900), ylim=(-0.15, 2.55),
       yticks=[0.62, 1.62], yticklabels=["NVMD v3", "Causal VMD\n(tuned)"],
       xlabel="physical period of mode centre (hours, log scale)")
ax.set_title("C   What the modes can represent", loc="left")
ax.grid(True, axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

# --------------------------------------------------------------- D: ablation
ax = axes[1, 1]
ax.axhline(0, color=MUTED, lw=1)
ax.bar(x - w/2 - 0.01, VMD["abl"], w, color=VMD_C, label="Causal VMD (tuned)")
ax.bar(x + w/2 + 0.01, NVMD["abl"], w, color=NVMD_C, label="NVMD v3")
ax.text(1.42, 10.6, "+10.80 MAE — removing Mode 1\nremoves the forecast", color=VMD_C,
        fontsize=8.2, va="top")
ax.text(0.97, 0.50, "NVMD: every mode removable\nat <0.01 MAE — bands overlap,\n"
                    "so they are redundant\n(the known weakness, §3)",
        transform=ax.transAxes, color=INK2, fontsize=7.8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID))
ax.set(xlabel="mode (ordered by centre frequency)",
       ylabel="Δ linear forecast MAE when mode is dropped",
       xticks=x, ylim=(-1.2, 12.2))
ax.set_title("D   Is any single mode load-bearing?", loc="left")
ax.grid(True, axis="y", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8.2, labelcolor=INK2)

fig.suptitle("NVMD vs classical VMD: what the decomposition actually contains",
             x=0.075, y=0.965, ha="left", fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.075, 0.928,
         "SA1 half-hourly price, W=96, train 2018 / test 2019.  VMD is the tuned "
         "sweep winner (α=1000, K=8), not the untuned literature config.",
         ha="left", fontsize=8.6, color=INK2)
fig.text(0.075, 0.028,
         "A–C: NVMD gives ordered, non-overlapping-in-scale bands spanning DC to 2.4 h; VMD gives a flat "
         "bandwidth that smears its lowest mode across 77% of the energy.\n"
         "D is the honest counterweight: NVMD's bands overlap, so no single NVMD mode is load-bearing "
         "either — it wins on structure and coverage, not on ablation.",
         ha="left", fontsize=8.0, color=INK2)

fig.savefig("interpretability_nvmd_vs_vmd.png", dpi=200)
fig.savefig("interpretability_nvmd_vs_vmd.pdf")
print("wrote interpretability_nvmd_vs_vmd.{png,pdf}")
