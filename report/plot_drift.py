#!/usr/bin/env python3
"""Two things the numbers state and a picture shows faster.

Left  -- where each band's centre sits, window by window. VMD re-solves and the
         centres wander; the bank is written down and does not move.
Right -- what period each band actually covers. VMD's highest-period band tops
         out below a day, so daily and weekly structure has nowhere to sit.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from vmdpy import VMD

from models.nvmd_v3 import StructuredSpectralNVMD

W, K, ALPHA, N = 96, 8, 1000, 300
DT = 0.5                                  # hours per sample

df = pd.read_csv("data/raw/compound_2018_2022.csv", parse_dates=["SETTLEMENTDATE"])
sig = df[df.SETTLEMENTDATE.dt.year == 2018]["SA1_price"].to_numpy(float)


def omega(t):
    c = np.pad(sig[t - W:t], (0, 20), mode="edge")
    _, _, om = VMD(c, ALPHA, 0.0, K, 0, 1, 1e-7)
    return np.sort(om[-1])


OM = np.array(Parallel(n_jobs=6)(delayed(omega)(t) for t in range(W, W + N)))
bank = StructuredSpectralNVMD(K=K, signal_len=W, adapt=0.0)
c, b = bank.bands()
BC, BB = c[0].detach().numpy(), b[0].detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

# ---- left: centre trajectories ------------------------------------------
x = np.arange(N)
for k in range(K):
    ax[0].plot(x, OM[:, k], color="#c0504d", lw=0.8, alpha=0.85)
    ax[0].axhline(BC[k], color="#4f81bd", lw=1.4, alpha=0.9)
ax[0].set_yscale("symlog", linthresh=0.005)
ax[0].set_xlabel(f"window index (consecutive, {N} half-hourly steps)")
ax[0].set_ylabel("band centre, normalised frequency")
ax[0].set_title("where each band sits, window by window", fontsize=10)
ax[0].set_xlim(0, N)
ax[0].plot([], [], color="#c0504d", lw=0.9, label="causal VMD (re-solved each window)")
ax[0].plot([], [], color="#4f81bd", lw=1.6, label="fixed bank (written down once)")
ax[0].legend(fontsize=8, loc="lower right", framealpha=0.9)

# ---- right: where each band is centred, in period -----------------------
# Spans are the wrong picture here: VMD's lowest bands are wider than their own
# centres, so centre - width goes negative and the "span" runs to infinity.
# That reads as coverage when it is the opposite -- a smear. Plot the centres.
RES = 1.0 / W                     # lowest frequency a length-W window resolves


def period(c):
    return np.where(c > 1e-9, DT / np.maximum(c, 1e-12), np.inf)

vper, bper = period(OM.mean(0)), period(BC)
y = np.arange(K)
for k in range(K):
    if np.isfinite(vper[k]):
        ax[1].plot(vper[k], k + 0.17, "o", color="#c0504d", ms=7)
    if np.isfinite(bper[k]):
        ax[1].plot(bper[k], k - 0.17, "o", color="#4f81bd", ms=7)
ax[1].axvspan(W * DT, 1e5, color="0.90", zorder=0)
ax[1].axvline(W * DT, color="0.25", ls="-", lw=1.2)
ax[1].set_xscale("log"); ax[1].set_xlim(0.8, 400)
ax[1].set_yticks(y); ax[1].set_yticklabels([f"band {k+1}" for k in y], fontsize=8)
ax[1].invert_yaxis()
ax[1].axvline(24, color="0.25", ls="--", lw=1.0)
ax[1].text(24 * 1.07, -0.72, "1 day", fontsize=8, color="0.25")
ax[1].text(W * DT * 1.10, -0.72, f"{W*DT:.0f} h = window length",
           fontsize=8, color="0.25")
ax[1].set_xlabel("period the band is centred on (hours, log scale)")
ax[1].set_title("what each band is tuned to", fontsize=10)
ax[1].plot([], [], "o", color="#c0504d", ms=7, label="causal VMD")
ax[1].plot([], [], "o", color="#4f81bd", ms=7, label="fixed bank")
ax[1].legend(fontsize=8, loc="upper left", framealpha=0.95)

vres = vper[vper <= W * DT].max()
bres = bper[bper <= W * DT].max()
ax[1].annotate("", xy=(vres, 6.9), xytext=(bres, 6.9),
               arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.1))
ax[1].text(6.0, 6.45,
           f"slowest resolvable band: VMD {vres:.0f} h, bank {bres:.0f} h.\n"
           f"Only the bank has one slower than a day.",
           fontsize=8, ha="left", color="0.25")
ax[1].text(52, 4.6,
           "shaded: slower than the\nwindow itself, so a trend\nslot rather "
           "than a resolvable\noscillation. VMD puts one\nband here (at "
           "14,700 h, off\nscale); the bank puts two.",
           fontsize=7, color="0.4", va="center")

for a in ax:
    a.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig("figures/drift_and_coverage.png", dpi=160)
print("figures/drift_and_coverage.png written\n")

step = np.abs(np.diff(OM, axis=0))
gap = np.diff(OM.mean(0)).mean()
print(f"VMD centre drift: {step.mean()/gap:.1%} of a band gap per step; "
      f"a centre crosses half a gap in {(step > 0.5*gap).any(1).mean():.1%} of steps")
print(f"bank centre drift: 0 by construction\n")
print(f"{'band':>6}{'VMD centre period (h)':>24}{'bank centre period (h)':>24}")
for k in range(K):
    v = f"{vper[k]:.1f}" if np.isfinite(vper[k]) else "DC"
    b = f"{bper[k]:.1f}" if np.isfinite(bper[k]) else "DC"
    print(f"{k+1:>6}{v:>24}{b:>24}")
print(f"\nwindow spans {W*DT:.0f} h, so nothing slower than that is resolvable")
print(f"slowest resolvable band: VMD {vres:.1f} h, bank {bres:.1f} h")
print(f"resolvable bands slower than the daily cycle: "
      f"VMD {int(((vper>24)&(vper<=W*DT)).sum())}, "
      f"bank {int(((bper>24)&(bper<=W*DT)).sum())}")
