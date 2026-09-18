#!/usr/bin/env python3
"""Overlay the two filter banks on one frequency axis, against the price spectrum.

The point of the figure: VMD's bands and the learned bank's bands are drawn
from the same two degrees of freedom -- a centre and a width per mode -- but
they land in completely different places, and the difference is visible without
any metric.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.nvmd_v3 import StructuredSpectralNVMD
import torch

W, K = 96, 8
# measured over 1500 consecutive 2018 windows, K=8, alpha=1000, W=96
VMD_C = np.array([0.0001, 0.0268, 0.0862, 0.1567, 0.2292, 0.3026, 0.3776, 0.4519])
VMD_B = np.full(K, 0.0631)          # VMD's bandwidth is flat in centre
VMD_SD = np.array([0.0001, 0.0126, 0.0291, 0.0310, 0.0321, 0.0324, 0.0257, 0.0221])

bank = StructuredSpectralNVMD(K=K, signal_len=W, adapt=0.0)
c, b = bank.bands()
BANK_C, BANK_B = c[0].detach().numpy(), b[0].detach().numpy()

f = np.fft.rfftfreq(W)
fine = np.linspace(0, 0.5, 2000)


def masks(centres, widths):
    g = np.exp(-0.5 * ((fine[None, :] - centres[:, None]) / widths[:, None]) ** 2)
    return g / g.sum(0, keepdims=True).clip(1e-8)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.2),
                       gridspec_kw={"width_ratios": [1, 1, 0.9]})

for a, (C, B, title, col) in zip(ax[:2], [
        (VMD_C, VMD_B, "causal VMD, K=8, alpha=1000", "#c0504d"),
        (BANK_C, BANK_B, "fixed bank, geometric, constant-Q", "#4f81bd")]):
    M = masks(C, B)
    for k in range(K):
        a.fill_between(fine, 0, M[k], alpha=0.30, color=col, lw=0)
        a.plot(fine, M[k], color=col, lw=1.1)
        a.axvline(C[k], color=col, lw=0.5, ls=":", alpha=0.6)
    a.set_title(title, fontsize=10)
    a.set_xlabel("normalised frequency")
    a.set_xlim(0, 0.5); a.set_ylim(0, 1.05)
    a.set_xscale("symlog", linthresh=0.01)
ax[0].set_ylabel("mask weight")

# VMD's centres also move window to window; the bank's do not
for k in range(K):
    ax[0].errorbar(VMD_C[k], 1.0, xerr=VMD_SD[k], fmt="|", color="black",
                   ms=5, lw=0.9, capsize=2)
ax[0].text(0.011, 1.02, "bars: centre drift across windows (1 s.d.)",
           fontsize=7, color="black")

df = pd.read_csv("data/raw/compound_2018_2022.csv", parse_dates=["SETTLEMENTDATE"])
sig = df[df.SETTLEMENTDATE.dt.year == 2018]["SA1_price"].to_numpy(float)
P = np.zeros(len(f))
n = 0
for t in range(W, W + 3000):
    P += np.abs(np.fft.rfft(sig[t - W:t])) ** 2
    n += 1
P /= n
# the DC bin is the window mean and dwarfs everything; the shape of interest
# starts at the first non-zero frequency
ax[2].plot(f[1:], P[1:] / P[1:].max(), color="0.35", lw=1.1)
ax[2].set_yscale("log")
ax[2].set_xscale("symlog", linthresh=0.01)
ax[2].set_title("price power spectrum, 2018 (normalised)", fontsize=10)
ax[2].set_xlabel("normalised frequency"); ax[2].set_ylabel("relative power")
ax[2].set_xlim(0, 0.5); ax[2].set_ylim(1e-4, 2)
for k in range(K):
    ax[2].axvline(BANK_C[k], color="#4f81bd", lw=0.8, alpha=0.8)
    ax[2].axvline(VMD_C[k], color="#c0504d", lw=0.8, ls="--", alpha=0.8)
frac = P[1:][f[1:] < 0.08].sum() / P[1:].sum()
ax[2].text(0.0012, 3e-4,
           f"{frac:.0%} of the power sits below f = 0.08\n"
           f"bank puts 5 of 8 bands there; VMD puts 3\n"
           "blue: bank centres   dashed red: VMD centres", fontsize=7)

for a in ax:
    a.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig("figures/band_comparison.png", dpi=160)
print("figures/band_comparison.png written")

print(f"\n{'mode':>5}{'VMD centre':>12}{'VMD width':>11}{'Q':>7}"
      f"{'bank centre':>13}{'bank width':>12}{'Q':>7}")
for k in range(K):
    qv = VMD_C[k] / VMD_B[k] if VMD_C[k] > 1e-6 else 0
    qb = BANK_C[k] / BANK_B[k] if BANK_C[k] > 1e-6 else 0
    print(f"{k+1:>5}{VMD_C[k]:>12.4f}{VMD_B[k]:>11.4f}{qv:>7.2f}"
          f"{BANK_C[k]:>13.4f}{BANK_B[k]:>12.4f}{qb:>7.2f}")
