#!/usr/bin/env python3
r"""The band-parameterised filter bank with per-band spatial coupling.

Drawn from models/nvmd_st.py + nvmd_v3.py.  The module names still say NVMD;
CLAUDE.md records why the method should not be called that: it shares no
objective, no algorithm and no optimisation with VMD.

The visual centre is the per-band coupling: standard spatio-temporal models
learn one adjacency A_ij; this learns A_ij^(k), one per frequency band, so
"which driver matters at which timescale" is an explicit, inspectable object.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

NAVY, TEAL, GOLD, GREY = "#12314F", "#0F6A72", "#B5761A", "#5A6472"
FIXED, LEARN, OFF = "#EEF2F6", "#FFF6E5", "#F4F4F4"

fig, ax = plt.subplots(figsize=(13.2, 7.0))
ax.set_xlim(0, 132)
ax.set_ylim(0, 70)
ax.axis("off")


def box(x, y, w, h, t, fc=FIXED, ec=NAVY, fs=8.2, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.5)


def arr(x1, y1, x2, y2, c=GREY, ls="-", lw=1.0, rad=None):
    cs = f"arc3,rad={rad}" if rad else None
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=11, linewidth=lw, color=c, linestyle=ls,
                 zorder=1, connectionstyle=cs))


def note(x, y, t, c=GREY, fs=7.1, ha="center"):
    ax.text(x, y, t, ha=ha, fontsize=fs, color=c, style="italic", linespacing=1.4)


# --------------------------------------------------------------- 1. input
box(2, 59, 27, 7.5, r"compound panel" "\n" r"$x\in\mathbb{R}^{B\times R\times L}$,  R=33, L=96", fc="white")
note(15.5, 56.0, "1 target price  ·  26 exogenous  ·  6 calendar")
arr(15.5, 59, 15.5, 51.5)
note(31, 55.0, r"reshape $(B{\cdot}R,1,L)$ — one shared bank, every channel", ha="left")

# --------------------------------------------------------------- 2. the bank
ax.add_patch(FancyBboxPatch((2, 25.5), 46, 25, boxstyle="round,pad=0.5",
             facecolor="#FAFBFC", edgecolor=NAVY, lw=1.4, ls="--", zorder=0))
ax.text(25, 48.3, "band-parameterised filter bank  ·  shared across all R channels",
        ha="center", fontsize=8.6, color=NAVY, weight="bold")
box(4.5, 39.5, 19, 6.6, r"gap logits $\ell_k$  (K=8)" "\n" r"geometric init, ratio 1.8", fc=LEARN, fs=7.6)
box(26.5, 39.5, 19, 6.6, r"log bandwidth $\log b_k$" "\n" r"floor $0.25\,\mathrm{gap}_k$", fc=LEARN, fs=7.6)
box(4.5, 32.0, 19, 5.8, r"$c=\mathrm{cumsum}(\mathrm{softmax}\,\ell)\times 0.5$", fs=7.6)
box(26.5, 32.0, 19, 5.8, r"$b_k=b^{\min}_k+\mathrm{softplus}(\log b_k)$", fs=7.6)
arr(14, 39.5, 14, 37.8)
arr(36, 39.5, 36, 37.8)
box(4.5, 26.5, 41, 4.4,
    r"Gaussian masks $g_k(f)=\exp[-\frac{1}{2}((f-c_k)/b_k)^2]$,  normalised $g_k/\sum_j g_j$", fs=7.7)
arr(14, 32.0, 20, 30.9)
arr(36, 32.0, 30, 30.9)
ax.text(25, 23.5, r"rFFT $\odot$ masks $\to$ irFFT       $\sum_k u_k = x$ exactly",
        ha="center", fontsize=8.0, color=NAVY)

box(51, 39.5, 16, 6.6, "SignalEncoder\ncausal CNN 15/7/3", fc=OFF, ec="#B9BFC6", fs=7.2)
arr(51, 42.8, 45.5, 42.8, c="#B9BFC6", ls=":")
note(59, 36.5, "adapt = 0.0\nbank is input-independent", c="#B9BFC6")

arr(25, 22.6, 25, 18.0)
note(29.5, 20.2, r"$(B,R,K,L)$", ha="left")

# ------------------------------------------- 3. per-band coupling (the contribution)
ax.add_patch(FancyBboxPatch((2, 3.0), 46, 14.0, boxstyle="round,pad=0.5",
             facecolor="#F2F9F8", edgecolor=TEAL, lw=1.9, zorder=0))
ax.text(25, 15.2, r"PerBandSpatialCoupling  ·  one $R\times R$ matrix per band",
        ha="center", fontsize=8.8, color=TEAL, weight="bold")
for i in range(4):
    x0, y0 = 5.5 + i * 1.5, 6.4 - i * 0.55
    ax.add_patch(Rectangle((x0, y0), 7.0, 6.0, facecolor="white",
                 edgecolor=TEAL, lw=1.0, zorder=4 - i * 0.1))
ax.text(12.6, 9.8, r"$A_k=I+\Delta_k$", ha="center", fontsize=8.2, zorder=6)
ax.text(12.6, 7.9, r"$k=1\ldots K$", ha="center", fontsize=7.2, color=TEAL, zorder=6)
note(12.0, 4.2, "identity-init, so training\nstarts as temporal-only")
box(25.5, 6.2, 21, 6.0, r"$\mathrm{exo}_k=\sum_{r\neq t}A_k[t,r]\,u_{r,k}$" "\n" "self term zeroed",
    ec=TEAL, fs=7.6)
arr(20.5, 9.2, 25.5, 9.2, c=TEAL)

# --------------------------------------------------------------- 4. head
box(56, 6.0, 24, 8.4, "concat\n" r"$[\,u_t \,\|\, \mathrm{exo}\,]$" "\n" r"$K$ lossless $+$ $K$ exogenous",
    fc="#EAF4EC", ec=TEAL, fs=8.0)
arr(46.5, 9.2, 56, 9.6, c=TEAL)
arr(25, 3.0, 62, 5.6, c=NAVY, lw=1.3, rad=-0.22)
note(45, 1.3, "the target's own modes bypass the mixing — this is what replace-mode destroyed", c=NAVY)

box(56, 19.0, 24, 6.4, "BiLSTM  2 × 128\n" r"input $2K$", fc=LEARN, fs=8.0)
box(56, 29.0, 24, 5.6, "Linear 256→128 → ReLU → 1", fs=8.0)
box(61, 39.0, 14, 5.0, r"$\hat y_{t+1}$", fc="white", fs=9.2)
arr(68, 14.4, 68, 19.0)
arr(68, 25.4, 68, 29.0)
arr(68, 34.6, 68, 39.0)

# ------------------------------------------- 5. what makes it different
ax.add_patch(FancyBboxPatch((85, 30.0), 45, 22.0, boxstyle="round,pad=0.5",
             facecolor="#F2F9F8", edgecolor=TEAL, lw=1.4, zorder=0))
ax.text(107.5, 49.6, "why this is not an STGNN adjacency",
        ha="center", fontsize=8.8, color=TEAL, weight="bold")
ax.text(107.5, 43.4,
        "standard:   " r"$A_{ij}$" "\n"
        "how related are $i$ and $j$\n\n"
        "here:   " r"$A_{ij}^{(k)}$" "\n"
        r"how related are they, at band $k$",
        ha="center", va="center", fontsize=8.6, linespacing=1.55)
note(107.5, 33.4, "the coupling carries frequency semantics, so\n"
                  "\"which driver matters at which timescale\"\n"
                  "is a readable object, not a black-box attention")

# --------------------------------------------------------------- 6. objective
ax.add_patch(FancyBboxPatch((85, 8.0), 45, 18.0, boxstyle="round,pad=0.5",
             facecolor="#FFF9F0", edgecolor=GOLD, lw=1.4, zorder=0))
ax.text(107.5, 23.6, "objective", ha="center", fontsize=8.8, color=GOLD, weight="bold")
ax.text(107.5, 18.2,
        r"$\mathcal{L}=\mathrm{Huber}_\beta(\hat y,y)+0.05\,\mathcal{L}_{\mathrm{bw}}$" "\n"
        r"$+\,1.0\,\mathcal{L}_{\mathrm{sep}}+0.01\,\|\Delta_k\|_{1,\mathrm{off}}$",
        ha="center", va="center", fontsize=8.2, linespacing=1.6)
note(107.5, 11.8, "was MSE. With $2K$ input channels MSE spent the extra\n"
                  "capacity on the tail: RMSE fell, MAE rose. Huber does not.", c=GOLD)
arr(85, 17.0, 80, 17.0, c=GOLD)

for i, (c, e, t) in enumerate([(LEARN, NAVY, "learned"), (FIXED, NAVY, "deterministic"),
                               (OFF, "#B9BFC6", "present, disabled")]):
    ax.add_patch(FancyBboxPatch((85 + i * 15, 2.5), 3.0, 2.0, boxstyle="round,pad=0.12",
                 facecolor=c, edgecolor=e, lw=1.0))
    ax.text(89 + i * 15, 3.5, t, fontsize=7.2, va="center")

fig.suptitle("Band-parameterised filter bank  ·  scale-conditioned spatial coupling", fontsize=12,
             color=NAVY, y=0.975, weight="bold")
fig.savefig("figures/architecture_nvmd_st.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig("figures/architecture_nvmd_st.pdf", bbox_inches="tight", facecolor="white")
print("wrote figures/architecture_nvmd_st.{png,pdf}")
