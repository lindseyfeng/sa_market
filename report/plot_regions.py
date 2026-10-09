#!/usr/bin/env python3
"""Where the five targets are, and what the 37 input channels are.

The model predicts the five NEM regional reference prices jointly. The map is
here for one reason: the NEM is a chain, not a mesh. QLD1 reaches SA1 only
through NSW1 and VIC1, and TAS1 reaches anything at all only through Basslink.
Nothing in the model is told this. The per-band coupling is initialised at
identity and left free, so the topology is context for reading a learned
coupling, not a prior imposed on it.

Geometry is Natural Earth 1:50m admin-1, vendored to data/geo/au_states.json so
this runs without network. Western Australia and the Northern Territory are
drawn grey: they are not in the NEM and are not modelled.

    python report/plot_regions.py --out assets/regions_map.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

GEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "geo", "au_states.json")

FILL = {"SA1": "#c2410c", "NSW1": "#1d4ed8", "VIC1": "#047857",
        "QLD1": "#b45309", "TAS1": "#6d28d9"}

# Regional reference node / load centre of each region.
CITY = {"SA1": (138.60, -34.93, "Adelaide"),
        "NSW1": (151.21, -33.87, "Sydney"),
        "VIC1": (144.96, -37.81, "Melbourne"),
        "QLD1": (153.03, -27.47, "Brisbane"),
        "TAS1": (147.33, -42.88, "Hobart")}

# Label offsets in degrees, hand-placed so nothing sits on a coastline.
LAB = {"SA1": (-3.9, 1.1, "right"), "NSW1": (2.1, 1.3, "left"),
       "VIC1": (-2.7, -1.9, "right"), "QLD1": (2.1, 1.2, "left"),
       "TAS1": (1.9, -1.3, "left")}

# Reanalysis grid points behind the 12 weather channels, read from
# data/raw/wx_raw.json. All four sit in SA or western Victoria: the exogenous
# weather is SA-centric even though the targets are national.
WX = [(138.541, -34.903, "adelaide"), (138.598, -33.146, "nsa_wind"),
      (140.414, -37.715, "sesa_wind"), (144.940, -37.786, "melbourne")]

# Nominal interconnector capacity, MW. AC unless marked DC. Round numbers:
# actual limits are direction- and condition-dependent.
LINKS = [("QLD1", "NSW1", "QNI + Terranora, ~1,300 MW"),
         ("NSW1", "VIC1", "VIC-NSW, ~1,900 MW"),
         ("VIC1", "SA1", "Heywood 650 MW + Murraylink 220 MW (DC)"),
         ("VIC1", "TAS1", "Basslink, 500 MW (DC)")]

CHANNELS = [
    ("Regional prices", 5, "#c2410c", "SA1, NSW1, VIC1, QLD1, TAS1 - also the 5 targets"),
    ("Regional demand", 6, "#1d4ed8", "the 5 regions plus demand_NEM"),
    ("SA1 spreads", 4, "#0e7490", "SA1 minus each other region"),
    ("Ramp / scarcity", 3, "#047857", "ramp_SA1, ramp_VIC1, scarcity_SA1"),
    ("Weather", 12, "#b45309", "temp / wind100 / solar at the 4 sites"),
    ("Calendar", 7, "#6d28d9", "day, week, year sin/cos, and weekend"),
]


def draw_map(ax):
    geo = json.load(open(GEO))
    for key in ("WA", "NT"):
        for ring in geo[key]:
            ax.add_patch(Polygon(ring, closed=True, fc="#edecE8", ec="#c9c6be",
                                 lw=0.6, zorder=1))
    for r in FILL:
        for i, ring in enumerate(geo[r]):
            ax.add_patch(Polygon(ring, closed=True, fc=FILL[r], alpha=0.17,
                                 ec=FILL[r], lw=0.9, zorder=2))

    for a, b, _ in LINKS:
        x1, y1, _ = CITY[a]
        x2, y2, _ = CITY[b]
        ax.plot([x1, x2], [y1, y2], color="#3f3f46", lw=1.5, ls=(0, (4, 2.5)),
                alpha=0.85, zorder=4)

    for x, y, _ in WX:
        ax.plot(x, y, marker="^", ms=7, mfc="#fbbf24", mec="#78350f", mew=1.0,
                zorder=6)

    for r, (x, y, nm) in CITY.items():
        ax.plot(x, y, "o", ms=7, mfc=FILL[r], mec="white", mew=1.2, zorder=7)
        dx, dy, ha = LAB[r]
        ax.text(x + dx, y + dy, f"{r}\n{nm}", fontsize=9.5, fontweight="bold",
                color=FILL[r], ha=ha, va="center", zorder=7, linespacing=1.3)

    # Legend block in the empty ocean south-west of the continent.
    ax.plot(114.5, -40.2, marker="^", ms=6, mfc="#fbbf24", mec="#78350f",
            mew=0.9, zorder=6, clip_on=False)
    ax.text(116.0, -40.2, "reanalysis grid point (weather channels)",
            fontsize=7.4, color="#3f3f46", va="center")
    ax.plot([113.8, 115.2], [-41.5, -41.5], color="#3f3f46", lw=1.5,
            ls=(0, (4, 2.5)), zorder=6)
    ax.text(116.0, -41.5, "interconnector (schematic, not a line route)",
            fontsize=7.4, color="#3f3f46", va="center")
    for i, (_, _, lab) in enumerate(LINKS):
        ax.text(116.0, -42.6 - 1.05 * i, lab, fontsize=7.0, color="#52525b",
                va="center")
    ax.text(113.8, -38.7, "WA and NT are outside the NEM and are not modelled.",
            fontsize=7.2, color="#71717a", va="center", style="italic")

    ax.set_xlim(111.5, 160.5)
    ax.set_ylim(-45.5, -9.5)
    ax.set_aspect(1 / np.cos(np.radians(30)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)



def draw_channels(ax):
    total = sum(n for _, n, _, _ in CHANNELS)
    x0 = 7.0                       # bars start here; names live to the left
    for i, (name, n, col, note) in enumerate(CHANNELS):
        y = -i
        ax.barh(y, n, left=x0, height=0.58, color=col, alpha=0.85, zorder=3)
        ax.text(x0 - 0.45, y, name, fontsize=9, ha="right", va="center",
                fontweight="bold", color="#27272a")
        ax.text(x0 + n + 0.45, y, str(n), fontsize=9, ha="left", va="center",
                color=col, fontweight="bold")
        ax.text(x0 + n + 1.7, y, note, fontsize=7.6, ha="left", va="center",
                color="#52525b")
    y = -len(CHANNELS)
    ax.plot([x0, x0 + total], [y + 0.45, y + 0.45], color="#27272a", lw=1.1)
    ax.text(x0 - 0.45, y, "Total", fontsize=9, ha="right", va="center",
            fontweight="bold", color="#27272a")
    ax.text(x0 + 0.45, y, f"37 channels  x  96 half-hours  =  3,552 values "
                          f"per forecast window", fontsize=9, ha="left",
            va="center", fontweight="bold", color="#27272a")

    ax.text(x0 - 0.45, y - 1.1,
            "96 half-hours is 48 h of history. The +ctx336 arms hand the "
            "frequency bank 336 steps (7 days) while the\nreadout point and "
            "the scored rows stay identical, so a paired test against a "
            "short-context arm\nis still valid. All 37 channels enter the "
            "bank; the 5 prices are inputs as well as targets.",
            fontsize=7.6, color="#52525b", va="top", linespacing=1.5)

    # The NEM is a path graph. The model is never told this.
    ty = y - 4.9
    chain = [("QLD1", 4.0), ("NSW1", 12.0), ("VIC1", 20.0), ("SA1", 28.0)]
    for i in range(len(chain) - 1):
        ax.annotate("", xy=(chain[i + 1][1] - 2.1, ty),
                    xytext=(chain[i][1] + 2.1, ty),
                    arrowprops=dict(arrowstyle="<->", color="#3f3f46", lw=1.3))
    ax.annotate("", xy=(20.0, ty - 2.5), xytext=(20.0, ty - 0.75),
                arrowprops=dict(arrowstyle="<->", color="#3f3f46", lw=1.3))
    for nm, x in chain + [("TAS1", 20.0)]:
        yy = ty - 3.0 if nm == "TAS1" else ty
        ax.text(x, yy, nm, fontsize=8.6, fontweight="bold", ha="center",
                va="center", color=FILL[nm], zorder=4,
                bbox=dict(boxstyle="round,pad=0.34", fc="white",
                          ec=FILL[nm], lw=1.1))
    ax.text(x0 - 0.45, ty + 1.9,
            "The grid is a path, not a mesh", fontsize=9, fontweight="bold",
            color="#27272a", va="center")
    ax.text(x0 - 0.45, ty - 4.3,
            "QLD1 reaches SA1 only through NSW1 and VIC1; TAS1 reaches "
            "anything only through Basslink.\nThe per-band coupling is "
            "initialised at identity and left free, so the model is never "
            "told this -\nwhich is what makes a learned coupling worth "
            "reading against the map rather than a restatement of it.",
            fontsize=7.6, color="#52525b", va="top", ha="left",
            linespacing=1.5)

    ax.set_xlim(0, 42)
    ax.set_ylim(ty - 7.4, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/regions_map.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(14.0, 7.2), dpi=190)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.02,
                          left=0.012, right=0.99, top=0.885, bottom=0.03)
    draw_map(fig.add_subplot(gs[0, 0]))
    draw_channels(fig.add_subplot(gs[0, 1]))
    fig.suptitle("Five regional prices, one 37-channel window",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=0.975,
                 color="#18181b")
    fig.text(0.012, 0.915, "A. The five NEM regions, and how they connect",
             fontsize=11, ha="left", color="#18181b")
    fig.text(0.505, 0.915, "B. What one input window contains",
             fontsize=11, ha="left", color="#18181b")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print("->", a.out)


if __name__ == "__main__":
    main()
