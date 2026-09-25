#!/usr/bin/env python3
r"""What the two representations actually discovered.

Temporal: where the learned bands sit, and what physical period each covers.
Spatial:  row `target` of A_k, i.e. which exogenous channel the model reaches
          for at which timescale.  This is the object the literature check says
          is the distinguishing feature -- A_ij^(k), not A_ij.
"""
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.nvmd_v3 import StructuredSpectralNVMD

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/runs_cat_s1/best.pt"
DT_H = 0.5                                        # half-hourly

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd, regions = ck["model_state"], ck["regions"]
print(f"{CKPT}   best MAE {ck['best_mae']:.3f}   R={len(regions)}\n")

# ---------------------------------------------------------------- temporal
dec = StructuredSpectralNVMD(K=8, signal_len=96, adapt=0.0)
dec.gap_logits.data = sd["decomposer.decomposer.gap_logits"]
dec.log_bw.data = sd["decomposer.decomposer.log_bw"]
with torch.no_grad():
    c, b = dec.bands()
c, b = c[0].numpy(), b[0].numpy()

print("TEMPORAL — the learned bands")
print(f"  {'k':>2} {'centre':>8} {'bandwidth':>10} {'period (h)':>12} {'Q':>6}  covers")
for k in range(8):
    per = np.inf if c[k] < 1e-9 else DT_H / c[k]
    q = c[k] / b[k] if b[k] > 0 else 0.0
    lo = DT_H / (c[k] + b[k]) if c[k] + b[k] > 0 else np.inf
    hi = DT_H / max(c[k] - b[k], 1e-9)
    span = "DC / trend" if not np.isfinite(per) else f"{lo:5.1f} – {min(hi,999):5.1f} h"
    ptxt = "  inf" if not np.isfinite(per) else f"{per:7.1f}"
    print(f"  {k+1:>2} {c[k]:8.4f} {b[k]:10.4f} {ptxt:>12} {q:6.2f}  {span}")

# ---------------------------------------------------------------- spatial
delta = sd["decomposer.coupling.delta"]                 # (K, R, R)
A = delta + torch.eye(len(regions)).unsqueeze(0)
w = A[:, 0, :].clone()                                  # row 0 = the price target
w[:, 0] = 0.0                                           # self term is zeroed in exogenous()
w = w.numpy()

print(f"\nSPATIAL — row 0 of $A_k$, the weight on each exogenous channel per band")
print(f"  total |off-diagonal| mass by band:")
for k in range(8):
    per = "DC" if c[k] < 1e-9 else f"{DT_H/c[k]:.1f} h"
    bar = "#" * int(60 * np.abs(w[k]).sum() / max(np.abs(w).sum(1).max(), 1e-9))
    print(f"    k={k+1} ({per:>7}) {np.abs(w[k]).sum():7.3f}  {bar}")

print(f"\n  top 4 channels per band")
for k in range(8):
    per = "DC/trend" if c[k] < 1e-9 else f"{DT_H/c[k]:.1f} h"
    idx = np.argsort(-np.abs(w[k]))[:4]
    top = "  ".join(f"{regions[i]}:{w[k][i]:+.3f}" for i in idx)
    print(f"    k={k+1:<2} {per:>9}  {top}")

# which channels are used at all, and are they band-specific?
print(f"\n  channels ranked by total |weight| across all bands")
tot = np.abs(w).sum(0)
for i in np.argsort(-tot)[:10]:
    where = np.argsort(-np.abs(w[:, i]))[:2]
    pers = ", ".join("DC" if c[k] < 1e-9 else f"{DT_H/c[k]:.1f}h" for k in where)
    print(f"    {regions[i]:22} {tot[i]:6.3f}   strongest at {pers}")

# the question that matters: is the coupling band-specific or flat?
norm = np.abs(w) / np.maximum(np.abs(w).sum(0, keepdims=True), 1e-9)   # per channel over bands
conc = (norm ** 2).sum(0)                                              # 1 = one band, 1/8 = flat
print(f"\n  band concentration per channel (1.0 = lives in one band, {1/8:.3f} = spread evenly)")
for i in np.argsort(-tot)[:8]:
    print(f"    {regions[i]:22} {conc[i]:.3f}")
print(f"\n  mean concentration over the 10 strongest channels: "
      f"{conc[np.argsort(-tot)[:10]].mean():.3f}   (flat would be {1/8:.3f})")
