#!/usr/bin/env python3
"""Consolidate every experiment in this line of work into one markdown file.

Regenerate at any time; it reads whatever result files exist and marks the
rest pending.
"""
import json, os
from datetime import datetime
import numpy as np

def load(p):
    return json.load(open(p)) if os.path.exists(p) else []

FOUR  = load("results/three_arms_results.json")          # claim-3, R=33 arms
STAB  = load("results/stability_results.json")           # architecture + zoo
DOSE  = load("results/dose_results.json")
SPAT  = load("results/spatial_2x2_results.json")
CHURN = {r["method"]: r for r in load("results/basis_stability.json")}

PATH = {"fixed_geo": "internal", "fixed_vmdmean": "internal",
        "nvmd_trained": "internal", "vmd_price_res": "precomputed",
        "bank": "precomputed", "wpt": "precomputed", "ewt": "precomputed",
        "emd": "precomputed"}
BASIS = {"fixed_geo": "fixed", "fixed_vmdmean": "fixed", "nvmd_trained": "fixed",
         "vmd_price_res": "re-solved", "bank": "fixed", "wpt": "fixed",
         "ewt": "re-solved", "emd": "re-solved"}
CH = {"vmd_price_res": "vmd", "bank": "bank", "wpt": "wpt", "ewt": "ewt",
      "emd": "emd", "fixed_geo": "bank", "fixed_vmdmean": "bank",
      "nvmd_trained": "bank"}

def agg(rows, key="test_mae"):
    v = np.array([r[key] for r in rows if key in r])
    if not len(v):
        return None
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)

L = []
A = L.append
A("# Decomposition for electricity price forecasting: the whole picture\n")
A(f"*generated {datetime.now():%Y-%m-%d %H:%M}*\n")

_ip = [r["test_mae"] for r in STAB if PATH.get(r["arm"]) == "internal"]
_pp = [r["test_mae"] for r in STAB if PATH.get(r["arm"]) == "precomputed"]
_rng = (f"internal {min(_ip):.3f}-{max(_ip):.3f} vs precomputed "
        f"{min(_pp):.3f}-{max(_pp):.3f}, "
        f"{'no overlap' if max(_ip) < min(_pp) else 'OVERLAPPING'}"
        if _ip and _pp else "pending")

A("## Summary\n")
A("Train 2018 / test 2019, SA1 half-hourly price. Two results hold independently "
  "of every protocol question raised below.\n")
A("- **The published gains from VMD price forecasting are leakage, not "
  "decomposition**, and what leaks is a linearly readable aggregate rather than "
  "a forecasting signal. Section 1.")
A("- **A band decomposition costs 10^3-10^5x less**, 0.1 s/year against "
  "59-18,178 s/year. Section 2.\n")
A("Every other result here is a margin against a baseline, and each is smaller "
  "than at least one protocol choice this project initially got wrong: the "
  "residual channel (0.237 MAE), the training objective (0.442 on the widest "
  "arm), epoch selection (up to 0.170), the seed itself (0.116). Read the "
  "section that measures a claim, not a one-line verdict on it.\n")
A(f"One such verdict has already moved. On the current results the ranges are "
  f"{_rng}, so **decomposing inside the forecaster no longer cleanly beats the "
  f"decompose-then-forecast pipeline** -- that separation held on two seeds and "
  f"does not on three. Section 6.\n")

A("\n## 1. The leakage finding\n")
A("This is the result the project rests on, and nothing in the later work "
  "touches it. A per-mode linear AR(48) probe asks whether each mode can be "
  "extrapolated. If modes are predictable to far below the series\' own "
  "variability, they encode the future.\n")
A("| decomposition | summed MAE | median per-mode error | % of price sigma | verdict |")
A("|---|---:|---:|---:|---|")
A("| **Per-year VMD** (what the literature runs) | 4.288 | 0.205 | 0.42% | **LEAKING** |")
A("| Causal VMD, window 96 | 15.907 | 1.357 | 2.77% | ok |")
A("| NVMD v2 | 18.837 | 1.262 | 2.58% | ok |")
A("| NVMD v3 | 16.159 | 1.606 | 3.28% | ok |")
A("\nSame algorithm, same data, same K. **Only the decomposition window "
  "changed, and per-mode extrapolation error rose 6.6x.**\n")
A("The second, independent probe is capacity irrelevance. On per-year VMD "
  "modes a **625-parameter linear regression (MAE 4.29) beats a 5.7M-parameter "
  "CNN-BiLSTM (MAE 13.42)**. When capacity does not help, nothing is being "
  "learned -- the modes are being *read*, not forecast.\n")
A("We reproduced the original **7.11 MAE / 11.58 RMSE** exactly and traced it "
  "to this effect. Segment the decomposition causally and it degrades to "
  "**~10.7**.\n")
A("Two mechanisms are entangled in that 7.11, and separating them is worth "
  "saying out loud: per-year VMD solves **one** decomposition for the whole "
  "year, so it is both leaky *and* perfectly consistent across windows. Causal "
  "VMD removes the leakage and gives up the consistency. A fixed filter bank "
  "keeps the consistency without the leakage.")

A("\n## 2. Cost\n")
A("| | mode generation, one year (~17k windows) |")
A("|---|---|")
A("| Causal VMD, window 96, 7 cores | 123.4 s |")
A("| Causal VMD across the (alpha, K) sweep | **59 s to 18,178 s** |")
A("| Fixed / learned filter bank | **0.1 s** |")
A("\nA factor of **600 to 180,000**. The point is not speed for its own sake: "
  "the 18-config VMD sweep that made the baseline fair cost 22.5 hours, and the "
  "bank equivalent is minutes. **Tuning is affordable for one method and not "
  "the other**, which is what makes a fair comparison possible at all.")

A("\n## 3. The architecture\n")
A("A **learnable frequency-domain filter bank placed inside the forecaster**. "
  "In one line:\n")
A("$$x \\;\\rightarrow\\; \\mathrm{rFFT} \\;\\rightarrow\\; K "
  "\\text{ Gaussian bands} \\;\\rightarrow\\; \\mathrm{irFFT} "
  "\\;\\rightarrow\\; K \\text{ modes} \\;\\rightarrow\\; "
  "\\mathrm{BiLSTM} \\;\\rightarrow\\; \\hat y$$\n")

A("**1. The input window.** $x \\in \\mathbb{R}^{B\\times 1\\times L}$, "
  "with $L=96$ half-hourly steps.\n")

A("**2. Into the frequency domain.** An rFFT gives $X(f)$. The split is "
  "decided in frequency, not in time: the model decides which frequency "
  "belongs to which mode.\n")

A("**3. $K$ Gaussian bands.** Each mode carries a centre $c_k$ and a "
  "bandwidth $b_k$, giving an unnormalised mask\n")
A("$$G_k(f) = \\exp\\!\\Big(-\\frac{(f-c_k)^2}{2b_k^2}\\Big)$$\n")
A("The centres are **not** free parameters. They are built as\n")
A("$$\\text{gap logits } \\theta \\;\\rightarrow\\; "
  "\\mathrm{softmax} \\;\\rightarrow\\; \\mathrm{cumsum} "
  "\\;\\rightarrow\\; \\text{rescaled to } [0,\\tfrac12]$$\n")
A("which forces $c_1 < c_2 < \\dots < c_K$ and pins $c_1 = 0$, so the first "
  "mode is a dedicated trend and low-frequency channel.\n")

A("**4. Normalise the masks.** At every frequency,\n")
A("$$M_k(f) = \\frac{G_k(f)}{\\sum_{j=1}^{K} G_j(f)}, \\qquad "
  "\\text{so} \\qquad \\sum_{k=1}^{K} M_k(f) = 1 \\;\\; \\forall f$$\n")
A("That is a **partition of unity**.\n")

A("**5. The modes.** $X_k(f) = M_k(f)\\,X(f)$, then $z_k = "
  "\\mathrm{irFFT}(X_k)$. Because the masks sum to one,\n")
A("$$\\sum_{k=1}^{K} z_k = x$$\n")
A("The decomposition is **lossless**, so no residual channel is needed and "
  "none can become a junk dump. Measured reconstruction error is 4.8e-07, "
  "which is float32 round-off.\n")

A("**6. Forecast.** The $K$ modes go to a 2-layer bidirectional LSTM with "
  "hidden size 128, then a $256 \\rightarrow 128 \\rightarrow 1$ head.\n")

A("![architecture](figures/architecture_nvmd_st.png)\n")
A("*`analysis/plot_architecture.py`, drawn from the code. The visual centre is "
  "the per-band coupling: one $R\\times R$ matrix per frequency band, which is "
  "the part of this design the literature does not already contain (section "
  "9).*\n")
A("### What \"fixed across windows\" does and does not mean\n")
A("It does **not** mean the band parameters are untrained. It means they are "
  "**global model parameters**: learned through the forecast loss, but shared "
  "by every window. VMD re-solves a fresh set of modes for each window; this "
  "learns one set and applies it everywhere.\n")
A("| | causal VMD | this layer |")
A("|---|---|---|")
A("| where the bands come from | an optimisation solved per window | global "
  "parameters shared across all windows |")
A("| where it sits | preprocessing | inside the forecaster |")
A("| mode identity | mode $k$ can mean different things in different windows | "
  "mode $k$ has a stable frequency meaning |")
A("| what it optimises | a decomposition objective, separate from the forecast "
  "| the forecast loss, end to end |")
A("\nThree structural guarantees follow, none of them a penalty term:\n")
A("1. centres strictly increasing, so modes cannot swap order;")
A("2. the first mode pinned to DC, so the trend always has a channel;")
A("3. exact reconstruction, so no junk residual channel exists.\n")
A("So it is not a deep neural decomposer. It is **a learnable frequency-domain "
  "filter bank with hard frequency-ordering and reconstruction constraints, "
  "trained end to end by the forecasting task**.\n")

A("### Where the parameters actually are\n")
A("| component | trainable parameters |")
A("|---|---:|")
A("| the bands ($\\theta$ and $\\beta$, 8 each) | **16** |")
A("| BiLSTM | 536,576 |")
A("| head | 33,025 |")
A("| total | 627,265 |")
A("\nThe decomposition is **0.0026%** of the model. Whatever it contributes is "
  "structural, not capacity. Two further facts from the code, both relevant to "
  "how the result should be stated:\n")
A("- With `adapt=0` the masks have shape $(1, K, F)$, not $(B, K, F)$. They do "
  "not depend on the input at all, so the decomposition is a **fixed linear "
  "operator** -- eight filters applied by circular convolution. An "
  "input-adaptive variant was tested and lost.")
A("- The layer still carries an unused `SignalEncoder` and `gap_head` totalling "
  "**57,640 parameters**, dead when `adapt=0`. They inflate any reported "
  "parameter count by ~9% and should be deleted or gated.\n")
A("Section 7 measures what those 16 parameters are worth: training them rather "
  "than hard-coding them moves test MAE by **0.002**, against a seed spread of "
  "0.11. The layer pays; the learning does not.\n")

A("```mermaid")
A("flowchart LR")
A("  X[\"price window<br/>x : (B, 1, L)\"] --> F[\"rFFT\"]")
A("  G[\"gap logits θ (K)\"] --> SM[\"softmax → cumsum<br/>→ rescale to [0, ½]\"]")
A("  SM --> C[\"centres c_k<br/>strictly increasing<br/>c_1 = DC\"]")
A("  BW[\"log bandwidth β (K)\"] --> B[\"widths b_k<br/>tied to the local gap\"]")
A("  C --> M[\"Gaussian masks<br/>normalised to a<br/>partition of unity\"]")
A("  B --> M")
A("  F --> MUL[\"multiply\"]")
A("  M --> MUL")
A("  MUL --> I[\"irFFT\"]")
A("  I --> Z[\"K modes<br/>sum exactly to x\"]")
A("  Z --> L1[\"BiLSTM 128 × 2\"]")
A("  L1 --> H[\"256 → 128 → 1\"]")
A("  H --> Y[\"ŷ\"]")
A("```")

A("\n### The spatial variant\n")
A("The same bank runs on every channel of the panel -- NEM regional demand, "
  "interconnector spreads, weather at four sites, calendar terms, 33 in all. "
  "After decomposition, one $R \\times R$ mixing matrix $A_k$ **per frequency "
  "band** lets channels exchange information only within the same band, so a "
  "6-hour wind ramp cannot leak into the 3-day price trend.\n")
A("The target\'s own $K$ modes are carried through **untouched**, and a purely "
  "exogenous block of $K$ modes is appended, giving the LSTM $2K$ input "
  "channels. $A_k = I + \\Delta$ with $\\Delta$ zero-initialised and shape "
  "$(K, R, R)$, so at step 0 the model is exactly the temporal one and can only "
  "depart from it if the data pays. That adds 8,704 parameters.\n")
A("```mermaid")
A("flowchart LR")
A("  P[\"panel<br/>(B, R, L)\"] --> BK[\"shared filter bank<br/>per channel\"]")
A("  BK --> MM[\"modes (B, R, K, L)\"]")
A("  MM --> OWN[\"target's own K modes<br/>lossless, untouched\"]")
A("  MM --> CP[\"per-band mixing A_k<br/>self weight zeroed\"]")
A("  CP --> EXO[\"exogenous block (B, K, L)<br/>zero at init\"]")
A("  OWN --> CAT[\"concat → 2K channels\"]")
A("  EXO --> CAT")
A("  CAT --> LS[\"BiLSTM + head\"]")
A("```")
A("\nAn earlier version let the mixed modes **replace** the target\'s own. "
  "That destroys the partition of unity -- reconstruction error 0.00 to 1.82, "
  "cross terms 2-6.5x the self term -- so the head never saw a faithful price "
  "encoding. It corrupted the DC and daily bands, which carry ~88% of ordinary "
  "intervals, while the fast bands gained real spike information. **MAE got "
  "worse while RMSE got better**, consistently. Section 9 has what the concat "
  "form is actually worth.\n")
A("![decomposed waves](figures/decomposed_waves.png)\n")
A("*One 48 h window of 2019, every band shown before and after coupling. The "
  "second panel is the argument: pre-coupling the modes sum to the input "
  "exactly, post-coupling they do not. Bands 2-4 -- trend, daily, half-daily -- "
  "are visibly rescaled, which is where the ordinary intervals live, while "
  "bands 6-8 pick up real spike structure at the right-hand edge. That is the "
  "MAE-worse/RMSE-better trade drawn out.*\n")

A("\n## 4. Why the classical bands underperform: the physics, not the algorithm\n")
A("This section is what makes the later null results legible. Swapping "
  "decomposition algorithms moves nothing, because they all hand the model "
  "bands with the same three defects.\n")
A("| | causal VMD, tuned K=8 | NVMD v3, 9 modes |")
A("|---|---:|---:|")
A("| participation ratio | 1.63 | **6.11** |")
A("| energy in top mode | 77.36% | **28.16%** |")
A("| slowest **resolvable** band (window is 48 h) | 21.8 h | **26.9 h** |")
A("| bands slower than the daily cycle | 0 | **1**, plus a trend slot |")
A("| top-mode ablation cost | +10.80 MAE | +0.03 MAE |")
A("| centres ordered | post-hoc omega sort | **by construction, every window** |")
A("\n**Defect 1: the lowest band is not a band.** VMD\'s mode 1 has bandwidth "
  "0.0631 at centre 0.0249, so its width is **2.5x its own centre** and the "
  "band spans DC. Its bandwidth is also flat regardless of centre, where a "
  "filter bank scales width with centre (constant-Q, stable at Q ~ 0.7-0.9 "
  "from mode 3 up).\n")
A("**Defect 2: K modes are not K modes.** Participation ratio 1.63. Mode 1 "
  "holds 77% of the energy and costs +10.80 MAE to ablate, while 6 of 9 come "
  "out at under 0.01 MAE. You ask for eight bands and get about 1.6.\n")
A("**Defect 3: everything slower than a day lands in the smear.** A 96-sample "
  "window at half-hourly sampling is **48 hours long**, so nothing slower than "
  "that is a resolvable oscillation. Inside that limit VMD\'s slowest band is "
  "centred at **21.8 h** -- the daily cycle is its slowest channel, and "
  "everything below that frequency falls into the mode-1 smear from defect 1. "
  "The bank\'s slowest resolvable band sits at **26.9 h**, with a separate "
  "sub-resolution slot at 75 h, so slow drift and the daily cycle occupy "
  "different channels instead of being merged into one.\n")
A("*Correction to `attic/RESULTS-superseded.md` section 3, which reports a \"longest period "
  "represented\" of 307.2 h for the 9-mode configuration and calls it 15x "
  "VMD\'s reach. A 48-hour window cannot resolve a 307-hour period. That "
  "number describes a filter\'s nominal centre, not resolvable content, and "
  "overstates the difference. The defensible version is the one above.*\n")
A("![drift and coverage](figures/drift_and_coverage.png)\n")
A("*Left: band centres over 300 consecutive windows. VMD re-solves and the "
  "centres wander -- 12.0% of a band gap per step, crossing half a gap in "
  "30.8% of steps -- while the bank\'s are flat lines because they are written "
  "down once. Right: the period each band is tuned to. Everything in the "
  "shaded region is slower than the window itself, so it is a trend slot "
  "rather than an oscillation; VMD puts one band there and the bank two. "
  "Regenerate with `python3 -m report.plot_drift`.*")
A("These are properties of the **modes**, and every classical method we tested "
  "shares them. That is why sections 7, 8 and 10 come back empty: they vary "
  "the *algorithm* while the physics of the resulting bands stays put. The one "
  "comparison that does move the metric changes what the bands are.")

A("![interpretability](figures/interpretability_nvmd_vs_vmd.png)\n")
A("*Band structure, energy concentration, period coverage and per-mode "
  "ablation. Panel D is the one this project loses, and it is included "
  "deliberately: a reviewer will run that test.*\n")
A("\n### The two constructions, side by side\n")
A("VMD solves, per window, for K modes $u_k$ and centres $\\omega_k$:\n")
A("$$\\min_{\\{u_k\\},\\{\\omega_k\\}} \\sum_k \\Big\\| "
  "\\partial_t\\big[(\\delta(t) + \\tfrac{j}{\\pi t}) * u_k(t)\\big]"
  "e^{-j\\omega_k t} \\Big\\|_2^2 \\quad \\text{s.t.} \\quad "
  "\\sum_k u_k = f$$\n")
A("The objective is narrowbandness per window, and the bands are whatever "
  "the iteration converges to.\n")
A("The bank instead **parameterises** the same two quantities and fixes them:\n")
A("$$c_k = \\frac{1}{2}\\cdot\\frac{\\sum_{j\\le k}g_j - g_1}"
  "{\\sum_j g_j - g_1}, \\qquad g = \\mathrm{softmax}(\\theta), "
  "\\qquad b_k = b_{\\min,k} + \\mathrm{softplus}(\\beta_k)$$\n")
A("$$m_k(f) = \\frac{\\exp\\!\\big(-\\tfrac12 (f-c_k)^2/b_k^2\\big)}"
  "{\\sum_{k\'} \\exp\\!\\big(-\\tfrac12 (f-c_{k\'})^2/b_{k\'}^2\\big)}, "
  "\\qquad u_k = \\mathcal{F}^{-1}\\!\\big[m_k \\odot \\mathcal{F}f\\big]$$\n")
A("Writing the bands down rather than solving for them buys a set of "
  "guarantees, none of which is a penalty term and none of which the "
  "variational form provides:\n")
A("| property | fixed bank | causal VMD |")
A("|---|---|---|")
A("| a band exists at DC | **yes** -- $c_1 = 0$ by construction | no; measured "
  "lowest centre 0.0001 with width 0.0631, so the band spans DC rather than "
  "sitting on it |")
A("| centres ordered, identity stable across windows | **yes** -- $c_k$ is a "
  "cumsum of a softmax, so $c_1 < \\dots < c_K$ for any $\\theta$ | no; "
  "centres move 12.4% of a band gap between adjacent windows and cross half a "
  "gap in 8.2% of steps |")
A("| modes sum exactly to the input | **yes** -- $\\sum_k m_k(f) = 1$ for "
  "every $f$, so no residual channel and none can become a junk dump | no; the "
  "residual is 8.5-9.5% of price sigma |")
A("| width scales with centre (constant-Q) | **yes** -- $b_k$ tied to the "
  "local gap $\\tfrac12(c_{k+1}-c_{k-1})$, Q ~ 0.8-1.9 | no; width is flat "
  "in centre, so Q runs 0.42 to 7.16 |")
A("| resolution follows the signal\'s energy | **yes** -- geometric spacing "
  "puts five of eight bands below $f = 0.08$, where 73% of the power is | no; "
  "near-uniform spacing puts three there and five where there is almost "
  "nothing |")
A("\nThese are properties of the construction, not results we tuned for. They "
  "hold for any $\\theta$, on any signal, in every window.\n")
A("![band comparison](figures/band_comparison.png)\n")
A("*Left: causal VMD\'s bank, with bars marking how far each centre drifts "
  "between adjacent windows. Its lowest bands are broad plateaus spanning DC "
  "rather than bands, and its width is flat in centre, so Q runs from 0.42 at "
  "mode 2 to 7.16 at mode 8. Middle: the fixed geometric bank, constant-Q at "
  "Q ~ 0.8-1.9 from mode 2 up. Right: the price power spectrum. **73% of the "
  "power sits below f = 0.08** -- the bank puts five of eight bands there, VMD "
  "puts three and spends the other five where there is almost nothing.*\n")
A("| mode | VMD centre | VMD width | VMD Q | bank centre | bank width | bank Q |")
A("|---:|---:|---:|---:|---:|---:|---:|")
for _k, (_vc, _vb, _bc, _bb) in enumerate(zip(
        [0.0001, 0.0268, 0.0862, 0.1567, 0.2292, 0.3026, 0.3776, 0.4519],
        [0.0631] * 8,
        [0.0000, 0.0066, 0.0186, 0.0401, 0.0789, 0.1486, 0.2741, 0.5000],
        [0.0028, 0.0079, 0.0142, 0.0254, 0.0456, 0.0813, 0.1442, 0.0938])):
    _qv = _vc / _vb if _vc > 1e-6 else 0.0
    _qb = _bc / _bb if _bc > 1e-6 else 0.0
    A(f"| {_k+1} | {_vc:.4f} | {_vb:.4f} | {_qv:.2f} | {_bc:.4f} | "
      f"{_bb:.4f} | {_qb:.2f} |")
A("\nRegenerate with `python3 -m report.plot_bands`.")

A("\n---\n")
A("**Sections 5 onward are the matched-protocol study.** Train 2018 / test "
  "2019, SA1 half-hourly price, window 96, horizon 1, identical rows, one head, "
  "one budget. Selection on a validation tail of the train year with a "
  "96-window embargo; test scored once from those weights. `honest` is that "
  "number. `cherry` is the minimum of test MAE over epochs, which is the "
  "statistic `attic/RESULTS-superseded.md` section 13.3 and `benchmark_seeds.py` report, kept "
  "alongside so the two sets of tables can be reconciled.\n")

# ---------------------------------------------------------------- experiment 1
A("\n## 5. Claim 3: does spatio-temporal NVMD beat VMD?\n")
A("Yes on RMSE, robustly. On MAE only under a tail-insensitive objective, and "
  "the two arms cross. Both answers need VMD given its residual channel first, "
  "a correction worth more than the margin under test.\n")
A("![loss sweep](figures/loss_sweep.png)\n")
A("*`analysis/plot_loss_sweep.py`. L1 and MSE are the training-time names for "
  "MAE and RMSE: an L1-trained model optimises exactly the metric the left "
  "panel reports, which is why it sits lowest there.*\n")
A("This figure is the clearest statement of the result. Sliding the objective from L1 to MSE, the "
  "spatial arm moves 0.736 MAE and the baseline 0.281 -- a 2.6x steeper slope, "
  "which is what having $2K$ head inputs against $K{+}1$ buys the objective to "
  "reallocate. **The lines cross.** Under MSE the spatial arm is 0.108 *behind*; "
  "under L1 it is 0.347 ahead. Meanwhile the RMSE gap sits at a near-constant "
  "**~1.4, about 5%**, at every objective.\n")
A("So the honest summary is two statements, not one:\n")
A("- **The RMSE advantage is a property of the representation.** It is ~5% and "
  "no choice of objective moves it.")
A("- **The MAE advantage is a property of the representation *and* the "
  "objective.** It exists under L1 and Huber and reverses under MSE. Reporting "
  "it without naming the loss would be reporting a choice as a result.\n")
A("**The answer.** Identical rows, `R=33` panel, matched Huber objective, "
  "residual returned to VMD:\n")
A("| objective | `vmd_price_res` | `nvmd_st` | MAE | RMSE |")
A("|---|---:|---:|---:|---:|")
A("| L1 | 14.091 / 26.553 | **13.744 / 25.077** | **-2.5%** | **-5.6%** |")
A("| Huber $\\beta$=1 | 14.232 / 26.564 | **14.038 / 25.048** | **-1.4%** | **-5.7%** |")
A("| MSE | **14.372** / 26.427 | 14.480 / **25.121** | +0.8% | **-4.9%** |")
A("\nThe L1 row is the best matched pair and is single-seed on the spatial "
  "arm, so the Huber row is the one to quote. The MSE row is in the table "
  "because a result that reverses under a defensible objective should not be "
  "hidden.")
A("\n**VMD must be given its residual.** VMD does not reconstruct its input "
  "exactly; the residual is **8.5-9.5% of the price standard deviation**. "
  "Scoring it on $K$ modes alone hands it ~91% of the signal while a "
  "partition-of-unity filter bank gets 100%, and costs it **0.237 MAE** "
  "(14.561 against 14.324 on seed 1). Every arm in this document carries a "
  "residual channel. Anyone reproducing a VMD baseline should check this "
  "first.\n")
A("**The objective must match the reported metric.** Training on "
  "`F.mse_loss` while selecting and reporting MAE is not neutral for an arm "
  "with spare capacity, and `--concat` gives the spatial arm a block the "
  "others do not have. Section 8 has the mechanism and a test of it.\n")
if FOUR:
    A("**What the original MSE run said**, correct for its protocol and wrong "
      "as a verdict on the architecture:\n")
    A("| arm | information | decomposition | honest | cherry | selection effect |")
    A("|---|---|---|---:|---:|---:|")
    for a_, info, dec in [("nvmd_temporal", "temporal", "joint, coupling frozen"),
                          ("nvmd_st", "spatial", "joint, per-band coupling"),
                          ("vmd_price", "temporal", "univariate VMD, no residual"),
                          ("vmd_panel", "spatial", "univariate VMD x 26")]:
        rs = [r for r in FOUR if r["arm"] == a_]
        if not rs:
            continue
        h = agg(rs); c = agg(rs, "test_mae_cherry")
        gap = f"{h[0]-c[0]:+.3f}" if c else "--"
        cs = f"{c[0]:.3f}" if c else "--"
        A(f"| `{a_}` | {info} | {dec} | **{h[0]:.3f}** ± {h[1]:.3f} | {cs} | {gap} |")
    A("\nRead as it stands, `nvmd_st` (14.480) beats `vmd_price` (14.524) and "
      "claim 3 is true. Read with the residual returned, `vmd_price_res` "
      "(14.382) beats `nvmd_st` and claim 3 is false. Read with the objective "
      "matched as well, the spatial arm wins again. **Three protocols, three "
      "verdicts, one architecture.**\n")
    A("**Two findings in that table survive every correction.**\n")
    A("- **Handing classical VMD the same 26 exogenous channels is "
      "catastrophic**, ~18.0 against 14.4 for the same channels through a "
      "joint decomposition. Having the data is not the same as being able to "
      "use it. That arm shares hyperparameters with an 8-channel arm and is "
      "arguably under-tuned, but not by 3.6 MAE.")
    A("- **The selection effect differs by a factor of ten across arms**, "
      "+0.011 to +0.170. Selecting on test is not a neutral transformation: it "
      "moves some arms much further than others, so a table built that way can "
      "reorder methods. See the retraction in section 7.")
A("\n**Still open.** Two seeds against a seed noise of **0.116** "
  "(`vmd_price_res` scores 14.324 / 14.440 across seeds, while five "
  "decomposition families span roughly 0.04 -- one method's seed variation is "
  "several times the difference between methods). Any margin here that is not "
  "paired across seeds is reporting the seed. The L1 pair of the matched "
  "baseline is still running, and `nvmd_temporal` is absent from the headline "
  "table because Huber makes it *worse*, 14.163 to 14.258.")

# ---------------------------------------------------------------- confound
# ---------------------------------------------------------------- experiment 2
A("\n## 6. Where neural decomposition earns its place\n")
A("Two ways to deliver a decomposition to a sequence model:\n")
A("- **internal** -- hand the model the raw signal window and decompose it "
  "inside the forward pass, as a differentiable layer. The model sees each "
  "mode\'s waveform across one consistent window.")
A("- **precomputed** -- run the decomposition offline per window, keep the "
  "last sample of each mode, and feed the resulting per-timestep mode vectors. "
  "This is what the entire decomposition-plus-deep-learning literature does, "
  "including every comparison in `attic/RESULTS-superseded.md` sections 2, 8 and 10.\n")
if STAB:
    A("| arm | basis | delivery | churn | honest | seeds |")
    A("|---|---|---|---:|---:|---:|")
    order = ["fixed_geo", "fixed_vmdmean", "nvmd_trained",
             "bank", "vmd_price_res", "wpt", "ewt", "emd"]
    for a in order:
        rs = [r for r in STAB if r["arm"] == a]
        ch = CHURN.get(CH.get(a, ""), {}).get("churn")
        chs = f"{ch:.1%}" if ch is not None else "--"
        if not rs:
            A(f"| `{a}` | {BASIS[a]} | {PATH[a]} | {chs} | *pending* | 0 |")
            continue
        h = agg(rs)
        A(f"| `{a}` | {BASIS[a]} | {PATH[a]} | {chs} | **{h[0]:.3f}** ± {h[1]:.3f} | {h[2]} |")
    ip = [r["test_mae"] for r in STAB if PATH.get(r["arm"]) == "internal"]
    pp = [r["test_mae"] for r in STAB if PATH.get(r["arm"]) == "precomputed"]
    if ip and pp:
        _sep = max(ip) < min(pp)
        _spread = max(max(ip) - min(ip), max(pp) - min(pp))
        A(f"\nEvery internal run lands in **{min(ip):.3f}-{max(ip):.3f}**; every "
          f"precomputed run lands in **{min(pp):.3f}-{max(pp):.3f}**. "
          + (f"No overlap, and the gap between the groups is larger than the "
             f"seed spread within either."
             if _sep else
             f"**They overlap.** An earlier version of this section reported "
             f"separation; that held on two seeds per arm and does not on "
             f"three. The within-group spread is now {_spread:.3f}, against a "
             f"seed noise of 0.116, so the delivery path is not resolved by "
             f"these runs. What the numbers still support is the weaker "
             f"statement that the *worst* outcomes are all on the precomputed "
             f"side."))
    A("\nThe *comparison* is clean even though the result is not: `fixed_geo` "
      "and `bank` are **the same Gaussian filter bank**, differing only in "
      "whether the decomposition happens inside the model or is precomputed "
      "per timestep. So whatever separation exists is an architectural effect "
      "and not a basis effect -- the design isolates the right thing. What it "
      "has not yet done is resolve it.")
    A("\n**What this can and cannot carry.** It cannot carry a paper on its own "
      "at three seeds with overlapping ranges. What it does have is a clean "
      "contrast that costs nothing to extend -- more seeds on `fixed_geo` "
      "against `bank` is the cheapest open experiment in this document, and it "
      "either resolves the delivery path or shows the effect was seed noise. "
      "The attraction of the hypothesis is unchanged: it does not depend on "
      "the learned parameters doing anything, which is what would make it "
      "robust to the \"did you tune the baseline as hard\" objection, and it "
      "transfers to any decomposition expressible as a differentiable "
      "filtering step.")
    A("\n**The mechanism is plausible but untested.** On the precomputed path "
      "the value at time t is the *last sample* of the decomposition of window "
      "[t-95, t], so a sequence of them is a trajectory of last samples. The "
      "internal path hands the model each mode\'s waveform across one "
      "consistent window, which is strictly more. We have not isolated that, "
      "and it is the next thing to test.")

# ---------------------------------------------------------------- retraction
A("\n## 7. Retracted: basis stability predicts accuracy\n")
A("We proposed that what separates these methods is whether the basis is "
  "re-solved in every window, measured as **churn** -- the fraction of "
  "adjacent-window steps in which some mode's spectral centroid moves more "
  "than half a band gap.\n")
if CHURN:
    A("| method | basis | drift | churn |")
    A("|---|---|---:|---:|")
    for m in ("bank", "wpt", "vmd", "emd", "ewt"):
        r = CHURN.get(m)
        if r:
            A(f"| {m} | {r['kind']} | {r['drift']:.1%} | {r['churn']:.1%} |")
    A("\nChurn separates the two families by 12-27x with no overlap. **Accuracy does "
      "not follow it.** Within the matched precomputed path, fixed and "
      "re-solved bases interleave, and the whole group spans 0.3% while churn "
      "spans a factor of 26.")
A("\nThe hypothesis was rejected by a control that was part of the design: "
  "`bank` holds the filter bank identical to `fixed_geo` and changes only the "
  "delivery path. Most of the gap we had attributed to stability moved with "
  "the path, not the basis.")
A("\n**`EMD` is a separate story.** It is the worst arm by a wide margin and "
  "churn does not explain it either. On 96-sample windows EMD often fails to "
  "sift 8 IMFs:\n")
A("| method | live modes per window | windows where the live set changes |")
A("|---|---|---:|")
A("| EMD | mean **5.93**, min 4, max 8 | **23.1%** |")
A("| EWT / WPT / bank | always 8 | 0% |")
A("\nChannels are not merely drifting, they intermittently do not exist.")

# ---------------------------------------------------------------- spatial
A("\n## 8. The spatially-encoded variant\n")
A("**The short version.** Under MSE the spatial arm loses by 0.317 and the obvious\nreading is that spatial coupling does not pay. That reading is an artefact of\nthe objective: MSE decides where the arm's extra capacity goes, and it sends it\nto the tail. Under a matched Huber objective the same arm wins, and the\nmechanism makes a prediction that holds -- a narrower arm moves a third as far\nwhen the objective changes. The rest of this section is that argument in order.")

st = [r for r in FOUR if r["arm"] == "nvmd_st"]
tp = [r for r in FOUR if r["arm"] == "nvmd_temporal"]
if st and tp:
    A("Both arms are the same model on the **internal** path, differing only in "
      "whether the per-band cross-channel coupling is enabled. So this sits "
      "inside the architecture family that wins section 6, and isolates the "
      "spatial encoding itself.\n")
    A("| arm | seed 1 | seed 2 | mean | seed spread | selection effect |")
    A("|---|---:|---:|---:|---:|---:|")
    for name, rs in (("nvmd_temporal", tp), ("nvmd_st", st)):
        v = [r["test_mae"] for r in sorted(rs, key=lambda z: z["seed"])]
        se = np.mean([r["test_mae"] - r["test_mae_cherry"] for r in rs])
        A(f"| `{name}` | {v[0]:.3f} | {v[1]:.3f} | **{np.mean(v):.3f}** | "
          f"{max(v)-min(v):.3f} | {se:+.3f} |")
    d = np.mean([r["test_mae"] for r in st]) - np.mean([r["test_mae"] for r in tp])
    sp = max(max(v) - min(v) for v in
             ([r["test_mae"] for r in st], [r["test_mae"] for r in tp]))
    A(f"\n**Spatial encoding costs {d:+.3f} MAE.** The larger of the two arms\' "
      f"seed spreads is {sp:.3f}, so the penalty is about {abs(d)/sp:.0f}x the "
      f"noise scale -- not decisive on two seeds, but consistent in sign and "
      f"size across both. It also lands worse than every arm on the "
      f"precomputed path except EMD, which means enabling spatial coupling "
      f"gives back more than the architecture won.\n")
    A("One measurement bears on why, and points at redundant conditioning "
      "rather than absent signal:\n")
    A("- `attic/RESULTS-superseded.md` section 13.4 measured the exogenous block taking 60-95% "
      "of head input variance while buying ~1% MAE. A block that dominates the "
      "input and moves the metric that little is behaving as redundant "
      "conditioning.\n")
    A("*An earlier draft also blamed the arm\'s large selection effect on its "
      "8x33x33 coupling tensor. That does not hold: across the stability run "
      "the selection effect ranges +0.000 to +0.329 with no relation to "
      "parameter count, so we have no validated mechanism for it and only "
      "report that it is arm-dependent.*\n")
    A("**This is a verdict on the current design, not on spatial information.** "
      "Two reasons to withhold judgement, both testable and both in flight:\n")
    A("1. **Horizon.** Every number above is h=1, which `attic/RESULTS-superseded.md` section 11 "
      "records as saturated -- persistence 14.40 against a best model of ~14.3. "
      "Section 11a measured the spatial coupling gain at **-2.21 MAE at h=6**, "
      "decaying to zero by h=48. Testing a 2.21-point effect in a 0.1-point "
      "window cannot resolve it.")
    A("2. **The exogenous channels are fed as history, not as forecasts.** "
      "`PanelWindowDataset` hands the model every channel over the trailing "
      "window and asks it to predict h steps ahead. Real load and price "
      "forecasting conditions on the *forecast* weather and demand for the "
      "target interval. Trailing weather is largely already priced into the "
      "recent spread; forward weather is where the incremental information "
      "should be.\n")
    A("\n**This line is open, not closed.** The panel carries real structure -- "
          "`spread_SA1_TAS1` alone correlates +0.513 with the target, and the panel's "
          "drivers are physically distinct in a way a univariate decomposition "
          "cannot represent at all. What has failed so far is one particular way of "
          "injecting that structure, at one horizon where nothing is resolvable. "
          "Designs still to try, in the order we would try them:\n")
    A("1. **Forward exogenous windows** -- in flight, as the `fwd` arm.")
    A("2. **Do not decompose the exogenous channels at all.** The target is being "
          "extrapolated and needs a representation; the exogenous channels are "
          "only being conditioned on. Decomposing them is cost and variance, which "
          "is how `vmd_panel` reached 18.0.")
    A("3. **Compress before injecting.** 26 channels into 2-4 learned directions, "
          "aimed straight at the 60-95% input-variance problem.")
    A("4. **Inject as a gate rather than as extra channels.** Exogenous drivers "
          "plausibly change the conditional *scale* of price, not its level, which "
          "makes FiLM over the target\'s bands the better inductive bias.")
    A("5. **Restrict coupling to the bands where 13.4 found signal**, instead of "
          "learning a full K x R x R tensor whose variance cost we have measured.")
    A("The `spatial 2x2` experiment crosses these two factors, so the outcome "
      "distinguishes \"the information is not there\" from \"we tested where "
      "there was no room\" from \"we fed it the wrong window\".")

# ---------------------------------------------------------- h=1 noise floor
if SPAT:
    ctl = [r for r in SPAT if r["cfg"] == "h1_price"]
    if len(ctl) > 1:
        t = sorted(r["test_mae"] for r in ctl)
        shifts = [r["test_mae"] - r["val_mae"] for r in ctl]
        A("\n### The h=1 row cannot resolve an exogenous effect\n")
        A("Measured, not assumed. The price-only control at h=1:\n")
        A("| seed | val MAE | test MAE | val -> test shift |")
        A("|---|---:|---:|---:|")
        for r in sorted(ctl, key=lambda z: z["seed"]):
            A(f"| {r['seed']} | {r['val_mae']:.3f} | {r['test_mae']:.3f} | "
              f"{r['test_mae']-r['val_mae']:+.3f} |")
        A(f"\nThe control alone varies by **{t[-1]-t[0]:.3f}** across two seeds, "
          f"and validation understates test by "
          f"{min(shifts):+.3f} to {max(shifts):+.3f}. Both exceed the total "
          f"headroom at this horizon, where persistence scores 14.40 against a "
          f"best model near 14.3.\n")
        A("So any h=1 comparison between price-only, trailing and forward "
          "exogenous is **inside the noise floor by construction**. We record "
          "the row for completeness and read nothing into it. This is also why "
          "the earlier conclusion that the spatial panel was useless was never "
          "evidence of absence -- it was measured here.")

# ---------------------------------------------------------------- noise
A('`--loss {mse,huber,l1}` and `--huber-beta` were added to\n`experiments/run_three_arms.py`. Sweeping the objective on `nvmd_st`:\n\n| loss | test MAE | test RMSE | seeds |\n|---|---:|---:|---:|\n| L1 | 13.744 | 25.077 | 1 |\n| Huber beta=0.5 | 13.909 | 25.165 | 1 |\n| **Huber beta=1.0** | **14.038** | **25.048** | 2 |\n| Huber beta=2.0 | 14.248 | 25.072 | 2 |\n| Huber beta=4.0 | 14.285 | 25.009 | 2 |\n| MSE | 14.480 | 25.121 | 2 |\n\n**Monotone in beta on MAE, and flat on RMSE.** Every step toward MSE costs MAE\nand buys nothing; the 0.74 spread is six times the seed noise of section 10 and\nlarger than any margin this project has claimed. Note what this does *not* say:\nwithin the arm the loss moves MAE only. RMSE sits at 25.00-25.17 throughout.')
A('**The mechanism, and why it is an interaction rather than "Huber is better."**\nMSE has gradient $\\partial L/\\partial\\hat y = -2e$, so a sample with $e=50$\npulls a hundred times harder than one with $e=5$. Half-hourly SA1 price is\nspike-heavy, so that weighting is not a technicality. `--concat` widens the\nhead\'s input from $K$ to $2K$: the spatial arm has a block of capacity the\ntemporal arm does not, and MSE decides where it goes. It goes to the tail.\nHuber is quadratic near zero and linear beyond $\\beta$, so the tail stops\ndominating and the same block can serve the bulk instead.\n\nThe prediction that follows is testable and holds: **an arm with less spare\ncapacity should move less when the objective changes.**\n\n| arm | head inputs | MAE under MSE | under Huber | moved by |\n|---|---:|---:|---:|---:|\n| `vmd_price_res` | $K+1 = 9$ | 14.372 | 14.232 | **0.140** |\n| `nvmd_st` | $2K = 16$ | 14.480 | 14.038 | **0.442** |\n\nThe baseline moves a third as far, and its RMSE does not improve at all\n(26.427 -> 26.564). So the story is not that Huber is a better objective -- it\nmade `nvmd_temporal` worse, 14.163 -> 14.258 -- but that\n\n> for a representation with spare capacity, MSE\'s tail-dominated gradients\n> decide *where that capacity is spent*, and Huber changes the answer.\n\n**This is a mechanism consistent with every number above, not a verified\ncause.** What is established is the interaction: the objective moves the wide\narm three times as far as the narrow one, and in opposite directions on the two\nmetrics. Attributing that specifically to tail-versus-bulk allocation would need\nthe error decomposed by $|s|$ stratum, which has not been run.')
A('Against the residual-corrected baseline of section 6, **trained on the same\nobjective**, both metrics favour the spatial arm:\n\n| arm | MAE | RMSE | seeds |\n|---|---:|---:|---:|\n| `vmd_price_res` (MSE) | 14.372 | 26.427 | 3 |\n| `vmd_price_res` (Huber beta=1.0) | 14.232 | 26.564 | 2 |\n| **`nvmd_st` (Huber beta=1.0)** | **14.038** | **25.048** | 2 |\n| | **-1.4%** | **-5.7%** | vs the matched baseline |\n\nRe-running the baseline on the same loss was the outstanding objection and it\ncosts part of the margin: MAE goes from -2.3% against the MSE baseline to\n**-1.4%** against the matched one. RMSE goes the other way, -5.2% to **-5.7%**,\nbecause Huber does not help the baseline\'s RMSE at all.\n\nSo claim 5 as stated in the summary table -- *"once VMD gets its residual, it\ndoes not"* -- was true of the MSE runs and is not true of these. The spatial arm\nwas losing by 0.098; it now wins by 0.334, and the objective change is worth\n0.442, four and a half times the gap it had to close.')
A('**Three things stop this being final.**\n\n1. ~~**The baseline has not been re-run under the same objective.**~~\n   **Done.** Under Huber the baseline reaches 14.232 / 26.564 and the margin\n   becomes -1.4% MAE, -5.7% RMSE. The L1 pair is still running.\n2. **Two seeds against a seed noise of 0.116** (section 8). The two Huber seeds\n   are 13.970 and 14.105, a spread of 0.135, so the 0.334 margin is about three\n   times the noise. The L1 and Huber-0.5 rows are single-seed and cannot be read\n   yet.\n3. **Huber makes `nvmd_temporal` worse**, 14.163 -> 14.258. It is not a\n   uniformly better objective; it is the objective under which the wider\n   representation can show a bulk-interval gain. That conditionality is part of\n   the finding, not a tuning detail to be quietly dropped.')
A('''## 9. Existing literature, and what this project adds

Every row names the paper that already reports the general claim, so the
contribution column is what is left once that paper is granted. Rows marked
*pending* are not yet evidence.

| the general claim | already reported by | what this project adds |
|---|---|---|
| **VMD-based price forecasting leaks** | [VMDNet](https://arxiv.org/abs/2509.15394), Feng, Tao, Cartlidge & Zheng, EUSIPCO 2026 -- asserts leakage, fixes it with sample-wise VMD, does not measure it | **A measurement and a characterisation.** The window alone raises per-mode AR extrapolation error **6.6x** (4.288 to 15.907). And what leaks is identified, not just detected: a **625-parameter linear regression (MAE 4.29) beats a 5.7M-parameter CNN-BiLSTM (13.42)** on the leaked modes, so the modes carry a *linearly readable aggregate* of the future rather than a hard forecasting signal. Capacity is irrelevant because nothing is being forecast -- the answer is being read off. |
| | | **Two mechanisms separated.** Per-year VMD is both leaky *and* perfectly consistent across windows, because one decomposition serves the whole year. The literature reports the combined effect. Causal VMD removes the leak and loses the consistency; a fixed bank keeps consistency without the leak. The published 7.11 MAE reproduces exactly and degrades to ~10.7 once segmented. |
| **The decomposition can be made learnable** | [Adaptive Deep-Unfolded VMD](https://arxiv.org/html/2509.00703), Sept 2025 -- unrolls VMD's ADMM into a differentiable module with learnable per-mode bandwidths, per series, on traffic | **Not an unrolled solver.** There is no VMD objective, no ADMM, and no reconstruction loss anywhere in this model. It is a band-parameterised filter bank of **16 parameters** whose centres and bandwidths are **co-trained by the predictive objective alone**, inside the forward pass. The bands are whatever minimises forecast error, not whatever minimises a decomposition criterion. |
| | | **And the delivery path is a live question.** *Internal* means decomposing the raw window inside the forward pass, so the model sees $K$ waveforms of length $L$. *Precomputed* means decomposing offline per window, keeping only each mode's last sample, and stringing those endpoints into a series -- $K$ numbers per timestep. That is what the whole decompose-then-forecast literature does, and it discards most of the decomposition. Internal looked strictly better on two seeds; on three the ranges **overlap** (section 6), so this is posed, not settled. |
| **Spatial information helps price forecasting** | multi-price-zone STGNNs (Applied Energy 2024), R-vine copula spatial dependence (Int. J. Forecasting 2023), PJM LMP spatiotemporal deep learning | **The coupling is indexed by frequency band.** An STGNN learns one adjacency $A_{ij}$. `PerBandSpatialCoupling` learns $A_{ij}^{(k)}$, one $R \\times R$ matrix per band, over physical bands -- DC, 74.7 h, 26.3 h, 12.1 h, 6.3 h, 3.4 h, 1.8 h, 1.0 h. |
| **Channels can be decomposed jointly** | MVMD (Rehman & Aftab 2019), now standard on wind and marine panels, stated aim to preserve cross-source correlation *during* decomposition | **Coupling by parameterisation rather than by constraint.** MVMD ties channels by forcing shared centre frequencies and learns no cross-channel weight. Here the bands are shared and a learned matrix per band says how much of each other channel enters. |
| **Multi-scale decomposition plus a graph model** | Rawal & Ahmad 2024, wavelet/EMD then mutual-information graph then modified GCNN | **Coupling inside the decomposition, not after it.** Theirs is sequential: decompose, build a graph, run a GCNN. |
| **Multi-scale decomposition for EPF** | WT-SAE-LSTM, WPD-TCN-LSTM, MODWT+EMD+Seq2Seq, VMD-LSTM; 2025-26 adds VMD+attention, VMD+Transformer, V-MAF | **Nothing.** This is not a contribution and should not be claimed as one. V-MAF in particular fuses VMD features with channel attention; the difference from this design is that band indexing is structural rather than learned by an attention head. |
| **Spatial dependence is scale-dependent** | -- | *Pending.* This would be the claim worth making, and it is a claim about the market rather than about a model. See below: the evidence originally offered for it has been withdrawn. |

### The status of the last row

The obvious evidence -- read $A_k$ and see which driver sits in which band --
does not survive a second seed.

> Inspecting the trained couplings (`analysis/what_was_learned.py`): five of the
> eight bands correlate at about **-0.9** between seeds and three at about
> **+0.9**, for an overall **-0.409**. That is a sign symmetry, $A_k \\to -A_k$
> with $W \\to -W$ leaving the forecast unchanged, so no individual coupling's
> sign is identifiable. Band concentration is **0.205** against **0.125** for a
> channel spread evenly across all eight bands, and the largest coupling mass
> sits at the **1.0 h** band where the signal is mostly noise.
>
> What *is* stable is magnitude: per-channel total $|w|$ correlates **+0.936**
> across seeds, and both seeds rank `ramp_VIC1`, `ramp_SA1`, `demand_NSW1`
> first. **Which** channels are used reproduces; **at which band** does not.

The instrument has to be intervention, not inspection: zero a contribution and
measure the damage, which the sign symmetry cannot touch
(`analysis/band_ablation.py`, running). Three outcomes, written down in advance
so the result is not read backwards:

| $\\Delta L_{k,c}$ comes out | then |
|---|---|
| band-specific and stable across seeds | the claim stands, on intervention evidence rather than weight inspection -- a stronger footing than reading $A_k$ ever had |
| stable but **flat across bands** | the model was given the freedom and largely declined to use it. A clean negative result about scale-specificity, publishable as one |
| near zero everywhere | the exogenous block is redundant conditioning, consistent with taking 60-95% of head input variance for ~1% MAE. The spatial line closes |
''')
A("\n## 10. Caveats\n")
A("- One region, two years, one target, horizon 1. The horizon matters: "
  "`attic/RESULTS-superseded.md` section 11 records h=1 as **saturated** -- persistence scores "
  "14.40 against a best model of ~14.3 -- so everything above is measured "
  "where there is ~0.1 MAE of room. The spatial experiment tests h=6 for "
  "exactly this reason.")
A("- `vmd_panel` shares hyperparameters with arms that have 8 inputs rather "
  "than 215, so its collapse shows that naive per-channel concatenation "
  "hurts, not that joint decomposition is superior to multi-channel VMD.")
A("- MVMD (Rehman & Aftab 2019) extends VMD to joint multi-channel "
  "decomposition. \"VMD cannot use spatial information\" remains **not** a "
  "defensible sentence.")
A("- The forward-exogenous condition in the spatial experiment uses "
  "**reanalysis at the target time**. It is an upper bound on what a real "
  "forecast could deliver, and is the right measurement for \"is the "
  "information there\", not for \"what would this earn\".")
A("- Claims 1 and 2 of `attic/RESULTS-superseded.md` are untouched by any of this. They do not "
  "depend on epoch selection, on the residual channel, or on the delivery "
  "path.")

open("FINDINGS.md", "w").write("\n".join(L) + "\n")
print(f"FINDINGS.md written: four-arm {len(FOUR)}, zoo {len(STAB)}, "
      f"dose {len(DOSE)}, spatial {len(SPAT)}")
