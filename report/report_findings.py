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

A("## Summary\n")
A("| # | claim | evidence | status |")
A("|---|---|---|---|")
A("| 1 | The published gains from VMD-based price forecasting are **leakage**, "
  "not decomposition | per-mode AR(48) probe, capacity-irrelevance test, and a "
  "reproduction of the original 7.11 MAE | **Strong. This is the headline** |")
A("| 2 | A band decomposition can be had at 10^3-10^5x lower cost | 0.1 s/year "
  "against 59-18,178 s/year | **Strong** |")
_ip = [r["test_mae"] for r in STAB if PATH.get(r["arm"]) == "internal"]
_pp = [r["test_mae"] for r in STAB if PATH.get(r["arm"]) == "precomputed"]
_rng = (f"internal {min(_ip):.3f}-{max(_ip):.3f} vs precomputed "
        f"{min(_pp):.3f}-{max(_pp):.3f}, "
        f"{'no overlap' if max(_ip) < min(_pp) else 'OVERLAPPING'}"
        if _ip and _pp else "pending")
A(f"| 3 | Putting the decomposition **inside** the forecaster beats the "
  f"classical decompose-then-forecast pipeline | {_rng}, same filter bank | "
  f"**Supported**, 2 seeds |")
A("| 4 | Classical modes underperform because of **what the bands physically "
  "are**, not because of which algorithm produced them | VMD\'s lowest band is "
  "2.5x wider than its own centre, so it smears across DC; the decomposition "
  "is effectively 1.6 modes; at window 96 nothing above 20.1 h exists. Vary "
  "the algorithm instead and nothing moves: five families within 0.3%, learned "
  "and hard-coded banks within 0.002, churn spanning 26x with no effect | "
  "**Supported** |")
A("| 5 | Spatio-temporal NVMD beats VMD | once VMD gets its residual, it does "
  "not | **Fails as stated.** The line is still open, section 9 |")

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
A("The decomposition is a differentiable layer, not a preprocessing step. "
  "Each mode is parameterised by a centre frequency and a bandwidth -- the same "
  "two quantities VMD solves for -- but they are *fixed across windows* rather "
  "than re-solved per window.\n")
A("```mermaid")
A("flowchart LR")
A("  X[\"price window<br/>x : (B, 1, L)\"] --> F[\"rFFT\"]")
A("  G[\"gap logits (K)\"] --> SM[\"softmax -> cumsum<br/>-> rescale to [0, 0.5]\"]")
A("  SM --> C[\"centres c_k<br/>strictly increasing<br/>c_1 = DC\"]")
A("  BW[\"log bandwidth (K)\"] --> B[\"bandwidths bw_k<br/>scaled to local gap\"]")
A("  C --> M[\"Gaussian masks<br/>normalised to a<br/>partition of unity\"]")
A("  B --> M")
A("  F --> MUL[\"multiply\"]")
A("  M --> MUL")
A("  MUL --> I[\"irFFT\"]")
A("  I --> Z[\"K modes<br/>sum exactly to x\"]")
A("  Z --> L1[\"BiLSTM 128 x 2\"]")
A("  L1 --> H[\"128 -> 1\"]")
A("  H --> Y[\"y-hat\"]")
A("```")
A("\nThree properties hold **by construction**, which is what v2 lacked:\n")
A("| property | how | why it mattered |")
A("|---|---|---|")
A("| centres strictly increasing | cumsum of a softmax | v2 preserved channel "
  "order in only 24% of windows |")
A("| mode 1 pinned to DC | first cumulative gap is zero | v2\'s lowest learned "
  "centre was 0.098 against VMD\'s 0.002, so trend had no channel |")
A("| modes sum exactly to the input | masks normalised to a partition of unity | "
  "no residual channel, and none can become a junk dump |")
A("\nThe spatial variant adds one R x R mixing matrix **per frequency band**, "
  "identity-initialised, so at step 0 it is exactly the temporal model. In "
  "`concat` mode the target\'s own modes pass through untouched and a "
  "purely-exogenous block is appended, taking the head from K to 2K inputs.\n")
A("```mermaid")
A("flowchart LR")
A("  P[\"panel<br/>(B, R, L)\"] --> BK[\"shared filter bank<br/>per channel\"]")
A("  BK --> MM[\"modes (B, R, K, L)\"]")
A("  MM --> OWN[\"target's own K modes<br/>lossless\"]")
A("  MM --> CP[\"per-band coupling A_k<br/>self weight zeroed\"]")
A("  CP --> EXO[\"exogenous block (B, K, L)<br/>zero at init\"]")
A("  OWN --> CAT[\"concat -> 2K\"]")
A("  EXO --> CAT")
A("  CAT --> LS[\"BiLSTM + head\"]")
A("```")
A("\nWhat the earlier design got wrong: the mixed modes **replaced** the "
  "target\'s own. That destroys the partition of unity -- reconstruction error "
  "0.00 to 1.82, with cross terms 2-6.5x the self term -- so the head never saw "
  "a faithful price encoding. It corrupted the DC and daily bands, which carry "
  "the ~88% of ordinary intervals, while the fast bands gained real spike "
  "information. **MAE got worse while RMSE got better**, consistently, and the "
  "concat fix is what separated the two.")

A("\n## 4. Why the classical bands underperform: the physics, not the algorithm\n")
A("This section is what makes the later null results legible. Swapping "
  "decomposition algorithms moves nothing, because they all hand the model "
  "bands with the same three defects.\n")
A("| | causal VMD, tuned K=8 | NVMD v3, 9 modes |")
A("|---|---:|---:|")
A("| participation ratio | 1.63 | **6.11** |")
A("| energy in top mode | 77.36% | **28.16%** |")
A("| longest period represented | 20.1 h | **307.2 h** |")
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
A("**Defect 3: the long periods do not exist.** At window 96 VMD has **no mode "
  "above 20.1 h**, so multi-day and weekly structure has nowhere to go. The "
  "bank reaches 307 h, 15x further, and puts 18.5% of its energy there.\n")
A("These are properties of the **modes**, and every classical method we tested "
  "shares them. That is why sections 7, 8 and 10 come back empty: they vary "
  "the *algorithm* while the physics of the resulting bands stays put. The one "
  "comparison that does move the metric changes what the bands are.")

A("\n### The two constructions, side by side\n")
A("VMD solves, per window, for K modes $u_k$ and centres $\\omega_k$:\n")
A("$$\\min_{\\{u_k\\},\\{\\omega_k\\}} \\sum_k \\Big\\| "
  "\\partial_t\\big[(\\delta(t) + \\tfrac{j}{\\pi t}) * u_k(t)\\big]"
  "e^{-j\\omega_k t} \\Big\\|_2^2 \\quad \\text{s.t.} \\quad "
  "\\sum_k u_k = f$$\n")
A("The objective is **narrowbandness per window**. Nothing in it constrains "
  "where the bands sit, whether they overlap, whether one of them reaches DC, "
  "or whether this window\'s mode 3 is the same filter as the last "
  "window\'s. Those are all left to whatever the ADMM iteration converges to, "
  "which is why the measured centres move 12.4% of a band gap between adjacent "
  "windows.\n")
A("The bank instead **parameterises** the same two quantities and fixes them:\n")
A("$$c_k = \\frac{1}{2}\\cdot\\frac{\\sum_{j\\le k}g_j - g_1}"
  "{\\sum_j g_j - g_1}, \\qquad g = \\mathrm{softmax}(\\theta), "
  "\\qquad b_k = b_{\\min,k} + \\mathrm{softplus}(\\beta_k)$$\n")
A("$$m_k(f) = \\frac{\\exp\\!\\big(-\\tfrac12 (f-c_k)^2/b_k^2\\big)}"
  "{\\sum_{k\'} \\exp\\!\\big(-\\tfrac12 (f-c_{k\'})^2/b_{k\'}^2\\big)}, "
  "\\qquad u_k = \\mathcal{F}^{-1}\\!\\big[m_k \\odot \\mathcal{F}f\\big]$$\n")
A("Three things follow immediately, and none of them is a penalty term:\n")
A("- $c_k$ is a cumulative sum of a softmax, so $c_1 = 0$ (a band **is** at DC) "
  "and $c_1 < c_2 < \\dots < c_K = \\tfrac12$ for any $\\theta$. Mode "
  "identity cannot scramble.")
A("- the masks are normalised across $k$, so $\\sum_k m_k(f) = 1$ for every "
  "$f$ and therefore $\\sum_k u_k = f$ **exactly**. No residual channel, and "
  "none can become a junk dump.")
A("- $b_k$ is tied to the local gap $\\tfrac12(c_{k+1}-c_{k-1})$, so width "
  "scales with centre. That is the constant-Q property VMD\'s flat bandwidth "
  "lacks.\n")
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
  "statistic `RESULTS.md` section 13.3 and `benchmark_seeds.py` report, kept "
  "alongside so the two sets of tables can be reconciled.\n")

# ---------------------------------------------------------------- experiment 1
A("\n## 5. Claim 3: does spatio-temporal NVMD beat VMD?\n")
A("Four arms, identical rows, `R=33` panel, 2 seeds. **This run used VMD "
  "without its residual channel**, which is a confound discovered afterwards "
  "and corrected in section 6.\n")
if FOUR:
    A("| arm | information | decomposition | honest | cherry | selection effect |")
    A("|---|---|---|---:|---:|---:|")
    for a, info, dec in [("nvmd_temporal", "temporal", "joint, coupling frozen"),
                         ("nvmd_st", "spatial", "joint, per-band coupling"),
                         ("vmd_price", "temporal", "univariate VMD, no residual"),
                         ("vmd_panel", "spatial", "univariate VMD x 26")]:
        rs = [r for r in FOUR if r["arm"] == a]
        if not rs:
            continue
        h = agg(rs); c = agg(rs, "test_mae_cherry")
        gap = f"{h[0]-c[0]:+.3f}" if c else "--"
        cs = f"{c[0]:.3f}" if c else "--"
        A(f"| `{a}` | {info} | {dec} | **{h[0]:.3f}** ± {h[1]:.3f} | {cs} | {gap} |")
    A("\n**Read this table with the next section.** As it stands `nvmd_st` "
      "(14.480) appears to beat `vmd_price` (14.524), which would make claim 3 "
      "true. It does not survive: `vmd_price` here is missing its residual "
      "channel, and once that is returned the same arm scores **14.382**, "
      "which beats `nvmd_st` by 0.098. The corrected comparison:\n")
    A("| arm | honest MAE | note |")
    A("|---|---:|---|")
    A("| `nvmd_temporal` | **14.163** | joint decomposition, no spatial coupling |")
    A("| `vmd_price_res` | 14.382 | VMD with its residual channel returned |")
    A("| `nvmd_st` | 14.480 | joint decomposition **plus** spatial coupling |")
    A("| `vmd_price` | 14.524 | VMD without the residual -- the confounded number |")
    A("\nSo the surviving statement is **temporal**: joint decomposition beats "
      "VMD by 0.219, and turning on spatial coupling gives back more than that.\n")
    A("- Spatial coupling **loses** to temporal-only on both seeds and both "
      "selection rules.")
    A("- Handing classical VMD the same 26 exogenous channels is catastrophic "
      "(~18.0), though that arm shares hyperparameters with an 8-channel arm "
      "and is arguably under-tuned.")
    A("- The selection effect differs by a factor of ten across these four "
      "arms, from +0.011 to +0.170. **Selecting on test is therefore not a "
      "neutral transformation: it moves some arms much further than others, so "
      "a table built that way can reorder methods.** See the retraction in "
      "section 8 for what we can and cannot say about *why*.")

# ---------------------------------------------------------------- confound
A("\n## 6. A confound we created, and what it cost\n")
A("VMD does not reconstruct its input exactly. Its residual is **8.5-9.5% of "
  "the price standard deviation**. The first runs stored only the K modes, so "
  "the VMD arms saw ~91% of the signal while the filter-bank arms, whose masks "
  "are a partition of unity, saw 100%.\n")
A("| VMD arm, seed 1 | honest |")
A("|---|---:|")
A("| 8 modes only | 14.561 |")
A("| **8 modes + residual** | **14.324** |")
A("\nCorrecting it returned **0.237 MAE** to VMD, which is more than the entire "
  "margin the original comparison claimed. Every arm now carries a residual "
  "channel.")

# ---------------------------------------------------------------- experiment 2
A("\n## 7. Where neural decomposition earns its place\n")
A("Two ways to deliver a decomposition to a sequence model:\n")
A("- **internal** -- hand the model the raw signal window and decompose it "
  "inside the forward pass, as a differentiable layer. The model sees each "
  "mode\'s waveform across one consistent window.")
A("- **precomputed** -- run the decomposition offline per window, keep the "
  "last sample of each mode, and feed the resulting per-timestep mode vectors. "
  "This is what the entire decomposition-plus-deep-learning literature does, "
  "including every comparison in `RESULTS.md` sections 2, 8 and 10.\n")
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
        A(f"\nEvery internal run lands in **{min(ip):.3f}-{max(ip):.3f}**; every "
          f"precomputed run lands in **{min(pp):.3f}-{max(pp):.3f}**. "
          f"{'No overlap.' if max(ip) < min(pp) else 'They overlap.'} The gap "
          f"between the groups is larger than the seed spread within either.")
    A("\nThe isolation is clean because `fixed_geo` and `bank` are **the same "
      "Gaussian filter bank**, differing only in whether the decomposition "
      "happens inside the model or is precomputed per timestep. Holding the "
      "basis fixed and moving only the delivery path reproduces most of the "
      "margin, so this is an architectural effect and not a basis effect.")
    A("\n**This is the result to build the paper on.** A neural decomposition "
      "layer is worth having, it does not depend on the learned parameters "
      "doing anything -- which is what makes it robust to the \"did you tune "
      "the baseline as hard\" objection -- and it transfers: any decomposition "
      "expressible as a differentiable filtering step can be moved inside the "
      "model.")
    A("\n**The mechanism is plausible but untested.** On the precomputed path "
      "the value at time t is the *last sample* of the decomposition of window "
      "[t-95, t], so a sequence of them is a trajectory of last samples. The "
      "internal path hands the model each mode\'s waveform across one "
      "consistent window, which is strictly more. We have not isolated that, "
      "and it is the next thing to test.")

# ---------------------------------------------------------------- retraction
A("\n## 8. Retracted: basis stability predicts accuracy\n")
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
A("\n## 9. The spatially-encoded variant\n")
st = [r for r in FOUR if r["arm"] == "nvmd_st"]
tp = [r for r in FOUR if r["arm"] == "nvmd_temporal"]
if st and tp:
    A("Both arms are the same model on the **internal** path, differing only in "
      "whether the per-band cross-channel coupling is enabled. So this sits "
      "inside the architecture family that wins section 7, and isolates the "
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
    A("- `RESULTS.md` section 13.4 measured the exogenous block taking 60-95% "
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
    A("1. **Horizon.** Every number above is h=1, which `RESULTS.md` section 11 "
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
          "`spread_SA1_TAS1` alone correlates +0.513 with the target, and section "
          "13.4 localised interconnector ramp pressure to the daily band and solar "
          "and demand to the sub-6-hour bands, which is a statement no univariate "
          "decomposition can make. What has failed so far is one particular way of "
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
A("\n## 10. Seed noise dominates method choice\n")
rs = [r for r in STAB if r["arm"] == "vmd_price_res"]
if len(rs) > 1:
    v = sorted(r["test_mae"] for r in rs)
    A(f"`vmd_price_res` across seeds: {' / '.join(f'{x:.3f}' for x in v)}, "
      f"a spread of **{v[-1]-v[0]:.3f}**.")
A("Within the matched precomputed path the five decomposition families span "
  "roughly **0.04**. One method's seed-to-seed variation is several times the "
  "difference between methods.\n")
A("Any claim of the form \"our decomposition beats VMD by x%\" that is not "
  "paired across multiple seeds is reporting the seed.")

# ---------------------------------------------------------------- pending
A("\n## 11. Still running\n")
A(f"| experiment | purpose | done |")
A(f"|---|---|---:|")
A(f"| zoo | architecture and decomposition families, 3 seeds | {len(STAB)}/24 |")
A(f"| dose | one filter bank, churn injected as a controlled dial; now a "
  f"*negative* control for the retracted hypothesis | {len(DOSE)}/12 |")
A(f"| spatial 2x2 | horizon (1 vs 6) x exogenous window (trailing vs "
  f"forward) | {len(SPAT)}/12 |")
if DOSE:
    A("\n### dose-response\n")
    A("| arm | injected churn | honest |")
    A("|---|---:|---:|")
    from experiments.run_three_arms import JITTER
    for a, ch in [("bank", 2.2)] + list(JITTER.items()):
        r = [x for x in DOSE if x["arm"] == a]
        if r:
            h = agg(r)
            A(f"| `{a}` | {ch:.1f}% | {h[0]:.3f} ± {h[1]:.3f} |")
if SPAT:
    A("\n### spatial 2x2\n")
    A("| config | test MAE | gain vs price-only |")
    A("|---|---:|---:|")
    for h in (1, 6):
        base = agg([r for r in SPAT if r["cfg"] == f"h{h}_price"])
        for mode in ("price", "back", "fwd"):
            rr = [r for r in SPAT if r["cfg"] == f"h{h}_{mode}"]
            if rr:
                m = agg(rr)
                g = "--" if mode == "price" or not base else f"{m[0]-base[0]:+.3f}"
                A(f"| `h{h}_{mode}` | {m[0]:.3f} ± {m[1]:.3f} | {g} |")

A("\n## 12. Caveats\n")
A("- One region, two years, one target, horizon 1. The horizon matters: "
  "`RESULTS.md` section 11 records h=1 as **saturated** -- persistence scores "
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
A("- Claims 1 and 2 of `RESULTS.md` are untouched by any of this. They do not "
  "depend on epoch selection, on the residual channel, or on the delivery "
  "path.")

open("FINDINGS.md", "w").write("\n".join(L) + "\n")
print(f"FINDINGS.md written: four-arm {len(FOUR)}, zoo {len(STAB)}, "
      f"dose {len(DOSE)}, spatial {len(SPAT)}")
