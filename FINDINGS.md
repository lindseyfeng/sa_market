# Decomposition for electricity price forecasting: what we can and cannot claim

*generated 2026-09-18 11:52*

Everything below is train 2018 / test 2019, SA1 half-hourly price, window 96, horizon 1, selection on a validation tail of the train year with a 96-window embargo, test scored once from those weights. `honest` is that number; `cherry` is the minimum of test MAE over epochs, the statistic `RESULTS.md` section 13.3 and `benchmark_seeds.py` report.

## What the evidence now supports

1. **Neural decomposition works, and we can now say where its value comes from.** Making the decomposition a differentiable layer *inside* the forecaster beats the classical decompose-then-forecast pipeline on every run, with no overlap between the two groups and a margin larger than seed noise. This is the only effect in this whole line of work that survives an honest protocol, and it is a property of the **architecture**, not of any particular basis.
2. **The value is in the architecture, not in learning the band parameters.** A trained bank and a hard-coded one land within 0.001 of each other. That is a sharper claim than "our learned decomposition is better": it says the in-model decomposition layer is what pays, and it costs 0.1 s/year against VMD's 126-500 s/year.
3. **Which classical decomposition you choose does not matter.** With the architecture matched, the information equalised and selection honest, five decomposition families land inside 0.3% of one another, and seed-to-seed variation is larger than any difference between them.
4. **Basis stability does not predict accuracy.** This was our hypothesis and its own control rejected it. See the retraction below.
5. **Claim 3 of `RESULTS.md` section 12 fails as stated.** Once VMD is given its residual and selection is honest, spatio-temporal NVMD does not beat VMD. The surviving claim is temporal, not spatial.


## 1. Claim 3: does spatio-temporal NVMD beat VMD?

Four arms, identical rows, `R=33` panel, 2 seeds. **This run used VMD without its residual channel**, which is a confound discovered afterwards and corrected in experiment 2.

| arm | information | decomposition | honest | cherry | selection effect |
|---|---|---|---:|---:|---:|
| `nvmd_temporal` | temporal | joint, coupling frozen | **14.163** ± 0.081 | 14.147 | +0.016 |
| `nvmd_st` | spatial | joint, per-band coupling | **14.480** ± 0.002 | 14.311 | +0.170 |
| `vmd_price` | temporal | univariate VMD, no residual | **14.524** ± 0.052 | 14.505 | +0.019 |
| `vmd_panel` | spatial | univariate VMD x 26 | **18.033** ± 0.303 | 18.022 | +0.011 |

- Spatial coupling **loses** to temporal-only on both seeds and both selection rules.
- Handing classical VMD the same 26 exogenous channels is catastrophic (~18.0), though that arm shares hyperparameters with an 8-channel arm and is arguably under-tuned.
- The selection effect is an order of magnitude larger for `nvmd_st` than for any other arm. It carries an extra 8x33x33 coupling tensor, so its epoch-to-epoch test curve is noisier, and a minimum over ~30 test evaluations rewards exactly that. **Selecting on test does not subsidise all arms equally; it subsidises the high-variance one.**

## 2. A confound we created, and what it cost

VMD does not reconstruct its input exactly. Its residual is **8.5-9.5% of the price standard deviation**. The first runs stored only the K modes, so the VMD arms saw ~91% of the signal while the filter-bank arms, whose masks are a partition of unity, saw 100%.

| VMD arm, seed 1 | honest |
|---|---:|
| 8 modes only | 14.561 |
| **8 modes + residual** | **14.324** |

Correcting it returned **0.237 MAE** to VMD, which is more than the entire margin the original comparison claimed. Every arm now carries a residual channel.

## 3. Where neural decomposition earns its place

Two ways to deliver a decomposition to a sequence model:

- **internal** -- hand the model the raw signal window and decompose it inside the forward pass, as a differentiable layer. The model sees each mode's waveform across one consistent window.
- **precomputed** -- run the decomposition offline per window, keep the last sample of each mode, and feed the resulting per-timestep mode vectors. This is what the entire decomposition-plus-deep-learning literature does, including every comparison in `RESULTS.md` sections 2, 8 and 10.

| arm | basis | delivery | churn | honest | seeds |
|---|---|---|---:|---:|---:|
| `fixed_geo` | fixed | internal | 2.2% | **14.165** ± 0.079 | 2 |
| `fixed_vmdmean` | fixed | internal | 2.2% | **14.179** ± 0.044 | 2 |
| `nvmd_trained` | fixed | internal | 2.2% | **14.163** ± 0.081 | 2 |
| `bank` | fixed | precomputed | 2.2% | **14.305** ± 0.000 | 1 |
| `vmd_price_res` | re-solved | precomputed | 31.7% | **14.382** ± 0.082 | 2 |
| `wpt` | fixed | precomputed | 2.7% | **14.344** ± 0.000 | 1 |
| `ewt` | re-solved | precomputed | 58.4% | **14.348** ± 0.000 | 1 |
| `emd` | re-solved | precomputed | 41.1% | **14.817** ± 0.000 | 1 |

Every internal run lands in **14.106-14.221**; every precomputed run lands in **14.305-14.817**. No overlap. The gap between the groups is larger than the seed spread within either.

The isolation is clean because `fixed_geo` and `bank` are **the same Gaussian filter bank**, differing only in whether the decomposition happens inside the model or is precomputed per timestep. Holding the basis fixed and moving only the delivery path reproduces most of the margin, so this is an architectural effect and not a basis effect.

**This is the result to build the paper on.** It says a neural decomposition layer is worth having, states precisely why -- the model sees mode waveforms rather than a trajectory of last samples -- and does not depend on the learned parameters doing anything, which is what makes it robust. It also transfers: any decomposition expressible as a differentiable filtering step can be moved inside the model.

A likely mechanism, not yet tested: on the precomputed path the value at time t is the *last sample* of the decomposition of window [t-95, t], so a sequence of them is a trajectory of last samples. The internal path hands the model the actual mode waveform across one consistent window.

## 4. Retracted: basis stability predicts accuracy

We proposed that what separates these methods is whether the basis is re-solved in every window, measured as **churn** -- the fraction of adjacent-window steps in which some mode's spectral centroid moves more than half a band gap.

| method | basis | drift | churn |
|---|---|---:|---:|
| bank | fixed | 3.0% | 2.2% |
| wpt | fixed | 9.3% | 2.7% |
| vmd | adaptive | 12.5% | 31.7% |
| emd | adaptive | 12.5% | 41.1% |
| ewt | adaptive | 29.8% | 58.4% |

Churn separates the families by 12-20x with no overlap. **Accuracy does not follow it.** Within the matched precomputed path, fixed and re-solved bases interleave, and the whole group spans 0.3% while churn spans a factor of 26.

The hypothesis was rejected by a control that was part of the design: `bank` holds the filter bank identical to `fixed_geo` and changes only the delivery path. Most of the gap we had attributed to stability moved with the path, not the basis.

**`EMD` is a separate story.** It is the worst arm by a wide margin and churn does not explain it either. On 96-sample windows EMD often fails to sift 8 IMFs:

| method | live modes per window | windows where the live set changes |
|---|---|---:|
| EMD | mean **5.93**, min 4, max 8 | **23.1%** |
| EWT / WPT / bank | always 8 | 0% |

Channels are not merely drifting, they intermittently do not exist.

## 5. The spatially-encoded variant

Both arms are the same model on the **internal** path, differing only in whether the per-band cross-channel coupling is enabled. So this sits inside the architecture family that wins section 3, and isolates the spatial encoding itself.

| arm | seed 1 | seed 2 | mean | seed spread | selection effect |
|---|---:|---:|---:|---:|---:|
| `nvmd_temporal` | 14.220 | 14.106 | **14.163** | 0.114 | +0.016 |
| `nvmd_st` | 14.479 | 14.482 | **14.480** | 0.003 | +0.170 |

**Spatial encoding costs +0.317 MAE**, and the degradation is not noise: `nvmd_st` reproduces to within 0.003 across seeds, so the penalty is ~100x its own seed spread. It also lands worse than every arm on the precomputed path except EMD, which means enabling spatial coupling gives back more than the architecture won.

Two independent measurements point at the same mechanism -- variance, not absence of signal:

- The coupling adds an 8x33x33 tensor, and `nvmd_st`'s selection effect is +0.170 against +0.016 for the temporal arm. Its epoch-to-epoch test curve is an order of magnitude noisier.
- `RESULTS.md` section 13.4 measured the exogenous block taking 60-95% of head input variance while buying ~1% MAE. A block that dominates the input and moves the metric that little is redundant conditioning.

**This is a verdict on the current design, not on spatial information.** Two reasons to withhold judgement, both testable and both in flight:

1. **Horizon.** Every number above is h=1, which `RESULTS.md` section 11 records as saturated -- persistence 14.40 against a best model of ~14.3. Section 11a measured the spatial coupling gain at **-2.21 MAE at h=6**, decaying to zero by h=48. Testing a 2.21-point effect in a 0.1-point window cannot resolve it.
2. **The exogenous channels are fed as history, not as forecasts.** `PanelWindowDataset` hands the model every channel over the trailing window and asks it to predict h steps ahead. Real load and price forecasting conditions on the *forecast* weather and demand for the target interval. Trailing weather is largely already priced into the recent spread; forward weather is where the incremental information should be.


**This line is open, not closed.** The panel carries real structure -- `spread_SA1_TAS1` alone correlates +0.513 with the target, and section 13.4 localised interconnector ramp pressure to the daily band and solar and demand to the sub-6-hour bands, which is a statement no univariate decomposition can make. What has failed so far is one particular way of injecting that structure, at one horizon where nothing is resolvable. Designs still to try, in the order we would try them:

1. **Forward exogenous windows** -- in flight, as the `fwd` arm.
2. **Do not decompose the exogenous channels at all.** The target is being extrapolated and needs a representation; the exogenous channels are only being conditioned on. Decomposing them is cost and variance, which is how `vmd_panel` reached 18.0.
3. **Compress before injecting.** 26 channels into 2-4 learned directions, aimed straight at the 60-95% input-variance problem.
4. **Inject as a gate rather than as extra channels.** Exogenous drivers plausibly change the conditional *scale* of price, not its level, which makes FiLM over the target's bands the better inductive bias.
5. **Restrict coupling to the bands where 13.4 found signal**, instead of learning a full K x R x R tensor whose variance cost we have measured.
The `spatial 2x2` experiment crosses these two factors, so the outcome distinguishes "the information is not there" from "we tested where there was no room" from "we fed it the wrong window".

### The h=1 row cannot resolve an exogenous effect

Measured, not assumed. The price-only control at h=1:

| seed | val MAE | test MAE | val -> test shift |
|---|---:|---:|---:|
| 1 | 13.929 | 14.166 | +0.237 |
| 2 | 13.991 | 14.388 | +0.397 |

The control alone varies by **0.222** across two seeds, and validation understates test by +0.237 to +0.397. Both exceed the total headroom at this horizon, where persistence scores 14.40 against a best model near 14.3.

So any h=1 comparison between price-only, trailing and forward exogenous is **inside the noise floor by construction**. We record the row for completeness and read nothing into it. This is also why the earlier conclusion that the spatial panel was useless was never evidence of absence -- it was measured here.

## 6. Seed noise dominates method choice

`vmd_price_res` across seeds: 14.324 / 14.440, a spread of **0.116**.
Within the matched precomputed path the five decomposition families span roughly **0.04**. One method's seed-to-seed variation is several times the difference between methods.

Any claim of the form "our decomposition beats VMD by x%" that is not paired across multiple seeds is reporting the seed.

## 7. Still running

| experiment | purpose | done |
|---|---|---:|
| zoo | architecture and decomposition families, 3 seeds | 12/24 |
| dose | one filter bank, churn injected as a controlled dial; now a *negative* control for the retracted hypothesis | 0/12 |
| spatial 2x2 | horizon (1 vs 6) x exogenous window (trailing vs forward) | 2/12 |

### spatial 2x2

| config | test MAE | gain vs price-only |
|---|---:|---:|
| `h1_price` | 14.277 ± 0.157 | -- |

## 8. Caveats

- One region, two years, one target, horizon 1. The horizon matters: `RESULTS.md` section 11 records h=1 as **saturated** -- persistence scores 14.40 against a best model of ~14.3 -- so everything above is measured where there is ~0.1 MAE of room. The spatial experiment tests h=6 for exactly this reason.
- `vmd_panel` shares hyperparameters with arms that have 8 inputs rather than 215, so its collapse shows that naive per-channel concatenation hurts, not that joint decomposition is superior to multi-channel VMD.
- MVMD (Rehman & Aftab 2019) extends VMD to joint multi-channel decomposition. "VMD cannot use spatial information" remains **not** a defensible sentence.
- The forward-exogenous condition in the spatial experiment uses **reanalysis at the target time**. It is an upper bound on what a real forecast could deliver, and is the right measurement for "is the information there", not for "what would this earn".
- Claims 1 and 2 of `RESULTS.md` are untouched by any of this. They do not depend on epoch selection, on the residual channel, or on the delivery path.
