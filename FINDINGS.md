# Decomposition for electricity price forecasting

*generated 2026-10-09 12:01*

## Summary

Two results hold independently of every protocol question below, and neither is touched by the later work. Both are in [`attic/FINDINGS-full.md`](attic/FINDINGS-full.md), which carries the long record and regenerates from the same result files.

- **The published gains from VMD price forecasting are leakage, not decomposition**, and what leaks is a linearly readable aggregate rather than a forecasting signal.
- **A band decomposition costs 10^3-10^5x less**, 0.1 s/year against 59-18,178 s/year.

Everything else is a margin against a baseline, and each is smaller than at least one protocol choice this project initially got wrong: the residual channel (0.237 MAE), the objective (0.442 on the widest arm), epoch selection (up to 0.170), the seed itself (0.116). [`PITFALLS.md`](PITFALLS.md) records those, and the ones found since.

The current state, measured at h=6 on 2021 with the control that was always missing -- the same head reading the raw window:

- **The band decomposition is worth -6.8%** (DM p=0.000, 33 of 48 half-hours). It splits in two: narrowing 37 channels to 16 is worth -3.5%, and making those 16 *bands* rather than a learned projection of the same width is worth a further -3.4%.
- **The per-band spatial coupling is worth -5.6%** (p=0.000, 32/48), and it survived a change of objective and of schedule.
- **The decomposition is not information.** A partition-of-unity bank is an invertible linear map: ridge on bands and ridge on the raw window agree to **0.0000 $/MWh**. It is a capacity limit plus a shortcut past the LSTM's sequential bottleneck.
- **Against a ridge the whole stack ties, and only after the objective was fixed** -- 37.79 against 37.77 on MAE, better by 1.6% on RMSE at a higher Jacobian exponent, better on negative prices, worse on high ones.

## Joint multi-region forecasting

Predict all five NEM regional prices at once rather than SA1 alone, at h=6 on the 2021 test year, one seed, 30 epochs. Every neural arm shares the window list, the objective and the seed. Diebold-Mariano is one-sided with an Andrews AR(1) bandwidth: the loss differential runs AC1 0.66 here, where the textbook zero-lag variance overstates z by an order of magnitude.

| model | MAE | RMSE | negative (n=3335) | calm (n=12030) | high (n=1939) | spike (n=115) |
|---|---:|---:|---:|---:|---:|---:|
| `arx_window` -- ridge on the raw 96 x 37 window | 37.77 | 254.3 | 55.40 | 16.26 | 63.51 | 1343.28 |
| `a0.25` -- bands + coupling, Jacobian weight 0.25 | 37.79 | 254.8 | 54.64 | 16.06 | 65.88 | 1348.32 |
| `a0.5` -- the same at 0.5 | 37.85 | 251.3 | 53.63 | 16.71 | 65.89 | 1319.59 |
| `f0.25` -- FiLM plus Jacobian 0.25 | 38.40 | 254.8 | 56.55 | 16.62 | 65.22 | 1339.48 |
| `a0.75` -- the same at 0.75 | 38.41 | 250.3 | 51.62 | 18.35 | 65.63 | 1295.05 |
| `joint_lstm_film` -- the same plus a FiLM gate | 38.45 | 255.0 | 58.34 | 16.30 | 64.73 | 1336.03 |
| `joint` -- bands + per-band coupling, LSTM head | 38.50 | 254.9 | 58.24 | 15.95 | 66.81 | 1348.54 |
| `gbt_pca` -- boosted trees, prices + 8 exogenous components | 38.92 | 255.0 | 64.60 | 15.33 | 62.95 | 1356.69 |
| `f0.5` -- FiLM plus Jacobian 0.5 | 38.94 | 253.8 | 57.99 | 16.38 | 69.58 | 1330.25 |
| `gbt_all` -- boosted trees, all 3,552 window values | 39.09 | 255.2 | 65.52 | 15.16 | 63.74 | 1359.45 |
| `var` -- VAR on the five price windows | 39.55 | 254.0 | 57.08 | 18.26 | 64.26 | 1341.21 |
| `global_linear` -- one pooled linear model across regions | 39.71 | 254.2 | 55.86 | 18.73 | 64.60 | 1345.64 |
| `ar_window` -- ridge, target window only | 39.80 | 253.4 | 56.24 | 18.93 | 64.01 | 1338.71 |
| `red16` -- a learned 16-channel projection, no bands | 39.84 | 255.2 | 60.55 | 17.19 | 66.55 | 1359.26 |
| `red8` -- a learned 8-channel projection | 39.93 | 255.5 | 57.44 | 17.39 | 70.93 | 1366.89 |
| `gbt_prices` -- boosted trees, five regional price windows | 40.25 | 255.8 | 64.41 | 16.71 | 65.95 | 1368.22 |
| `f0.75` -- FiLM plus Jacobian 0.75 | 40.59 | 254.8 | 58.97 | 18.36 | 74.83 | 1255.87 |
| `red37` -- a learned 37-channel projection | 40.75 | 255.6 | 64.20 | 17.90 | 63.69 | 1365.12 |
| `joint_nocouple` -- coupling frozen at the identity | 40.77 | 256.2 | 62.88 | 17.52 | 68.03 | 1372.21 |
| `a1.0` -- the same at 1.0 | 40.95 | 251.7 | 54.31 | 20.22 | 72.65 | 1287.89 |
| `single_SA1_price` -- single-task LSTM | 41.26 | 256.1 | 61.49 | 18.59 | 68.43 | 1367.65 |
| `panel_lstm` -- the same head on the raw window, no decomposition | 41.30 | 255.5 | 69.64 | 16.74 | 66.40 | 1366.24 |
| `gbt_own` -- boosted trees, target window only | 41.37 | 256.0 | 68.20 | 17.49 | 65.01 | 1362.10 |
| `naive_persist` -- the last observed price | 56.67 | 321.2 | 58.26 | 38.37 | 99.97 | 1195.70 |
| `naive_week` -- the same half-hour one week earlier | 67.83 | 350.1 | 71.61 | 48.79 | 100.45 | 1399.57 |

### What the decomposition is worth, and what it is

Against the control that matters -- the same head reading the raw window -- the decomposition is worth **-6.8%** (DM p=0.000). That control had never been run before; without it the table can only say which decomposition is least bad.

It splits cleanly into two mechanisms, and only one of them is regularisation:

| step | arm | MAE | effect |
|---|---|---:|---:|
| start | `panel_lstm`, 37 raw channels | 41.30 | |
| narrow to 16 channels | `red16`, a learned projection | 39.84 | -3.5% |
| make those channels bands | `joint` | 38.50 | -3.4% (p=0.000) |

**Capacity.** Width alone costs accuracy monotonically, and the validation curve says why: 37 raw channels peak at epoch 1 and degrade for 29 more, 16 channels train to epoch 18, 88 channels (`+exp`, the physical drivers given their own bands) peak at epoch 4-8. That half is regularisation.

**Coordinates.** The other half is not. Band masks are non-causal over the window, so `band_k(t)` carries the whole window at every step and the LSTM never integrates 96 of them. A ridge sees the window at once, has no such bottleneck, and gains exactly nothing: ridge on bands and ridge on the raw window return identical predictions, maximum absolute difference **0.0000 $/MWh**, correlation 1.000000. A partition-of-unity bank is an invertible linear map, so that is forced rather than observed -- and it is why no amount of decomposition can add information.

A tree agrees from the other side. Boosted trees on the target's own 96-step window score 41.37; on the same window as 8 bands x 96 steps they score 40.51, **4.4% worse**. Trees split on single features and cannot fold a rotation into a weight matrix, so they are the one learner where a genuinely better coordinate system would have to show up. It does not. The bands are not a better basis; they are a shortcut around a sequential bottleneck, and a capacity limit.

### The architecture, as it now stands

```
x : (B, C, L)          C = 37 channels, L = 96 half-hours
  |
  +-- StructuredSpectralNVMD, shared across channels
  |     Gaussian band masks normalised to a partition of unity,
  |     centres from a geometric prior (gap_k ~ 1.8^k). At K=8 and
  |     L=96 they land on 48h / 24h / 27h / 12.5h / 6.3h / 3.4h /
  |     1.8h / 1.0h -- the daily and half-daily cycles, and nothing
  |     above 48h, so the 168h weekly cycle is absent by construction.
  |
  v  modes : (B, C, K, L)
  |
  +-- own  = modes[:, target]                       (B, K, L)  lossless
  +-- exo  = sum_c A_k[target, c] * modes[:, c, k]  (B, K, L)
  |     A_k is one C x C matrix per band, identity-initialised, with
  |     the self term zeroed. Five targets means five live rows per
  |     band instead of one: on the 33-channel panel the single-target
  |     version left 97.1% of the coupling tensor without gradient.
  |
  v  cat([own, exo]) : (B, T, 2K, L)   T = 5 regions
  |
  +-- shared bidirectional LSTM, last hidden state
  +-- per-region embedding, then one shared MLP head
  v  (B, T)
```

Optional pieces, each an exact no-op at step 0 and each measured: a FiLM gate over the own bands (`+film`), a longer bank context (`+ctx336`), the physical drivers expanded into their own bands (`+exp`), a learned projection in place of the bank (`+red<N>`), and a seeded rather than identity coupling (`@<sd>`). Of these only the projection control changed a conclusion; the other three cost accuracy, which is reported below rather than omitted.

### The objective, and two corrections to it

Both corrections came from the same place -- FINDINGS already calls the objective this project's worst confound -- and neither is an architecture change. Together they are the only things in this whole sweep that improved the headline number.

**The shape.** Huber at beta=1.0 in standardised units is not a mild robustification of L1: measured on a trained model's residuals, **86.9%** of them fall inside |r| < 1, so the objective was quadratic for the bulk of the data while every reported number was an absolute error. The repo had already measured the ordering and it is monotone in beta -- L1 13.744, beta 0.5 13.909, beta 2.0 14.067, beta 4.0 14.234 -- so the default is now L1.

**The space.** Matching the shape still left the space mismatched. The loss lives in asinh coordinates and the metric is $/MWh, and dp/dz = w*cosh(asinh((p-c)/w)), so one unit of asinh-space error is worth 64 $/MWh in the calm band and 1,080 on a spike -- a factor of **22** that an unweighted L1 ignores. The model was therefore trained to ignore the rows that dominate the number being reported, and did: before the fix it was 1.8% better than a ridge in the calm band, which is 68.6% of rows, and 5.0% worse on negative prices.

`--w-jacobian` is the exponent on a per-sample |dp/dz| weight taken at the truth. 0 reproduces the old objective exactly and remains the ablation; 1 matches the metric to first order; between tempers it, because the spike gradients are 22x larger and asinh exists to stop exactly that from dominating.

### A Pareto frontier, not a ranking

Raising the exponent moves error out of the tails and into the bulk, monotonically, so no single point forecast wins all four regimes:

| alpha | negative | calm | high | spike | MAE | RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (none) | +5.1% (1.000) | -1.9% (0.095) | +5.2% (0.999) | +0.4% (0.719) | 38.50 | 254.9 |
| 0.25 | -1.4% (0.167) | -1.2% (0.178) | +3.7% (0.994) | +0.4% (0.852) | 37.79 | 254.8 |
| 0.5 | -3.2% (0.015) | +2.8% (0.969) | +3.7% (0.938) | -1.8% (0.129) | 37.85 | 251.3 |
| 0.75 | -6.8% (0.000) | +12.9% (1.000) | +3.3% (0.797) | -3.6% (0.075) | 38.41 | 250.3 |
| 1.0 | -2.0% (0.092) | +24.4% (1.000) | +14.4% (0.979) | -4.1% (0.212) | 40.95 | 251.7 |

Percentages are against the ridge, with the DM p in brackets. Two points are worth naming. **alpha=0.25** is the only arm that beats the ridge in both the negative band and the calm band at once, and it ties the ridge globally. **alpha=0.75** produces the only strongly significant segment win anywhere in this table, -6.8% on negative prices at p=0.000, and the best spike number of any trained model -- at the cost of a calm band 12.9% worse.

The global mean hides all of this, and it hides it for a measurable reason: 115 spike rows, 0.7% of the test set, carry 23% of the total absolute error, and every model including the ridge sits near 1,300 $/MWh on them. Negative prices are another 19.6% of rows and 29% of the error. A global MAE comparison is therefore settled mostly on rows where nothing has skill.

An oracle that routes each interval to the best arm for its realised regime reaches MAE 35.86 against the ridge's 37.77, -5.1%, and its assignment is interpretable -- negative to alpha=0.75, calm to the plain joint arm, high to the ridge, spike to persistence, which beats every trained model there because they all regress to the mean. That bound is not reachable: a classifier trained on the same window recalls 26% of negative prices and 3% of spikes, and hard routing on its output loses 2.4% against the ridge. Softening the routing into a probability weight recovers -1.1%, but a fixed weighting with no classifier at all reaches -0.9%, so the gain is diversification rather than regime detection, and the weights were not fitted out of sample.

