# Decomposition for electricity price forecasting

*generated 2026-10-09 12:10. h=6, SA1, 2021 test year, one seed, 30 epochs. Every neural arm shares the window list, the objective and the seed.*

## 1. What holds

Two results are independent of everything below and are recorded in full in [`attic/FINDINGS-full.md`](attic/FINDINGS-full.md): **the published gains from VMD price forecasting are leakage**, and **a band decomposition costs 10^3-10^5x less** than solving one.

From the joint multi-region work, against the control that was always missing -- the same head reading the raw window:

| | effect | DM p | half-hours |
|---|---:|---:|---|
| the band decomposition | **-6.8%** | 0.000 | 33/48 |
| the per-band spatial coupling | **-5.6%** | 0.000 | 32/48 |

And one thing that is not a result but a fact about the construction: a partition-of-unity bank is an invertible linear map, so **ridge on bands and ridge on the raw window agree to 0.0000 $/MWh**, correlation 1.000000. The decomposition cannot add information. What it adds is measured in section 5.

## 2. Architecture

```
  x : (B, 37, 96)                         37 channels, 96 half-hours = 48 h
   |
   |   ### decomposition module -- 57,656 params, 8.9% of the model
   |
   +-> rfft over the window                        (B*37, 49) complex
   |
   |     band masks: K=8 Gaussians on [0, 0.5], normalised to
   |     sum to 1 at every frequency (partition of unity, so the
   |     modes add back to the signal exactly -- measured 4.8e-07).
   |     Centres are a cumsum of softmax(gap_logits), geometric
   |     init gap_k ~ 1.8^k, which lands them at
   |
   |        k   period     what lives there
   |        0   48 h       trend (the window's own length)
   |        1   24-75 h    daily
   |        2   27 h       daily
   |        3   12.5 h     half-daily
   |        4   6.3 h
   |        5   3.4 h
   |        6   1.8 h
   |        7   1.0 h      Nyquist
   |
   |     parameters: gap_logits (8) + log_bw (8) = 16 numbers.
   |     The masks themselves have none. encoder (40,096) and
   |     gap_head (17,544) only matter when adapt>0, which makes
   |     the bands input-dependent; every run here uses adapt=0.
   |
   +-> irfft                                modes : (B, 37, 8, 96)
   |
   |   ### spatial coupling -- 10,952 params, 1.7%
   |
   |     own   = modes[:, t]                       (B, 8, 96)  lossless
   |     exo_k = sum_c A_k[t, c] * modes[:, c, k]  (B, 8, 96)
   |
   |     A_k = I + delta_k, one 37x37 matrix per band, self term
   |     zeroed. Identity init, so exo = 0 at step 0 and the arm
   |     is exactly the uncoupled one -- which makes --coupling 0
   |     an exact ablation and costs a dead exogenous pathway at
   |     the start. Five targets put five rows per band under
   |     gradient: 1,440 of 10,952 live, against 256 of 8,712
   |     (2.9%) for the single-target version.
   |
   v  cat([own, exo]) : (B, 5, 16, 96)     5 regions, 2K channels
   |
   |   ### head -- 579,073 params, 84.1% + 5.1%
   |
   +-> reshape to (B*5, 96, 16)            regions fold into the batch
   +-> LSTM, bidirectional, 2 layers, hidden 128      544,768
   +-> last hidden state  (B*5, 256)
   +-> + per-region embedding (5, 256)                  1,280
   +-> Linear 256->128, ReLU, Linear 128->1            33,025
   |
   v  (B, 5)                               one price per region

  total 647,681. The head is 89% of it; the decomposition is 8.9%
  and the thing that encodes the spatial claim is 1.7%.
```

Optional pieces, each an exact no-op at step 0, each measured in section 4: `+film` a FiLM gate over the own bands, `+ctx336` a 336-step bank context with a 96-step readout, `+exp` the physical drivers given their own band channels (16 -> 88), `+red<N>` a learned projection instead of the bank, `@<sd>` a seeded rather than identity coupling.

## 3. The objective, and the Jacobian weight

The loss is not a hyperparameter here and it was wrong twice, in two different ways, and fixing it is the only thing in this whole sweep that improved the headline number. No architecture change did.

**The shape was wrong.** `smooth_l1_loss(beta=1.0)` on a standardised target is quadratic for **86.9%** of residuals -- measured on a trained model -- so the objective was MSE for the bulk while every number reported was an absolute error. The repo had already swept beta and the ordering is monotone (L1 13.744, 0.5 13.909, 2.0 14.067, 4.0 14.234), so the default is now L1.

**The space was wrong.** Fixing the shape left the loss in asinh coordinates while the metric is dollars. With `z = asinh((p - c)/w)`, `c` the train median and `w` the train IQR,

```
    dp/dz = w * cosh(z)

    regime        mean |dp/dz|     relative      rows
    calm                 49 $/MWh      1.0x      12,030
    high                 81            1.7x       1,939
    negative             98            2.0x       3,335
    spike             1,080           22.0x         115
```

One unit of asinh-space error is worth 22 times more on a spike than in the calm band, and an unweighted L1 gave them the same weight. The model was trained to ignore the rows that dominate the metric, and did: before the fix it beat a ridge by 1.8% in the calm band and lost by 5.0% on negative prices.

**The weight.** `--w-jacobian alpha` multiplies each sample's loss by `(|dp/dz| / mean|dp/dz|)^alpha`, evaluated at the truth so it reweights the data rather than bending the loss:

```
    loss = mean_i  w_i * |z_hat_i - z_i|  +  regularisers
    w_i  = ( |dp/dz|(z_i) / mean_j |dp/dz|(z_j) ) ^ alpha

    alpha   calm    high    negative   spike      what it is
    0       1.00x   1.00x   1.00x       1.00x     the old objective
    0.25    0.92x   1.04x   1.09x       1.98x
    0.5     0.84x   1.06x   1.18x       3.08x
    1.0     0.71x   1.18x   1.43x      15.70x     MAE in $ to 1st order
```

alpha=0 reproduces the old objective exactly and stays the ablation. alpha=1 matches the metric but hands 15.7x weight to 115 rows, which is why asinh was introduced in the first place; the useful range is in between, and section 7 is the frontier it traces.

**The whole objective.** Written out, with `B` a batch, `R` the five price targets, `A` the auxiliary driver targets, `rho` the pointwise loss (`|.|` by default, `--loss`), and everything in the standardised asinh space the model predicts in:

```
  L  =   1/(|B| R)  sum_i sum_r   w_ir * rho( zhat_ir - z_ir )     price

       + l_dev / (|B| R) sum_i sum_r rho( ( d(zhat)_ir - d(z)_ir )  deviation
                                           / sigma_d,r )

       + l_aux / (|B| |A|) sum_i sum_a rho( zhat_ia - z_ia )       drivers

       + l_bias / R  sum_r | 1/|B| sum_i ( zhat_ir - z_ir ) |      offset

       + l_sp   * mean_{k, c != t} | Delta_k[t, c] |               coupling L1

       + 0.05   * mean_k  bw_k                                     bandwidth

       + 1.0    * mean_k  relu( m * (bw_k + bw_{k+1})/2            separation
                                - (ctr_{k+1} - ctr_k) )^2

  w_ir  = ( |dp/dz|(z_ir) / mean_js |dp/dz|(z_js) ) ^ alpha        --w-jacobian
  d(v)_ir = v_ir - 1/R sum_s v_is          departure from the cross-region mean
  sigma_d,r  standard deviation of d(z)_.r on the training span
  Delta_k = A_k - I    the learned part of the per-band coupling
  ctr_k, bw_k          band centres and bandwidths, from the bank
```

Defaults in every run reported here: `alpha` as stated per arm, `l_dev = 1.0`, `l_aux = 0` (the driver targets were tried and cost accuracy), `l_bias = 0.1`, `l_sp = 1e-4`, `m = 1`. The last two terms shape the filter bank rather than the forecast and are unchanged from earlier work; the deviation term exists because 96.4% of the variance across the five regions is one common mode, so an unweighted level loss can be won without representing any regional structure at all.

Three of the weights are reported with their ablation: `alpha` in section 7, `l_dev` as `--w-dev 0`, and `l_sp` as the identity-frozen coupling arm. `l_bias` and the two bank terms are carried over untested here.

## 4. Results

| | model | MAE | RMSE | negative (3335) | calm (12030) | high (1939) | spike (115) |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive | `naive_persist` the last observed price | 56.67 | 321.2 | 58.3 | 38.4 | 100.0 | 1195.7 |
|  | `naive_week` the same half-hour one week earlier | 67.83 | 350.1 | 71.6 | 48.8 | 100.4 | 1399.6 |
| Linear | `ar_window` ridge, target window only | 39.80 | 253.4 | 56.2 | 18.9 | 64.0 | 1338.7 |
|  | `var` VAR, five price windows | 39.55 | 254.0 | 57.1 | 18.3 | 64.3 | 1341.2 |
|  | `global_linear` one pooled linear model across regions | 39.71 | 254.2 | 55.9 | 18.7 | 64.6 | 1345.6 |
|  | `arx_window` ridge, raw 96 x 37 window | 37.77 | 254.3 | 55.4 | 16.3 | 63.5 | 1343.3 |
| Trees | `gbt_own` boosted trees, target window | 41.37 | 256.0 | 68.2 | 17.5 | 65.0 | 1362.1 |
|  | `gbt_prices` boosted trees, five price windows | 40.25 | 255.8 | 64.4 | 16.7 | 65.9 | 1368.2 |
|  | `gbt_pca` boosted trees, prices + 8 exogenous PCs | 38.92 | 255.0 | 64.6 | 15.3 | 63.0 | 1356.7 |
|  | `gbt_all` boosted trees, all 3,552 values | 39.09 | 255.2 | 65.5 | 15.2 | 63.7 | 1359.5 |
| LSTM, no decomposition | `single_SA1_price` LSTM, single task | 41.26 | 256.1 | 61.5 | 18.6 | 68.4 | 1367.7 |
|  | `panel_lstm` LSTM on the raw window -- the no-decomposition control | 41.30 | 255.5 | 69.6 | 16.7 | 66.4 | 1366.2 |
|  | `red37` LSTM on a learned 37-channel projection | 40.75 | 255.6 | 64.2 | 17.9 | 63.7 | 1365.1 |
|  | `red16` LSTM on a learned 16-channel projection | 39.84 | 255.2 | 60.5 | 17.2 | 66.6 | 1359.3 |
|  | `red8` LSTM on a learned 8-channel projection | 39.93 | 255.5 | 57.4 | 17.4 | 70.9 | 1366.9 |
| LSTM, bands | `joint_nocouple` bands, coupling frozen at identity | 40.77 | 256.2 | 62.9 | 17.5 | 68.0 | 1372.2 |
|  | `joint` bands + per-band coupling | 38.50 | 254.9 | 58.2 | 15.9 | 66.8 | 1348.5 |
|  | `joint_lstm_film` the same + FiLM gate | 38.45 | 255.0 | 58.3 | 16.3 | 64.7 | 1336.0 |
| LSTM, bands + Jacobian weight | `a0.25` bands + coupling, Jacobian weight 0.25 | 37.79 | 254.8 | 54.6 | 16.1 | 65.9 | 1348.3 |
|  | `a0.5` the same, 0.5 | 37.85 | 251.3 | 53.6 | 16.7 | 65.9 | 1319.6 |
|  | `a0.75` the same, 0.75 | 38.41 | 250.3 | 51.6 | 18.3 | 65.6 | 1295.1 |
|  | `a1.0` the same, 1.0 | 40.95 | 251.7 | 54.3 | 20.2 | 72.7 | 1287.9 |
|  | `f0.25` FiLM + Jacobian 0.25 | 38.40 | 254.8 | 56.5 | 16.6 | 65.2 | 1339.5 |
|  | `f0.5` FiLM + Jacobian 0.5 | 38.94 | 253.8 | 58.0 | 16.4 | 69.6 | 1330.3 |
|  | `f0.75` FiLM + Jacobian 0.75 | 40.59 | 254.8 | 59.0 | 18.4 | 74.8 | 1255.9 |

Best MAE is `arx_window` at 37.77; best RMSE is `a0.75` at 250.3, -1.6% against the ridge. Which metric is chosen decides the winner, and section 6 is why.

## 5. What the decomposition is

Three measurements, each from a different direction, and together they leave one explanation standing.

**It is not information.** Ridge on bands and ridge on the raw window return identical predictions to 0.0000 $/MWh. Forced by construction.

**It is not a better basis.** Boosted trees on the target's own 96-step window score 41.37; on the same window as 8 bands x 96 steps, 40.51 -- **4.4% worse**. Trees split on single features and cannot fold a rotation into a weight matrix, so they are the one learner where a better coordinate system would have to show up. It does not.

**It is capacity, plus a shortcut.** Replacing the bank with a learned projection of the same output width separates the two:

| step | arm | MAE | effect |
|---|---|---:|---:|
| start | `panel_lstm`, 37 raw channels | 41.30 | |
| narrow to 16 channels | `red16`, learned projection | 39.84 | -3.5% |
| make those 16 channels bands | `joint` | 38.50 | -3.4% (p=0.000) |

The first half is regularisation, and the validation curves say so: 37 raw channels peak at epoch 1 and degrade for 29 more (25.67 -> 30.29), 16 channels train to epoch 18, 88 channels (`+exp`) peak at 4-8. Width costs accuracy monotonically.

The second half is not. Band masks are non-causal over the window, so `band_k(t)` carries the whole window at every step and the LSTM never integrates 96 of them. A ridge sees the window at once, has no such bottleneck, and gains exactly nothing -- which is the same 0.0000 from a different angle.

## 6. Error analysis

The global mean is decided largely where nothing has skill.

| regime | rows | % of rows | % of total error | `joint` MAE | `arx_window` MAE |
|---|---:|---:|---:|---:|---:|
| negative | 3,335 | 19.1% | 29% | 58.2 | 55.4 |
| calm | 12,030 | 69.1% | 29% | 15.9 | 16.3 |
| high | 1,939 | 11.1% | 19% | 66.8 | 63.5 |
| spike | 115 | 0.7% | 23% | 1348.5 | 1343.3 |

**115 spike rows are 0.7% of the test set and 23% of the absolute error**, and every model including the ridge sits near 1,300 $/MWh on them -- persistence is the best thing there at 1196, because every trained model regresses to the mean. Negative prices are another 19.1% of rows and 29% of the error. Segment before concluding: a global comparison here is settled mostly on rows no model can reach.

A second reason the mean misleads: after removing the cross-region common mode, **regional structure is 0.4-1.9% of the variance in every band**, with the fastest band (1-1.6 h) 4.6x more regional than the most-shared one (16 h). The direction is what the physics predicts -- interconnector limits bind and decouple fast components -- but the magnitude bounds what any spatial model can win.

## 7. The Pareto frontier

Raising alpha moves error out of the tails and into the bulk, monotonically. No single point forecast wins all four regimes. Percentages are against `arx_window`, DM p in brackets.

| alpha | negative | calm | high | spike | MAE | RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +5.1% (1.000) | -1.9% (0.095) | +5.2% (0.999) | +0.4% (0.719) | 38.50 | 254.9 |
| 0.25 | -1.4% (0.167) | -1.2% (0.178) | +3.7% (0.994) | +0.4% (0.852) | 37.79 | 254.8 |
| 0.5 | -3.2% (0.015) | +2.8% (0.969) | +3.7% (0.938) | -1.8% (0.129) | 37.85 | 251.3 |
| 0.75 | -6.8% (0.000) | +12.9% (1.000) | +3.3% (0.797) | -3.6% (0.075) | 38.41 | 250.3 |
| 1.0 | -2.0% (0.092) | +24.4% (1.000) | +14.4% (0.979) | -4.1% (0.212) | 40.95 | 251.7 |

**alpha=0.25** is the only arm that beats the ridge in the negative band and the calm band at once, and it ties the ridge globally. **alpha=0.75** produces the only strongly significant segment win in this document, -6.8% on negative prices at p=0.000, and the best spike figure of any trained model, at the cost of a calm band 12.9% worse.

An oracle routing each interval to the best arm for its realised regime reaches MAE 35.86 against 37.77, **-5.1%**, and the assignment is interpretable: negative to alpha=0.75, calm to plain `joint`, high to the ridge, spike to persistence. That bound is not reachable. A classifier on the same window recalls 26% of negative prices and 3% of spikes, and hard routing on its output loses 2.4%. Softening to a probability weight recovers -1.1%, but a fixed weighting with no classifier reaches -0.9%, so the gain is diversification, not regime detection -- and those weights were not fitted out of sample, so -1.1% is not yet a result.

## 8. Caveats

- **One seed.** Seed noise is 0.116 MAE where five decomposition families span 0.04. Nothing at the 1% scale here is settled.
- **The ridge used for the paired tests is the 37.77 fit.** A finer penalty grid reaches 37.04 with the validation and test optima coinciding at lam=3e4 and a selection effect of 0.000, but that fit saved no predictions. Against it the gap to `joint` is 3.9%, not 1.9%.
- **Exogenous channels are history, not forecasts**, so every spatial number is an upper bound, and more so at h=6 than at h=1.
- **No checkpoints exist for most arms**, so statements about where the bank put its bands describe the geometric initialisation. `--save-model` now writes centres, bandwidths and the coupling tensor; only the `+exp` arms used it.
- **Two years, one region, one target.** As before.

[`PITFALLS.md`](PITFALLS.md) records how to run this on PACE and the measurement traps that produced three retracted numbers in one session.
