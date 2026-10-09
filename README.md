# Decomposition for electricity price forecasting

Can a learnable frequency decomposition improve electricity-price forecasting beyond a strong raw-window baseline?

**Current answer:** the band model improves on the raw-window LSTM in this experiment, but does not beat the best tuned ridge on overall MAE. Jacobian weighting changes the balance between typical prices and extreme prices; it does not produce a model that wins everywhere.

> **Experiment scope:** generated October 9, 2026; horizon `h=6` as reported; SA1 evaluation, 2021 test year; one seed; 30 epochs. The joint model predicts five regions. Neural arms share the window list, objective configuration except for stated ablations, and seed. No experiments were rerun for this rewrite.

## At a glance

| Comparison | MAE ($/MWh) | What it supports |
|---|---:|---|
| Raw-window LSTM (`panel_lstm`) | 41.30 | Neural control without decomposition |
| Learned 16-channel projection (`red16`) | 39.84 | Narrowing the input helps this LSTM |
| Bands + per-band coupling (`joint`) | 38.50 | The full band architecture improves further |
| Bands + coupling + Jacobian α = 0.25 | 37.79 | Objective weighting closes the gap to the original ridge fit |
| Raw-window ridge (`arx_window`) | 37.77 | Best MAE in the main comparison table |
| Ridge with a finer penalty grid | **37.04** | Best reported MAE; predictions were not saved for paired tests |

The full band model improves MAE by **6.8%** relative to `panel_lstm`. This is a comparison of complete input architectures, not an isolated estimate of the decomposition effect. Relative to the finer-grid ridge, `joint` has approximately **3.9% higher MAE**.

## 1. Architecture

![Architecture: shared frequency bank, separate own and exogenous pathways, shared LSTM, and region-specific embeddings.](assets/architecture.png)

### 1.1 Frequency bank

Each input window contains 37 channels and 96 half-hour observations:

$$
x\in\mathbb{R}^{B\times37\times96}.
$$

The bank applies an rFFT to each channel, multiplies the spectrum by eight Gaussian masks, and applies an inverse rFFT:

$$
x_{c,k}=\mathcal{F}^{-1}\!\left(M_k\odot\mathcal{F}(x_c)\right),
\qquad
M_k(f)\ge0,\quad\sum_{k=1}^{8}M_k(f)=1.
$$

The resulting modes have shape **`(B, 37, 8, 96)`**. Because the masks sum to one, the modes reconstruct the input:

$$
\sum_k x_{c,k}=x_c.
$$

Reported reconstruction error: `4.8e-07`.

The bank has eight center-gap parameters and eight bandwidth parameters. Centers are ordered through cumulative softmax gaps, initialized geometrically with gaps proportional to `1.8^k`. These masks cover slow through fast variations, up to the Nyquist frequency. Exact learned periods are unavailable for most arms because checkpoints were not saved.

**All reported runs use `adapt=0`: masks are shared across input windows.** The encoder and gap head for input-dependent masks are inactive in these runs.

### 1.2 Per-band coupling

For target region r, the model keeps its own eight modes and constructs eight weighted combinations of other channels:

$$
\operatorname{own}_{r,k}=x_{r,k},
\qquad
\operatorname{exo}_{r,k}=\sum_{c\ne r}\Delta_k[r,c]x_{c,k}.
$$

The source uses the parameterization `A_k = I + Δ_k`; the self-term is excluded from the exogenous pathway. At identity initialization, `Δ = 0`, so the exogenous features are zero.

Concatenation produces **`(B, 5, 16, 96)`**. Five target rows per band give `8 × 5 × 36 = 1,440` active off-diagonal coupling entries out of 10,952 allocated entries. The 37 channels include physical drivers as well as prices, so this pathway mixes variables as well as regions.

### 1.3 Shared prediction head

Regions are folded into the batch:

$$
(B,5,16,96)\rightarrow(B\cdot5,96,16).
$$

A two-layer bidirectional LSTM with hidden size 128 produces a 256-dimensional representation. A learned region embedding is added, followed by:

$$
256\xrightarrow{\mathrm{Linear}}128\xrightarrow{\mathrm{ReLU}}128
\xrightarrow{\mathrm{Linear}}1.
$$

The output has shape **`(B, 5)`**. Bidirectionality operates within the historical window; it does not by itself use observations after the forecast origin.

| Component | Allocated parameters | Share | Notes |
|---|---:|---:|---|
| Frequency-bank module | 57,656 | 8.9% | Only 16 bank-shape parameters participate when `adapt=0`; 57,640 adaptive-network parameters are inactive |
| Per-band coupling | 10,952 | 1.7% | 1,440 off-diagonal entries receive gradients from five target rows |
| Shared head | 579,073 | 89.4% | LSTM 544,768; embeddings 1,280; MLP 33,025 |
| **Total** | **647,681** | **100%** | Allocated count is not the same as active count |

### 1.4 Architectural variants

| Switch / arm | Change |
|---|---|
| `joint_nocouple` | Freeze coupling at identity; exogenous pathway remains zero |
| `+film` | Add FiLM modulation over own-region bands |
| `+ctx336` | Use a 336-step bank context with a 96-step readout |
| `+exp` | Give physical drivers separate band channels; head input grows from 16 to 88 channels |
| `+red<N>` | Replace the bank representation with a learned N-channel projection |
| `@<sd>` | Use seeded coupling initialization instead of identity |

These variants have different information paths, widths, or initializations. They should not all be described as exact no-ops at initialization without checking the implementation.

## 2. What does a linear decomposition buy?

### 2.1 The decomposition is linear when masks are fixed

With `adapt=0`, the trained bank is a fixed linear map:

$$
u=Dx,\qquad
u=[x_1^\top,\ldots,x_K^\top]^\top.
$$

Learnable parameters do not make the map nonlinear in its input. Input-dependent masks would change this conclusion.

Summing modes recovers x, so D is injective and has a left inverse. It is a redundant representation, rather than a square invertible matrix. The complete bank adds no new observations; downstream compression in the exogenous pathway can still discard information.

### 2.2 Why a linear predictor gains no expressive power

A linear model on the modes can always be written as a linear model on the raw window:

$$
\hat y=a^\top Dx=(D^\top a)^\top x.
$$

Conversely, any raw-window prediction can be represented using the modes:

$$
\beta^\top x=\sum_k\beta^\top x_k.
$$

Thus the two representations support the same set of linear prediction functions.

### 2.3 Why ridge predictions need not be identical

Ridge also penalizes coefficients. Raw-window ridge penalizes `||β||²`; mode-space ridge penalizes `||a||²`. These generally induce different penalties on the same prediction function.

For a full-column-rank D, the minimum mode-space coefficient norm needed to represent β is:

$$
\min_{a:D^\top a=\beta}\|a\|_2^2
=\beta^\top(D^\top D)^{-1}\beta.
$$

That is generally not `||β||²`. Partition of unity guarantees reconstruction; it does not guarantee an equivalent ridge penalty. Feature scaling, penalty selection, and implementation details matter.

**Reported observation:** ridge on bands and ridge on raw windows agree to `0.0000 $/MWh`, with prediction correlation `1.000000`. Treat this as an empirical result requiring implementation verification, not a mathematical consequence of reconstruction alone.

### 2.4 What the neural ablations suggest

| Step | Model | MAE | Change from previous row |
|---|---|---:|---:|
| Raw channels | `panel_lstm` | 41.30 | — |
| Learned 16-channel projection | `red16` | 39.84 | −3.5% |
| Frequency bands + coupling | `joint` | 38.50 | −3.4% |

Two plausible explanations are:

1. **Input compression / regularization.** The 37-channel arm reportedly reaches its best validation result at epoch 1 and then degrades; `red16` peaks at epoch 18. The 88-channel `+exp` arms peak at epochs 4–8. These observations suggest sensitivity to input width, but do not establish a universal monotonic relationship.
2. **Easier access to window-wide structure.** Whole-window filtering lets a mode value depend on other observations within the historical window. This may reduce the burden on an LSTM's sequential state. It is a mechanism hypothesis, not an isolated causal finding.

The tree comparison in the source is the right experiment stated against the wrong reference. Inside `analysis/basis_matters.py` — one learner, one subsampled training set (11,134 windows, every 4th), the same held-out test rows — it reads:

| Learner | Features | Dim | Test MAE | Change |
|---|---|---:|---:|---:|
| Ridge | raw window | 96 | 40.174 | — |
| Ridge | bands | 768 | 40.174 | −0.00% |
| Boosted trees | raw window | 96 | **38.797** | — |
| Boosted trees | bands | 768 | **40.509** | **+4.4%** |

So the band basis makes the tree **4.4% worse**, not 2.1% better. The `41.37` figure is `gbt_own` from a separate full-data run (44,535 training windows) and is not comparable to a subsampled band arm; pairing the two would charge a training-set-size difference to the basis. The validation column of that log is in-sample — `early_stopping=True` with `validation_fraction=None` stops on *training* loss, and the fit uses train plus val — so only the test column carries a conclusion. The test rows are held out in both arms, which is what makes the raw-versus-band contrast usable.

Ridge reading `40.174 → 40.174` is the expected null: a linear model absorbs a change of basis into its weights. A tree cannot — it splits on individual coordinates — so if the band basis exposed structure that raw time steps hide, a tree is where it would show. It does not show. This does not prove no useful representation exists; it does mean this one is not it, for this learner.

## 3. Objective: align training with the reported metric

### 3.1 Loss shape

The original Smooth L1 objective used `beta=1.0` on standardized targets. The source reports that 86.9% of trained-model residuals fell in its quadratic region, while evaluation used absolute error.

A prior loss sweep reported:

| Loss setting | Reported score in that sweep |
|---|---:|
| L1 | 13.744 |
| Smooth L1, beta = 0.5 | 13.909 |
| Smooth L1, beta = 2.0 | 14.067 |
| Smooth L1, beta = 4.0 | 14.234 |

These are a separate sweep's figures, not the MAEs in the main results table. The current default is **L1**.

### 3.2 Loss coordinates

The price transform compresses extreme values:

$$
z=\operatorname{asinh}\!\left(\frac{p-c}{w}\right),
\qquad p=c+w\sinh z,
\qquad J(z)=\left|\frac{dp}{dz}\right|=w\cosh z,
$$

where c is the training median and w is the training IQR.

| Regime | Mean Jacobian ($/MWh) | Relative to calm | Rows |
|---|---:|---:|---:|
| Calm | 49 | 1.0× | 12,030 |
| High | 81 | 1.7× | 1,939 |
| Negative | 98 | 2.0× | 3,335 |
| Spike | 1,080 | 22.0× | 115 |

An equal transformed-space error can correspond to a much larger dollar error in a spike.

### 3.3 Jacobian weighting

The price loss uses the truth-dependent weight:

$$
q_{ir}=\left(\frac{J(z_{ir})}{\operatorname{mean}_{j,s}J(z_{js})}\right)^\alpha,
\qquad
L_{\mathrm{price}}=\frac{1}{|B|R}\sum_{i,r}q_{ir}|\hat z_{ir}-z_{ir}|.
$$

`--w-jacobian alpha` controls the emphasis on high-Jacobian observations. At α = 0, this recovers the unweighted objective. At α = 1, it approximates dollar MAE locally:

$$
|\hat p-p|=w|\sinh\hat z-\sinh z|
\approx J(z)|\hat z-z|.
$$

It is not exactly dollar MAE for large prediction errors. If the model predicts a further standardized coordinate, include that coordinate's scale in the Jacobian and evaluate cosh in the unstandardized asinh coordinate. Specify whether the normalization mean is training-wide or batchwise in the implementation.

| α | Calm weight | High weight | Negative weight | Spike weight |
|---:|---:|---:|---:|---:|
| 0 | 1.00× | 1.00× | 1.00× | 1.00× |
| 0.25 | 0.92× | 1.04× | 1.09× | 1.98× |
| 0.5 | 0.84× | 1.06× | 1.18× | 3.08× |
| 1.0 | 0.71× | 1.18× | 1.43× | 15.70× |

Weights above are source-reported regime summaries.

### 3.4 Full objective

Let ρ be the pointwise loss, R the five price targets, and A the auxiliary driver targets:

$$
L=L_{\mathrm{price}}+\lambda_{\mathrm{dev}}L_{\mathrm{dev}}
+\lambda_{\mathrm{aux}}L_{\mathrm{aux}}+\lambda_{\mathrm{bias}}L_{\mathrm{bias}}
+\lambda_{\mathrm{sp}}L_{\mathrm{sp}}+0.05L_{\mathrm{bw}}+L_{\mathrm{sep}}.
$$

| Term | Definition | Default weight |
|---|---|---:|
| Price | Mean of `q_ir ρ(ẑ_ir − z_ir)` over samples and regions | 1 |
| Regional deviation | Mean of `ρ((d(ẑ)_ir − d(z)_ir) / σ_d,r)` | 1.0 |
| Auxiliary drivers | Mean driver-target loss `ρ(ẑ_ia − z_ia)` | 0 |
| Bias | Mean across regions of the absolute batch-mean residual | 0.1 |
| Coupling sparsity | Mean absolute off-diagonal coupling `|Δ_k[r,c]|` | 1e−4 |
| Bandwidth | Mean band width | 0.05 |
| Separation | Mean squared positive overlap-margin violation | 1.0 |

Here,

$$
d(v)_{ir}=v_{ir}-\frac1R\sum_s v_{is},
$$

and σ_d,r is the training standard deviation of the regional deviation. The separation term uses adjacent bands:

$$
L_{\mathrm{sep}}=\operatorname{mean}_k
\left[\max\left(0,\frac{m(bw_k+bw_{k+1})}{2}-(ctr_{k+1}-ctr_k)\right)\right]^2,
\qquad m=1.
$$

The deviation term encourages regional differences because the source attributes 96.4% of cross-region variance to a common mode. Auxiliary driver prediction was tried and reportedly reduced accuracy.

**Ablation coverage:** α is swept; `--w-dev 0` is listed as an available deviation ablation. Freezing coupling removes the coupling pathway, but does not isolate its L1 penalty. The bias and bank-regularization coefficients were not separately tested in this report.

## 4. Results

All error values below are in $/MWh. Regime columns report MAE.

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

The best main-table MAE is ridge at **37.77**; the best main-table RMSE is **250.3** at α = 0.75, about **1.6% below** ridge's 254.3. The finer-grid ridge's MAE is 37.04; its RMSE is not supplied, so it cannot be placed in the two-objective plot below.

## 5. Pareto view: which errors are being traded?

![MAE–RMSE trade-off and per-regime error changes across Jacobian exponents.](assets/pareto_frontier.png)

**Panel A:** both axes should be minimized. Among the five non-FiLM α variants, α = 0.25, 0.5, and 0.75 are nondominated: none improves both MAE and RMSE over another. α = 0 and α = 1 are dominated. Adding the paired-test ridge makes α = 0.25 dominated as well; the remaining displayed frontier is ridge, α = 0.5, and α = 0.75.

This is an **empirical frontier among the displayed candidates**, not a bound on what another model could achieve. Lines connect measured candidates and do not imply evaluated intermediate models.

**Panel B:** negative values mean lower regime MAE than ridge. Increasing α generally helps the tails while hurting calm-period accuracy, but the changes are not strictly monotonic. Percentages in the plot are recomputed from the rounded main-table values; small differences from the original percentage table are rounding-related.

### 5.1 Source-reported paired comparisons

The following table retains the source percentages and DM p-values. Percentages are relative to the 37.77 ridge fit; lower is better.

| alpha | negative | calm | high | spike | MAE | RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +5.1% (1.000) | -1.9% (0.095) | +5.2% (0.999) | +0.4% (0.719) | 38.50 | 254.9 |
| 0.25 | -1.4% (0.167) | -1.2% (0.178) | +3.7% (0.994) | +0.4% (0.852) | 37.79 | 254.8 |
| 0.5 | -3.2% (0.015) | +2.8% (0.969) | +3.7% (0.938) | -1.8% (0.129) | 37.85 | 251.3 |
| 0.75 | -6.8% (0.000) | +12.9% (1.000) | +3.3% (0.797) | -3.6% (0.075) | 38.41 | 250.3 |
| 1.0 | -2.0% (0.092) | +24.4% (1.000) | +14.4% (0.979) | -4.1% (0.212) | 40.95 | 251.7 |

The source does not specify the DM test's alternative hypothesis, serial-correlation correction, or adjustment for multiple comparisons. Values close to 1 suggest directional tests, but this needs confirmation. A displayed `0.000` is a rounded value, not literally zero.

### 5.2 Practical readings

| Candidate | Main advantage | Main cost |
|---|---|---|
| α = 0.25 | Nearly matches the original ridge MAE; slightly lower negative and calm errors | Does not improve high or spike error over ridge in this table |
| α = 0.5 | Lower RMSE and negative-price error than ridge | Calm and high errors increase |
| α = 0.75 | Lowest RMSE; best negative-price MAE among these α variants | Calm-period MAE rises substantially |
| α = 1 | Lowest spike MAE among the non-FiLM α variants | Worse overall MAE and RMSE than α = 0.75 |
| FiLM + α = 0.75 | Lowest spike MAE among trained models in the main table | Overall MAE is 40.59 |

At α = 0.75, negative-price error is reported as 6.8% below ridge with rounded p = 0.000. At α = 0.5, the negative-price comparison also has reported p = 0.015. Statistical interpretation requires the test specification and attention to the number of comparisons.

## 6. Error concentration and regime routing

### 6.1 A small tail contributes substantial total error

| Regime | Rows | Share of rows | Approx. share of joint absolute error | Joint MAE | Ridge MAE |
|---|---:|---:|---:|---:|---:|
| Negative | 3,335 | 19.1% | 29% | 58.2 | 55.4 |
| Calm | 12,030 | 69.1% | 29% | 15.9 | 16.3 |
| High | 1,939 | 11.1% | 19% | 66.8 | 63.5 |
| Spike | 115 | 0.7% | 23% | 1348.5 | 1343.3 |

Only 115 spike rows contribute roughly 23% of absolute error. Persistence has lower spike MAE than every trained model in the table. This shows weakness of the tested models on these events; it does not establish that the events are intrinsically unpredictable.

After removing the cross-region common mode, the source reports regional structure accounting for 0.4–1.9% of variance by band. The fastest band is reported to be 4.6 times more regional than the most shared band. This is consistent with a hypothesis about fast regional decoupling, but does not establish its physical cause or a numerical upper bound on predictive gains.

### 6.2 Oracle routing is a diagnostic, not a deployable result

A source-reported oracle chooses a model using the **realized future regime**:

| Realized regime | Selected model |
|---|---|
| Negative | α = 0.75 |
| Calm | `joint` |
| High | Ridge |
| Spike | Persistence |

Reported oracle MAE: **35.86**, versus ridge's 37.77 (**−5.1%**). This is an optimistic retrospective comparator for the stated experts and routing rule, not a generally achievable forecast or a universal bound.

A classifier using the historical window reportedly recalls 26% of negative-price events and 3% of spikes. Hard routing worsens MAE by 2.4%. Soft routing improves it by 1.1%, while a fixed mixture improves it by 0.9%. Those weights were not fitted out of sample, so neither mixture improvement is an established generalization result.

## 7. Evidence limits and checks needed

- **One seed and a narrow evaluation scope.** The report evaluates SA1 in the 2021 test year while training a five-output model; it also references a two-year data span. Verify split dates and horizon units before publication. The source reports seed noise of 0.116 MAE elsewhere, versus 0.04 across five decomposition families; small differences need replication.
- **Stronger ridge baseline.** The finer-grid ridge reaches 37.04 MAE at `lam=3e4`; the source says validation and test optima coincide on that grid. Use validation-only selection, save predictions, and rerun paired comparisons before claiming a gain over ridge.
- **Historical exogenous features.** Their use alone does not make the result an upper bound. Validate availability at the forecast origin, publication delays, and revisions. Forecast covariates, if used later, require their own availability checks.
- **Mask interpretation.** Most arms lack checkpoints. Initial center locations cannot be presented as learned bands. `--save-model` now records centers, widths, and coupling tensors; only `+exp` arms used it in the reported sweep.
- **Regime definitions.** Thresholds defining calm, high, negative, and spike are not included in the supplied text. Add them for reproducibility.
- **Ridge equivalence.** Audit band-feature construction, normalization, regularization, and prediction comparison before treating exact equality as a mechanism result.
- **Mechanism attribution.** The compression and whole-window explanations are hypotheses supported by partial controls. The existing comparisons do not uniquely identify them.
- **Statistics.** Document DM-test direction, loss differential, temporal dependence treatment, sample counts, and multiplicity. Save per-timestamp predictions for all arms.

## 8. Prior findings and repository references

The original report attributes two earlier findings to [`attic/FINDINGS-full.md`](attic/FINDINGS-full.md): leakage in the evaluated published VMD forecasting setups, and a reported `10³–10⁵×` compute advantage for a filter bank over the compared decomposition solver. Those claims are not independently verified by the tables in this README.

The source also reports paired effects of −6.8% for `joint` versus the raw-window LSTM and −5.6% for `joint` versus identity-frozen coupling, with rounded DM p = 0.000 and improvements in 33/48 and 32/48 half-hour slots, respectively. These are full-arm comparisons; their interpretation should reflect every architectural difference.

[`PITFALLS.md`](PITFALLS.md) records the PACE execution and measurement traps behind these runs. Both files are in this repository; the two claims above are still inherited from the earlier report and are not re-derived by the tables here.

---

### Editorial notes

This rewrite preserves the supplied numerical results and distinguishes observations from hypotheses. It corrects the ridge-equivalence argument, the best-spike claim, the strict-monotonicity claim, the interpretation of coupling-penalty ablation, and the unsupported upper-bound claim about historical exogenous inputs.

One correction has since been reversed against the logs. An earlier draft restated the tree comparison as `41.37 → 40.51`, a 2.1% improvement; those two numbers come from different experiments with different training-set sizes. The within-experiment comparison in `logs/basis_matters.log` is `38.797 → 40.509`, a 4.4% deterioration, and §2.4 now reports that. No new experiments were run for this rewrite.
