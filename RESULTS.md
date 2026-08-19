# NVMD results snapshot

Data: AEMO SA1 half-hourly RRP, filtered to `RRP in [1, 981.65]` (reproduces the
literature split: 63,538 train rows vs their 63,540).
Unless stated, experiments below use **train 2018 / test 2019**, W=96 windows.

---

## 1. Leakage in the existing benchmark family  *(strongest result)*

Per-mode linear AR(48) probe (`ar_probe.py`). If modes are extrapolable to far
below the series' own variability, they encode future information.

| decomposition | summed MAE | median per-mode err | % of RRP std | verdict |
|---|---|---|---|---|
| Per-year VMD | 4.288 | 0.205 | 0.42% | **LEAKING** |
| Causal VMD W=96 | 15.907 | 1.357 | 2.77% | ok |
| NVMD v2 | 18.837 | 1.262 | 2.58% | ok |
| NVMD v3 | 16.159 | 1.606 | 3.28% | ok |

Same algorithm, same data, same K -- only the decomposition window changed, and
per-mode extrapolation error rose 6.6x.

**Headline evidence:** on per-year VMD modes, a 625-parameter linear regression
(MAE 4.29) beats a 5.7M-parameter CNN-BiLSTM (MAE 13.42). When capacity is
irrelevant, nothing is being learned -- the modes are being read, not forecast.

Reproduced the project's original 7.11 MAE / 11.58 RMSE result exactly and
traced it to this effect (it degrades to ~10.7 once decomposition is segmented).

---

## 2. Forecast accuracy vs a leak-free, context-matched baseline

`benchmark_seeds.py`, 5 seeds, paired vs causal VMD W=96. `*` = exceeds 2 SE.

| method | Linear | MLP | LSTM |
|---|---|---|---|
| Causal VMD | 15.97 +/- 0.11 | 15.89 +/- 0.04 | 14.43 +/- 0.07 |
| NVMD v2 | 15.27 +/- 0.05 | 15.99 +/- 0.18 | 14.42 +/- 0.04 |
| **NVMD v3** | **15.17 +/- 0.09** | **15.37 +/- 0.08** | **14.28 +/- 0.06** |

Paired difference vs causal VMD (negative = better):

| method | Linear | MLP | LSTM |
|---|---|---|---|
| NVMD v2 | -0.70+/-0.09* | +0.10+/-0.14 | -0.01+/-0.05 |
| **NVMD v3** | **-0.79+/-0.16*** | **-0.52+/-0.08*** | **-0.15+/-0.08*** |

NVMD v3 wins on all three, all significant: **4.9% / 3.3% / 1.0%**.
On a ridge linear model the gap is 6.2% (14.703 vs 15.674).

**The margin shrinks as the predictor strengthens.** See section 5.

---

## 3. Interpretability  *(largest measured advantage after leakage)*

`interpret_modes.py`, test year.

| | Causal VMD (13 modes) | NVMD v3 (9 modes) |
|---|---|---|
| participation ratio | 1.75 | **6.11** |
| energy in top mode | 74.31% | 28.16% |
| longest period represented | 20.8 h | **307.2 h** |
| bandwidth profile | flat ~0.06-0.08 | 0.003 -> 0.19 (scales with centre) |
| centres ordered | by post-hoc omega sort | **by construction, every window** |

- VMD's "12-mode decomposition" is effectively **1.75 modes**: Mode_1 holds 74%
  of energy and costs +10.61 MAE to ablate; 10 of 13 modes are removable at
  <0.01 MAE.
- VMD's Mode_1 has bandwidth 0.0627 at centre 0.0240 -- bandwidth 2.6x the
  centre. That is a smear, not a band.
- At W=96 VMD has **no mode above 20.8 h** and cannot represent multi-day
  structure. NVMD's mode 1 sits at 307 h with 18.5% of energy.
- NVMD band table is physically readable: DC / 63 h / 20.5 h (daily) /
  11.0 h (half-daily) / 6.4 / 4.5 / 4.4 / 2.4 h.

Known weakness: NVMD's emitted modes are mutually redundant (8/9 removable at
<0.01 MAE) because adjacent bands overlap (0.04 bandwidths separation). Broad
high-frequency masks get dragged down by their low-frequency tails.

---

## 4. Runtime

| | mode generation, 1 year (~17k windows) |
|---|---|
| Causal VMD W=96 (7 cores) | 123.4 s |
| **NVMD v3** | **0.1 s** |

~1000x. VMD solves an optimisation per window; NVMD is one forward pass.
This is what makes multi-scale decomposition feasible for NVMD and not for VMD.

---

## 5. Why the accuracy margin is small  *(a finding, not a failure)*

Both decompositions are **invertible transforms** -- modes sum exactly to the
input. So neither adds information; they can only help via conditioning. The
benefit is therefore bounded by how much the predictor struggles without it,
which is exactly the observed pattern: Linear 4.9% -> MLP 3.3% -> LSTM 1.0%.

**The loss is nearly flat in the decomposition parameters.** Across configs
whose decompositions differ enormously:

| config | decomposition | val MAE |
|---|---|---|
| v3static | K=8 geometric bank | 14.16 |
| k16 | K=16 | 14.22 |
| fastband | bands wrecked, coverage hole at 2-6 h | 14.23 |
| k12 / adapt | K=12 static / input-adaptive | 14.29 |
| L=192 | 2x window | 14.26 |

0.9% spread. Wrecking the filter bank costs 0.07 MAE. The filter bank is not
what limits accuracy.

Ablation agrees: for both methods the forecast comes almost entirely from the
recent level/trend, which both represent adequately. They differ on fine band
structure, which at h=1 is mostly unforecastable noise. With RRP std 49 and
best-achievable MAE ~14.3, the noise floor dominates.

**Conclusion: NVMD is a better decomposition; decomposition quality is not what
limits short-horizon forecast accuracy once leakage is removed.** This explains
the literature -- the large reported gains were leakage, not decomposition.

**Implication:** to exceed this ceiling you need *new information*, not a better
basis. Hence spatial (cross-regional) extension.

---

## 6. Negative results worth recording

- K=12 and K=16 do not beat K=8 (14.29 / 14.22 vs 14.16).
- Input-adaptive masks (`adapt=0.5`) do not beat a static bank (14.29 vs 14.16).
- Longer window alone does not help (L=192: 14.26 vs L=96: 14.16).
- Aggressive band LR (3e-2) wrecks the bank and hurts accuracy (14.23).
  Band-parameter gradients have sign consistency 1.000 and are ~8x the LSTM's,
  so movement is step-budget limited -- but the optimum is near the geometric prior.
- Edge-padding the FFT window (the circular-boundary hypothesis) changes nothing
  (<0.5% on Linear/MLP/LSTM). Boundary energy ratio measured at 1.002.

---

## 7. Caveats to carry into any writeup

- ~~Causal VMD is untuned (`alpha=2000, K=12`). A fair paper must sweep it.~~
  **RESOLVED, section 10.** Swept 18 configs; the win survives at a reduced margin.
- CNN-BiLSTM is not yet in the multi-seed table; it is the cell where v2 lost
  and the closest analogue to the published baseline. **5-seed run in progress.**
  Note `CNNBiLSTMPredictor` hardcodes `lstm_layers=2` while `MRC_BiLSTM` defaults
  to 3, so it is not literally the published architecture.
- **`benchmark.train_and_eval` uses the TEST file as its validation loader**
  (`benchmark.py:186`) -- early stopping and model selection both run on the test
  year. It is a paired comparison so the bias partly cancels, but every "val MAE"
  in this document is best-epoch-on-test, and the section 6 config choices
  (K=8 over K=12/16, static over adapt) were therefore made on the test year.
  Decide how to handle this before any writeup; it is more serious than the
  untuned-baseline issue was.
- Univariate VMD is the only baseline compared. **MVMD** (Rehman & Aftab 2019)
  and per-channel-VMD-then-concatenate both exist, so "VMD cannot use spatial
  information" is not a defensible sentence -- compare or scope out explicitly.
- The `RRP in [1, 981.65]` filter drops 2,040 rows, creating time gaps up to
  34.5 h that rolling windows silently span. Inherited from the literature.
- 2022 test year is a severe distribution shift (mean 176 vs 50-93 in training
  years); results on the long split should be read in that light.
- NVMD modes are optimised jointly with an LSTM head, which biases
  cross-predictor comparison. A linear-head variant is implemented but untested.

---

## 8. Robustness of the NVMD > VMD result (5-seed, paired vs causal VMD)

All five v3 configurations beat causal VMD on all three predictors, every cell
significant. The win is not configuration-dependent.

| config | Linear | MLP | LSTM |
|---|---|---|---|
| v3-static | -0.79+/-0.16* | -0.52+/-0.08* | -0.15+/-0.08* |
| v3-k12 | -0.71+/-0.08* | -0.26+/-0.10* | -0.23+/-0.03* |
| v3-k16 | -0.77+/-0.14* | -0.28+/-0.07* | -0.19+/-0.10* |
| v3-adapt | -0.84+/-0.13* | -0.24+/-0.14* | -0.17+/-0.02* |
| v3-linhead | -0.83+/-0.07* | -0.33+/-0.09* | -0.23+/-0.03* |

No config escapes the ceiling (best Linear 5.3%, best LSTM 1.6%).
`linhead` (predictor-agnostic training) is best-or-tied on LSTM, so removing the
joint-training bias helps transfer slightly.

## 9. Multi-scale decomposition: negative result

Concatenating L=96/192/384 decompositions (24 modes vs 8).

| method | Linear | MLP | LSTM |
|---|---|---|---|
| v3 L96 | -0.79+/-0.16* | -0.52+/-0.08* | -0.15+/-0.08* |
| v3 L384 | -0.72+/-0.28* | -0.30+/-0.08* | +0.12+/-0.08* |
| v3 multiscale | **-0.91+/-0.19*** | -0.35+/-0.12* | +0.05+/-0.06 |

Helps the weakest predictor (best single margin observed, 5.7%), degrades the
stronger ones -- LSTM drops from a significant win to parity. The extra scales
re-encode the same window rather than adding information, so they are redundant
channels that dilute a strong predictor. Third independent confirmation of the
ceiling in section 5.

Longer window alone is monotonically worse: L=96 14.16, L=192 14.26, L=384 14.32.


---

## 10. Baseline fairness: causal VMD hyperparameter sweep

Answers section 7 item 1. 18 configs, `alpha` x `K` at the matched window W=96,
both years, generated from the same source CSVs as the v3static modes so row
counts stay aligned (17099 / 16529) and the pairing stays valid.

Screened with `ridge_screen.py` -- closed-form ridge, ~110 s per config, feature
construction mirroring `ModeWindowDataset` exactly. The full multi-seed benchmark
costs ~10.7 h per method pair, which makes an 18-point grid (~80 h) infeasible.
The screen reproduces the known ridge result to within 1 point (15.727 / 14.917
here vs 15.674 / 14.703 in section 2), erring *against* NVMD, and it targets the
Linear cell -- where NVMD's margin is largest, i.e. the most VMD-favourable test.

| config | test MAE | vs NVMD v3static |
|---|---|---|
| **NVMD v3static (K=8)** | **14.917** | -- |
| VMD a1000 k8  *(best by test)* | 15.470 | -3.6% |
| VMD a250 k8  *(best by train-val)* | 15.478 | -3.6% |
| VMD a2000 k12  *(the untuned baseline used above)* | 15.727 | -5.2% |
| VMD a250 k16  *(worst)* | 16.396 | -9.0% |

**Tuning the baseline recovers 0.25 of the 0.81 gap (31%). The remaining 3.6% is
real**, in the cell where the fairness objection was strongest.

The result does not depend on how the baseline is tuned: selecting VMD's config
honestly (train-val tail of 2018) gives 15.478, selecting it on the test year
gives 15.470 -- a 0.008 difference. Ridge lambda is likewise chosen on a held-out
tail of the train year, never on test.

**Two findings that strengthen section 5:**

1. **Alpha barely matters.** Across `alpha` in [250, 8000] at K=8 the spread is
   15.470-15.607 -- **0.9%**, the same flatness measured for NVMD in section 5.
   The insensitivity of the loss to decomposition parameters is not an NVMD
   quirk; it holds for classical VMD, now over 18 points rather than 5.
2. **VMD's optimum is K=8 -- the same K as NVMD.** Every K=8 config beats every
   K=12 config, which beats most K=16. The K=12 inherited from the literature was
   simply wrong, as section 3 predicted ("effectively 1.75 modes").

**Cost asymmetry (updates section 4).** Per year, VMD ranged **59 s to 18,178 s**
depending on (alpha, K) -- K=16 is pathological, up to 5 h for one year. NVMD is
0.1 s. The advantage is **600x to 180,000x**, not a single 1000x. The whole sweep
cost 22.5 h; the NVMD equivalent is minutes. *This is what the speed claim is
actually for: tuning is affordable for one method and not the other.*

**Still owed:** sections 2 and 8 tabulate against `a2000 k12`. The fairness
objection does not vanish, it moves to those tables -- they need regenerating
against `a250 k8`, which would cut the headline Linear margin from ~4.9% to
~3.6% and may take the LSTM cell to parity.

---

## 11. Horizon sweep: h = 6, 12, 24, 48

Two separate experiments, on different data. **They do not compose** -- see
section 12.

**(a) Spatial coupling on vs off, within NVMD** (`train_nvmd_st.py`, compound
panel, min test MAE over 2 seeds):

| horizon | xfilter on | off | gain |
|---|---|---|---|
| h=6  | 22.83 | 25.04 | **-2.21** |
| h=12 | 26.03 | 27.11 | **-1.08** |
| h=24 | 27.54 | 27.90 | -0.36 |
| h=48 | 28.16 | 28.17 | -0.01 |

Monotone decay, **not** the inverted-U the ridge probe suggested: the exogenous
panel's value roughly halves per horizon doubling and is gone by h=48. Only 2
seeds -- at h=48 the seed spread (27.93 vs 28.39) is 40x the gain, so the last
two rows are indistinguishable from zero, not "small but real".

**(b) NVMD v3static vs causal VMD** (`benchmark_seeds.py`, 3 seeds, paired,
`*` = exceeds 2 SE):

| horizon | Linear | MLP | LSTM |
|---|---|---|---|
| h=6  | -0.86* | -0.20 | -1.14* |
| h=12 | -0.77* | +0.02 | -0.92* |
| h=24 | -0.90* | +0.50 | -0.02 |
| h=48 | -0.72* | +0.60* | -0.03 |

The Linear win is flat and significant at every horizon (~3%). The LSTM win
exists only at short horizons and is dead by h=24. MLP is a genuine *loss* at
h=48. Section 5's ceiling shrinks with predictor strength *and* with horizon --
a fifth independent confirmation, after sections 5, 8, 9 and 10.

---

## 12. Claims status

| # | claim | evidence | status |
|---|---|---|---|
| 1 | Per-year/global VMD leaks; the literature's gains are leakage, not decomposition | Section 1: 6.6x per-mode AR extrapolation error from the window alone; 625-param linear (4.29) beats 5.7M CNN-BiLSTM (13.42); original 7.11 reproduced and traced | **Strong.** The headline |
| 2 | NVMD gives an equal-or-better decomposition at orders-of-magnitude lower cost | Sections 4 and 10: 0.1 s vs 59-18,178 s per year; 600x-180,000x | **Strong**, and previously understated |
| 3 | Spatio-temporal NVMD beats VMD | -- | **Not supported as stated. The comparison has never been run** |

**On claim 3.** Every VMD comparison in this document uses *temporal-only* NVMD
on the price-only per-year CSVs. Every spatial result is NVMD-vs-NVMD on
`compound_2018_2022.csv` -- different data, different pipeline, different MAE
scale (23-28 vs 25-29). "Spatial beats temporal NVMD" and "temporal NVMD beats
VMD" cannot be composed into "spatial NVMD beats VMD".

VMD being univariate does not rescue this. It makes the *setup* fair -- VMD gets
everything VMD can take -- but fairness is not measurement. The missing run is
three arms on identical rows and protocol:

| arm | information | decomposition |
|---|---|---|
| causal VMD, price only | temporal | univariate VMD |
| causal VMD per channel, concatenated | spatial | univariate VMD x ~26 |
| NVMD ST (concat) | spatial | joint |

Arm 1 vs 3 is the claim. Arm 2 is what stops a reviewer objecting that VMD was
never given the extra channels -- and beating arm 2 is the stronger result,
because it isolates *joint* decomposition from merely *having* the data.
Cost at the tuned K=8/alpha=1000: ~115 s/year/channel, ~1.7 h for both years
across the 26 non-calendar channels of the 33-channel panel.

**Defensible restatement:**

1. Per-year VMD leaks; we quantify it with a per-mode AR probe and a
   capacity-irrelevance test, and we reproduce and explain the published 7.11.
2. NVMD matches or beats VMD's decomposition at 10^2-10^5x lower cost, which is
   what makes tuning and multi-scale feasible at all.
3. Once leakage is removed, NVMD beats a **fully tuned** VMD by 3.6% on linear
   predictors, shrinking toward parity as predictor and horizon grow --
   decomposition quality is not what limits accuracy (section 5). Spatial
   exogenous information helps at short horizons only, and its gain is an
   **upper bound**: the weather is reanalysis, not forecast. That caveat is
   load-bearing here, because claim 1 is itself about leakage.
