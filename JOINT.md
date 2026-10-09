# Joint multi-region forecasting: what the decomposition is actually worth

h=6, SA1, 2021 test year, one seed, 30 epochs, L1 loss. Every neural arm shares
the window list, the loss and the seed; the only thing that differs is the arm.
Diebold-Mariano is one-sided with an Andrews AR(1) bandwidth -- the loss
differential runs AC1 0.66 here, where the textbook zero-lag variance overstates
z by an order of magnitude.

| arm | SA1 MAE | what it is |
|---|---:|---|
| `arx_ridge` | **37.772** | ridge on the raw 96 x 37 window |
| `joint/lstm+film` | 38.454 | bands + coupling + FiLM gate |
| `joint` | 38.504 | bands + per-band coupling, LSTM head |
| `joint/lstm+film+ctx336` | 39.388 | the same, bank seeing 336 steps |
| `joint_nocouple` | 40.768 | coupling frozen at the identity |
| `single:SA1_price` | 41.259 | single-task LSTM |
| `panel/lstm` | 41.303 | **the no-decomposition control** |

| contrast | effect | DM p | half-hours |
|---|---:|---:|---|
| decomposition (`panel` -> `joint`) | **-6.8%** | 0.000 | 33/48 |
| spatial coupling (`nocouple` -> `joint`) | **-5.6%** | 0.000 | 32/48 |
| multi-task (`single` -> `nocouple`) | -1.2% | 0.018 | 19/48 |
| FiLM gate | -0.1% | 0.403 | 9/48 |
| 336-step bank context | +2.0% | 0.998 | 5/48 |
| `joint` against ridge | +1.9% | 0.998 | 7/48 |

## The decomposition is worth 6.8%, and it is not information

Three measurements pin down what it is:

1. **For a linear reader it is exactly nothing.** Ridge on bands and ridge on
   the raw window return identical predictions -- maximum absolute difference
   **0.0000 $/MWh**, correlation 1.000000. A partition-of-unity bank is an
   invertible linear map, so this is forced, not observed.
2. **For a tree it is negative.** Gradient-boosted trees on the target's own
   96-step window score 38.797; on the same window as 8 bands x 96 steps,
   40.509 -- **4.4% worse**. Trees split on single features and cannot fold a
   rotation into a weight matrix, so this is the one learner where a genuinely
   better coordinate system would have to show up. It does not.
3. **For the LSTM it is 6.8%, and the mechanism is the sequential bottleneck.**
   The band masks are non-causal over the window, so `band_k(t)` already carries
   the whole window at every step. An LSTM reading raw channels has to integrate
   96 steps to get the same thing, and spends capacity doing it: `panel/lstm`
   reaches its best validation at **epoch 1** and then degrades monotonically
   for 29 more epochs (25.67 -> 30.29). Bands regularise it. Ridge, which sees
   the whole window at once, has no such bottleneck to shortcut, which is
   exactly why it gains nothing.

So the claim that survives is **"the band decomposition is worth 6.8% as an
inductive bias for a sequential head"**, not "the band decomposition reveals
structure". The distinction is load-bearing: if it revealed structure, a
nonlinear model with it would beat a linear model without it. It does not --
ridge still wins by 1.9%.

## What this says about where to look next

The LSTM is the binding constraint, not the decomposition. Of the three
properties that matter here, it has one:

| | sees the whole window at once | nonlinear | cannot absorb a rotation |
|---|---|---|---|
| ridge | yes | no | no |
| LSTM | **no** | yes | no |
| trees | yes | yes | yes |

Adding modules to the LSTM has now failed twice in a row -- the FiLM gate at
-0.1% (p=0.403) and a 336-step bank context at +2.0% (p=0.998) -- and the FiLM
gate was the only mechanism in the design that a linear reader provably cannot
copy, since `own * (1 + gamma(exo))` is a product. Meanwhile trees on the
target's own window alone reach 38.797 against ridge's 40.174 on the same
information, a 3.4% margin in the one class that satisfies all three properties.

The coupling is the part worth keeping: -5.6% at h=6, p=0.000, 32 of 48
half-hours, and it survived a change of loss and of schedule that moved the
arms it sits between.

## Caveats

- One seed. CLAUDE.md records seed noise at 0.116 MAE where five decomposition
  families span 0.04, so nothing here at the 1% scale is settled.
- The ridge used for the DM test is the 37.772 fit. A finer penalty grid reaches
  **37.037** with validation and test optima coinciding at lam=3e4 and a
  selection effect of 0.000, but that fit was computed without saving
  predictions, so the paired test could not use it. Against 37.037 the gap to
  `joint` is 3.9%, not 1.9%.
- The exogenous channels are history, not forecasts, so every spatial number
  here is an upper bound -- and more so at h=6 than at h=1.
