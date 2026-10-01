# The decomposition is not doing the work

*train 2018-2020 or a sliding year, test 2019 / 2021-2022, SA1 half-hourly*

**Summary.** Against a plain LSTM on the same 33 channels, the band
decomposition is worth 0.15 MAE and the exogenous channels are worth 2.45, on a
matched comparison that `FINDINGS.md` never ran. The 0.15 reverses sign across
error segments and across months. The bank is a partition of unity, so it
reconstructs its input exactly and is an invertible linear reparameterisation:
buying nothing is what it should do, and it is also what explains the project's
existing null results about band parameters. Separately, the target FINDINGS
measures on has had its hard part filtered out -- persistence on 2019 halves,
26.29 to 14.37 -- and a variance-stabilising transform nobody used is worth 28%
on the same AR, an order of magnitude more than every architectural difference
in that document combined.

Everything in [`FINDINGS.md`](FINDINGS.md) is measured on a target that drops
every negative price and everything above 981.65. This round puts those rows
back and asks the same questions again.

## 1. The filter was removing the problem

| year | rows dropped | negative | above 981.65 | persistence, raw | persistence, filtered |
|---|---:|---:|---:|---:|---:|
| 2018 | 1.6% | 1.2% | 0.28% | 26.39 | 14.09 |
| 2019 | 4.8% | 4.5% | 0.19% | 26.29 | 14.37 |
| 2020 | 10.5% | 9.8% | 0.09% | 18.27 | 9.72 |
| 2021 | 20.7% | 19.1% | 0.17% | 29.38 | 17.16 |
| 2022 | 20.7% | 18.7% | 0.51% | 70.35 | 34.03 |

The drop rate rises from 1.6% to 20.7% and is almost entirely negative prices,
which go from 1.2% to 19.1% of the year. That is the single largest structural
change in this market over the period -- South Australian solar and wind
pushing the price below zero in the middle of the day -- and the filter removes
it. Persistence on 2019 halves, 26.29 to 14.37, so **every 14.1-14.8 in
`FINDINGS.md` is measured on a series whose hard part has been taken out.**

It also filters train and test to different distributions: 1.6% of 2018 against
20.7% of 2022.

Removing rows rather than masking them also breaks the sampling grid. The
filtered series is not uniformly spaced, so two adjacent samples can be hours
apart, and a 96-step window spans a variable amount of real time -- which is
the one assumption a frequency decomposition cannot do without.

## 2. The error is a spike problem

2019, unfiltered:

| share of timesteps | share of total absolute error |
|---|---:|
| top 0.1% | 22.9% |
| top 1% | 37.3% |
| top 5% | 56.0% |
| top 10% | 67.2% |

Skewness 30.4, kurtosis 973, range -900 to 14,700. A mean absolute error on
this series is an average over a distribution that is one third its own worst
percentile, so a single number hides which part of the problem a method is
solving.

## 3. Variance stabilisation is worth more than any architecture here

The same AR(48), fitted on 2018-2020 and scored on 2021-2022:

| | MAE |
|---|---:|
| raw price | 63.72 |
| asinh, inverted for scoring | **45.69** |

**18.03 MAE, 28.3%**, stable across both test years (+26.7% and +28.5%). For
scale, every architectural difference in `FINDINGS.md` is under 0.7 MAE and
seed noise is 0.116. In raw space the same model loses to persistence (49.90);
in asinh space it beats it by 4.2. The transform is standard in electricity
price forecasting (Uniejewski, Weron & Ziel 2018) and nothing in this project
used it.

## 4. Why the band decomposition does not help at h=1

The bank is a partition of unity: measured reconstruction error 0.00. That
makes it an **invertible linear reparameterisation** of the window. The K band
signals sum exactly to the input, so any linear functional of the bands is a
linear functional of the window and conversely: for a linear read-out, bands
and raw window span the same function class and the decomposition adds
nothing. Its only possible value is as a preconditioner, making some
*non*-linear function easier to express.

This project's own results already say the bank's parameters do not matter:
learned and hard-coded banks within 0.002, five decomposition families within
0.3%, basis churn spanning 26x with no effect. That is what an invertible
reparameterisation predicts.

What is predictable at h=1 is also the wrong shape for a band decomposition.
In asinh space the one-step change has autocorrelation **-0.191 at lag 1** and
-0.044 at lag 2, noise thereafter:

| model | 2019 MAE |
|---|---:|
| persistence | 26.29 |
| AR(1) | 28.18 |
| AR(2) | 27.23 |
| AR(12) | 27.22 |
| AR(96) | 25.63 |

Three parameters capture 2.15 of the 2.16 that twelve capture. The predictable
component sits at the top of the band, next to Nyquist, and a log-spaced
partition of unity has overlapping filters there by construction, so it spreads
that one component across several channels and smooths it.

Spikes are worse. Measured over the eight bands, the share of each band's
energy that comes from the top 1% of windows is **6.9% to 7.6%** -- flat. A
transient is maximally non-sparse in frequency, so the decomposition does not
separate it from anything. In asinh space the same figure is 2.2% to 3.0%,
still flat.

**The signal class is wrong for the representation.** This series is a slowly
varying level, a daily shape, and sparse transients. A frequency bank is for a
sum of narrowband oscillations with distinct dynamics.

## 5. Where it does have something: h=24

At 12 hours ahead the conditions reverse. Persistence collapses, long lags
start paying, and the daily cycle dominates:

| horizon | persistence | seasonal naive | AR(96) | AR(336) |
|---|---:|---:|---:|---:|
| h=1 | 29.38 | 57.95 | **25.63** | 25.68 |
| h=6 | 56.47 | 57.96 | 39.34 | **38.13** |
| h=24 | 72.99 | 58.00 | 45.26 | **43.95** |
| h=48 | 57.95 | 58.03 | 46.60 | **45.44** |

At h=1 a week of history is worth -0.05; at h=24 it is worth +1.31.

On matched rows and the same 96-step window, at h=24 on 2021 the two models are
no longer collinear: prediction correlation **0.747**, against 0.992 at h=1,
and a least-squares combination of the two scores **44.58** against 46.21 and
46.36 alone, with both coefficients positive (1.055 and 0.616). The
decomposition carries something the linear model does not. It is just not
enough to win on its own.

## 6. Rolling retraining does not close the gap

Sliding one-year training window, refit each quarter, h=24, tested on 2019.
AR(96) is fitted once on 2018 and never refitted.

| | n | persistence | AR(96), never refit | nvmd_st, refit quarterly | nvmd_temporal | selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 4273 | 149.12 | **85.59** | 89.51 | 90.39 | 2 |
| Q2 | 4321 | 33.97 | **22.17** | 22.80 | 25.08 | 8 |
| Q3 | 4369 | 64.70 | **40.49** | 43.37 | 46.56 | 5 |
| Q4 | 4369 | 75.24 | **46.22** | 52.66 | 48.84 | 2 |
| pooled | 17332 | 80.51 | **48.49** | 51.96 | 52.59 | |

Two things to read here, and one not to.

**Not to:** the spatial coupling. It is +0.88, +2.27, +3.19 and then **-3.81**,
pooling to +0.63 on one seed. Three quarters in one direction and the fourth in
the other is not a result.

**To read:** the quarters where the network does worst are the quarters where
validation stopped it at epoch 2. Under a sliding window the validation split
is the tail of the training window, whose difficulty bears no stable relation
to the test quarter -- Q1 has val 27.6 against test 89.5. Model selection is
not working, and part of the gap in this table is that rather than capability.

## 7. The no-decomposition control: the bank is not on the causal path

Q2 2019, h=24, sliding one-year window. Identical head, identical window, one
seed. The only thing that changes down the table is what the head is fed.

| arm | inputs | MAE | spike 1% | next 9% | calm 90% | bias | pred sd |
|---|---|---:|---:|---:|---:|---:|---:|
| `nvmd_st` | 33 ch, decomposed, coupled | **22.81** | 113.66 | 61.87 | 17.87 | +0.25 | 24.0 |
| `lstm_panel` | 33 ch, **no decomposition** | **22.96** | 113.92 | 60.55 | 18.17 | -3.45 | 22.8 |
| `nvmd_temporal` | 33 ch, decomposed, uncoupled | 25.08 | 129.39 | 72.16 | 19.19 | -2.77 | 12.6 |
| `lstm` | price only, no decomposition | 25.41 | 127.39 | 71.46 | 19.65 | -2.96 | 11.3 |
| persistence | | 34.50 | 219.18 | 123.59 | 23.51 | +0.17 | 42.9 |

Truth sd 42.9.

**The decomposition is worth 0.15 MAE and the exogenous panel is worth 2.45.**

The 0.15 does not survive being looked at. `nvmd_st` wins the spike segment by
0.26 and the calm segment by 0.30 and **loses the middle segment by 1.33**; by
month it wins April by 2.66 and May by 1.18 and loses June, the hardest month,
by 3.15. Three segments in two directions and three months in two directions is
not an advantage, it is a wash. The two predictions correlate 0.874 and a
least-squares combination of them scores 22.50, barely better than either.

**The line that does separate the table is the prediction standard deviation,
and it does not follow the decomposition.** The two arms near 23 predict with
sd 24.0 and 22.8; the two arms near 25 predict with sd 12.6 and 11.3, against a
truth of 42.9. The split is between arms that move and arms that have collapsed
onto a near-constant, and it runs straight through the decomposed group:
`nvmd_st` is on one side of it and `nvmd_temporal` on the other.

So the band front-end is not what produces the numbers. The 32 exogenous
channels are. The spatial coupling and the raw panel are two routes to the same
place -- 22.81 and 22.96 -- and the decomposition sits off the causal path.

This is the control `FINDINGS.md` never had, and it reframes every arm-versus-
arm margin in that document: those tables compare decompositions to each other
without ever asking what decomposing buys over not decomposing.

It is also what section 4 predicts. An invertible linear reparameterisation
should buy approximately nothing, and it buys 0.15. It explains the project's
own null results -- learned against hard-coded banks within 0.002, five
families within 0.3%, churn spanning 26x with no effect -- as consequences of
the construction rather than as findings about decomposition.

**What is structurally wrong.** The bank is a partition of unity, so it
reconstructs its input exactly and spans the same functions as the raw window
for any linear read-out. It can only help by making a non-linear function
easier to express, and on this signal it makes it harder: the predictable
component at h=1 sits beside Nyquist where log-spaced Gaussian filters overlap
most, and spike energy is flat across all eight bands (6.9-7.6%), so the one
part of the signal that carries 37% of the error is exactly the part a
frequency basis cannot isolate. Splitting a window into K lower-amplitude
channels and asking a recurrent head to recombine them also costs amplitude:
both uncoupled decomposed arms collapse to a quarter of the target's spread.

The honest statement is not that decomposition hurts. It is that **this
decomposition, in this position in the architecture, is doing no work**, and
the project's positive results need to be re-attributed to the exogenous panel
until a control says otherwise.

## 8. What this says about where to put the decomposition

The failure is not the loss function. Training L1 instead of Huber moved the
calm 90% of 2021 from 13.77 to 13.54 while the prediction standard deviation
went from 86 to 161 against a truth of 261 -- the shrinkage eased and the error
did not follow.

Nor is it directional accuracy. On the h=1 change, nvmd_st gets the sign right
**61.6%** of the time against AR(48)'s 57.9%, and correlates +0.567 with the
true change. It has more directional information than the model that beats it,
and loses by predicting larger magnitudes: median |predicted change| 7.12
against AR's 4.89 and a true 8.27. Scaling its predicted change by 0.8 recovers
27.34 to 26.79; scaling per volatility regime recovers more (held-out: 26.33
against AR's 25.67, with factors 0.35 in the calmest bin and 0.80 in the most
volatile).

So the decomposition is finding something real and the architecture around it
is spending that advantage. The direction this points is to stop asking the
decomposition to be the whole representation. A linear branch over the target's
own lags, initialised to the least-squares solution, makes the linear baseline
the floor and leaves the decomposition to supply the correction -- which is the
part it demonstrably does well, beating AR by 5.05 on the next-9% segment at
h=1 while losing 1.56 on the calm 90%.

## Open

- One quarter for the no-decomposition control, not four. Section 7 is the
  comparison that matters most and it rests on 4,321 scored rows.
- One seed throughout. Nothing here separates two arms on its own.
- The weather channels enter through `merge_asof(direction="nearest", +/-60min)`
  and a `bfill`, both of which read forward. That is a leak in every spatial
  result, here and in `FINDINGS.md`.
