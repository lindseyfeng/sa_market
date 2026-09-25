# Three-arm comparison: does spatio-temporal NVMD beat VMD?

*generated 2026-09-18 11:46 -- 12/12 runs complete*

Settles claim 3 of `attic/RESULTS-superseded.md` section 12, which that document records as **never run**. Every arm sits on identical rows, windows, head, optimiser and budget, so the numbers below are the first that may be compared directly.

## Result

Test MAE on 2019, mean over seeds, lower is better. Two selection rules, both reported:

- **honest** -- weights chosen on the validation tail of the train year, test scored once from them
- **cherry** -- the minimum over epochs of test MAE, which is the statistic `attic/RESULTS-superseded.md` section 13.3 and `benchmark_seeds.py` report

Neither guarantees generalisation on one test year and one region. The **gap** between them is the size of the selection effect, and is a result in its own right.

| arm | information | decomposition | $n_{in}$ | test MAE (honest) | test MAE (cherry) | gap | seeds |
|---|---|---|---:|---:|---:|---:|---:|
| `vmd_price` | temporal | univariate VMD on the price | 8 | *pending* | | | 0 |
| `vmd_panel` | spatial | univariate VMD per channel, concatenated | 215 | *pending* | | | 0 |
| `nvmd_temporal` | temporal | joint, coupling frozen at $A_k=I$ | 33 | *pending* | | | 0 |
| `nvmd_st` | spatial | joint, per-band coupling, concat | 33 | *pending* | | | 0 |

## Protocol

| | |
|---|---|
| panel | `compound_2018_2022.csv`, 33 channels, target `SA1_price` |
| split | train 2018, test 2019, no overlap |
| VMD | causal, window 96, K=8, alpha=1000 (the *best-by-test* config of section 10, i.e. the most baseline-favourable choice) |
| decomposed | the 26 non-calendar channels; calendar enters `vmd_panel` raw |
| window | 96 steps, horizon 1 |
| scored rows | window ends at row >= 190 in **both** years, for **every** arm, because causal VMD has no mode before row 95 |
| head | LSTM 128x2 bidirectional, dropout 0.1, then 128 -> 1 |
| budget | AdamW, lr 3e-4, cosine to 1e-6, clip 5.0, batch 256, 30 epochs, patience 10 |
| selection | best MAE on the **validation tail of 2018** (last 15%, with a 96-window embargo); test scored once, from those weights |
| seeds | 1, 2, 3 |

## Why this had to be run

`attic/RESULTS-superseded.md` section 12 states the problem plainly: every VMD comparison in that document used *temporal-only* NVMD on the price-only per-year CSVs, while every spatial result was NVMD-against-NVMD on the compound panel. Different data, different pipeline, different MAE scale. "Spatial beats temporal NVMD" and "temporal NVMD beats VMD" cannot be composed into "spatial NVMD beats VMD".

`vmd_price` vs `nvmd_st` is claim 3. `vmd_panel` is the stronger test, because it hands classical VMD the same exogenous panel and so isolates *joint* decomposition from merely *having* the channels. `nvmd_temporal` is the fourth corner of the 2x2.


## Relation to section 13.3

That table read OFF 14.180, MIX 14.635, CONCAT 14.035, XFILTER 14.015 over 2 seeds. It reported the **cherry** statistic and did not align rows with any VMD arm. The cherry column above is the like-for-like comparison; the honest column is not.

The same selection rule is used by `benchmark_seeds.py`, whose `va_dl` is built from the *test* CSV, so sections 2 and 8 report the cherry statistic too. Section 10's ridge screen chose its lambda on a train-year tail and is unaffected, as are claims 1 and 2, which do not depend on epoch selection at all.

The rule was applied **symmetrically** to every method, so it is not favouritism. It biases a *comparison* only through variance: a minimum over ~30 test evaluations rewards whichever arm has the noisier epoch-to-epoch test curve, and `nvmd_st` carries an extra 8x33x33 coupling tensor that `nvmd_temporal` does not.


## Per-run detail

| arm | seed | val MAE | test MAE (honest) | ep | test MAE (cherry) | ep | test RMSE (honest) |
|---|---:|---:|---:|---:|---:|---:|---:|

## Caveats to carry forward

- Two years, one hub, one target. Nothing here speaks to other ISOs.
- The weather channels are **reanalysis, not forecast**, so any spatial margin is an **upper bound**. This caveat is load-bearing: claim 1 of this project is itself an accusation of leakage.
- `merge_asof(direction="nearest", +/-60min)` in the panel build lets a :30 settlement take a :00 reading up to 30 minutes ahead.
- MVMD (Rehman & Aftab 2019) extends VMD to joint multi-channel decomposition. `vmd_panel` is per-channel VMD, not MVMD, so "VMD cannot use spatial information" remains **not** a defensible sentence.
