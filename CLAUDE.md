# sa_market — neural VMD (NVMD) for SA1 electricity prices

AEMO NEM, South Australia, half-hourly RRP. Train 2018, test 2019 unless a file
says otherwise. `RESULTS.md` is the running evidence log and takes precedence
over anything remembered here; read it before claiming a result.

## What the project actually claims

| # | claim | status |
|---|---|---|
| 1 | Per-year/global VMD **leaks**; the literature's gains are leakage, not decomposition | **Strong — this is the headline** |
| 2 | NVMD matches or beats VMD's decomposition at 10²–10⁵× lower cost | **Strong** |
| 3 | Spatio-temporal NVMD beats VMD | **Being settled now — see `THREE_ARMS.md`** |

Claim 1's evidence is a per-mode AR(48) probe (per-year VMD extrapolates 6.6×
better than causal VMD, i.e. it is reading the future) plus capacity
irrelevance (a 625-parameter linear model beats a 5.7M-parameter CNN-BiLSTM on
those modes). The published 7.11 MAE was reproduced and traced to this.

**Claim 3 could not be composed.** Every VMD comparison in `RESULTS.md` used
*temporal-only* NVMD on price-only per-year CSVs; every spatial result was
NVMD-against-NVMD on the compound panel. Different data, pipeline and MAE
scale. `run_three_arms.py` is the matched run that settles it.

## The finding that decides how to tell the story

Both VMD and NVMD are **invertible transforms**, so neither adds information
and any gain is bounded by conditioning alone. Measured consequences:

- the margin shrinks as the predictor strengthens, 4.9% Linear → 1.0% LSTM
- the loss is nearly flat in the decomposition parameters, 0.9% spread across
  banks that differ enormously; wrecking the filter bank costs 0.07 MAE
- the same flatness holds for classical VMD across 18 configs

So **decomposition quality is not what limits short-horizon accuracy once
leakage is removed.** Do not pitch this work as "a better decomposition." The
defensible pitch is an audit (claim 1) plus a representation that admits
exogenous channels and reports which driver acts at which timescale.

## Architecture notes that cost real time to learn

- `SpatioTemporalNVMD` must run with `--concat 1`. The original path let mixed
  modes **replace** the target's own, destroying the partition of unity
  (reconstruction error 0.00 → 1.82) and corrupting the DC/trend bands, so MAE
  got worse while RMSE got better. Concat appends a purely exogenous block with
  the self-weight zeroed and is identically zero at init, so training starts as
  exact temporal-only NVMD.
- Neighbour prices must enter as **spreads, not levels**. Interconnectors
  arbitrage levels together (SA1–VIC1 correlation 0.914), so levels are
  near-duplicates of the target. The first price-only spatial attempt produced
  no gain for exactly this reason.
- `adapt=0.0` in the spatial runs, so the filter bank is static, not
  input-adaptive. Input-adaptive masks did not beat a static bank.
- MPS cannot run these models: no `rfft`, and complex multiply raises an
  internal assert. CPU only.

## Methodology traps, all of which have already bitten

1. **`patience=4` on a 30-epoch cosine schedule stops early.** A stale window
   that is still monotonically descending is not convergence. Use patience 10+.
2. **A plateau under `CosineAnnealingLR(T_max=10)` is not convergence** — the
   LR is 1e-6 by epoch 10. The 30-epoch schedule kept finding new bests through
   epoch 16–19.
3. **Re-run the control at any new budget.** OFF went 14.295 → 14.180 under
   30ep/patience-10; comparing a new arm to the old baseline inflated the
   margin by ~44%.
4. **Do not early-stop on the test set.** Section 13.3 did; `run_three_arms.py`
   selects on a validation tail of the train year with a 96-window embargo.

## Standing caveats that must appear in any writeup

- Weather channels are **reanalysis, not forecast**. Every spatial gain is an
  **upper bound**. This is load-bearing because claim 1 is itself an accusation
  of leakage.
- `merge_asof(direction="nearest", ±60min)` in the panel build lets a :30
  settlement take a :00 reading up to 30 minutes ahead.
- MVMD (Rehman & Aftab 2019) extends VMD to joint multi-channel decomposition,
  and per-channel VMD plus concatenation is trivial to build. "VMD cannot use
  spatial information" is **not** a defensible sentence; scope it to univariate
  VMD as used in the audited literature.
- The four ERCOT-style hub caveats do not apply here, but the two years / one
  region limit does.

## Running things

```bash
# causal VMD modes for every panel channel (~3.3 h, cached per channel-year)
python3 vmd_panel_modes.py --years 2018,2019

# the four-arm comparison; resumes from three_arms_results.json
./run_pipeline.sh            # nice -n 10, 6 threads, logs to three_arms.log
python3 report_three_arms.py # regenerates THREE_ARMS.md from the json
```

Both stages are **resumable**. VMD caches one `.npy` per channel-year with an
atomic write and validates shape and NaN pattern on resume; the trainer skips
any `(arm, seed)` already in the results json. A kill costs one unit of work,
not the run.

This box has 8 GB of RAM and the desktop stays in use. The first training
attempt saw epoch times swing from 91 s to 6451 s purely from swap pressure;
`nice` plus a capped thread count is why the current run is stable at ~35–55 s.
A `--max-windows` run is a pipeline check only and is force-redirected to
`*.check.json` so it can never pollute real results.
