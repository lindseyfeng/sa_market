# sa_market — band decomposition for SA1 electricity prices

AEMO NEM, South Australia, half-hourly RRP. Train 2018, test 2019 unless a file
says otherwise.

**[`FINDINGS.md`](FINDINGS.md) is the only current document.** Read it first
and treat it as authoritative. The older running log has been moved to
`attic/RESULTS-superseded.md` and carries a banner saying so -- it is kept only
because FINDINGS and THREE_ARMS cite its section numbers, and several of its
comparisons were later found confounded (VMD without its residual; epoch
selected on test).

## Naming

Stop calling the method NVMD. It shares no objective, no algorithm and no
optimisation with VMD. It is a **band-parameterised filter bank** applied as a
differentiable layer inside the forecaster. The old name anchors the work to a
baseline it does not resemble and invites the wrong review.

## What the project claims

| # | claim | status |
|---|---|---|
| 1 | Published VMD gains are **leakage**, not decomposition | **Strong — the headline** |
| 2 | The same band structure costs 10²–10⁵× less | **Strong** |
| 3 | Decomposing **inside** the forecaster beats the classical decompose-then-forecast pipeline | **Supported**, 2 seeds, no overlap |
| 4 | Classical modes fail on the **physics of the bands**, not the choice of algorithm | **Supported** |
| 5 | Spatio-temporal beats VMD | **Fails as stated.** Line still open |

Claim 4 is the one that makes the null results legible. Every classical method
hands the model bands with the same defects: VMD's lowest band is 2.5× wider
than its own centre so it smears across DC, the decomposition is effectively
1.6 modes, and at window 96 nothing above 20.1 h exists. Vary the *algorithm*
and nothing moves — five families within 0.3%, learned and hard-coded banks
within 0.002, basis churn spanning 26× with no effect. Vary what the bands
*are* and it does.

## Things that cost real time to learn

- **Give every arm 100% of the signal.** VMD's residual is 8.5–9.5% of price
  sigma. Dropping it handed VMD ~91% while a partition-of-unity bank got 100%,
  and correcting it returned 0.237 MAE — more than the margin being claimed.
- **Never select on test.** `benchmark.py`'s `va_dl` is built from the *test*
  CSV, so sections 2, 8 and 13.3 of `attic/RESULTS-superseded.md` report a minimum over epochs.
  That does not subsidise all arms equally: it subsidises the high-variance one
  (+0.170 for the spatial arm against +0.016 elsewhere).
- **Seed noise dominates.** One method varies 0.116 across seeds while five
  decomposition families span 0.04. Any unpaired single-seed margin is noise.
- **Delivery path is a confound.** Precomputed per-timestep modes and in-model
  decomposition are not interchangeable; the same filter bank scores 0.084
  apart. Match it before attributing anything to the basis.
- **`--concat 1` for the spatial variant.** Letting mixed modes replace the
  target's own destroys the partition of unity (recon error 0.00 → 1.82).
- **Neighbour prices as spreads, not levels** (SA1–VIC1 correlation 0.914).
- **Patience 10+ on a 30-epoch cosine schedule**; a descending stale window is
  not convergence.
- **MPS cannot run these models**: no `rfft`, and complex multiply asserts.

## Standing caveats for any writeup

- h=1 is **saturated**: persistence 14.40 against a best model near 14.3, and
  the control alone varies 0.222 across seeds. Nothing is resolvable there.
  Section 11a puts the spatial gain at −2.21 MAE at h=6.
- Weather is **reanalysis, not forecast**, so every spatial gain is an upper
  bound. Load-bearing, because claim 1 is itself an accusation of leakage.
- `merge_asof(direction="nearest", ±60min)` lets a :30 settlement take a :00
  reading up to 30 minutes ahead.
- MVMD extends VMD to joint multi-channel decomposition. "VMD cannot use
  spatial information" is **not** defensible; scope it to univariate VMD as
  used in the audited literature.
- Two years, one region, one target.

## Running things

Entry points are modules, from the repo root. See `README.md` for the layout.

```bash
python3 -m decomp.vmd_panel_modes --years 2018,2019   # causal VMD, ~3.3 h, cached
python3 -m decomp.decomp_zoo --methods ewt,emd,wpt,bank
python3 -m analysis.basis_stability --n 1500
./scripts/zoo_then_dose.sh        # decomposition families, then the churn ladder
./scripts/spatial.sh              # horizon x exogenous-window 2x2
python3 -m report.report_findings # regenerate FINDINGS.md
python3 -m report.plot_bands      # regenerate the band-comparison figure
```

Both runners **resume**: any `(arm, seed)` already in the result JSON is
skipped. Mode generation caches one `.npy` per channel-year with an atomic
write and validates it on resume. A kill costs one run.

Use the shell scripts rather than bare python. This box has 8 GB and the
desktop stays in use; without `nice` and a thread cap, epoch times swing from
91 s to 6451 s under swap pressure. A `--max-windows` run is a pipeline check
and is force-redirected to `*.check.json` so it cannot pollute results.
