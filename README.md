# sa_market

Whether signal decomposition actually helps electricity price forecasting, and
what the published gains are really made of. AEMO NEM, South Australia,
half-hourly RRP, train 2018 / test 2019.

**Start with [`FINDINGS.md`](FINDINGS.md)** — the whole picture, including the
leakage result the project rests on, the architecture, and every claim we have
since had to retract. [`attic/RESULTS-superseded.md`](attic/RESULTS-superseded.md) is the older running log.

## Layout

| directory | what is in it |
|---|---|
| `models/` | the decomposition layers: `nvmd_v3.py` (band-parameterised filter bank), `nvmd_st.py` (per-band spatial coupling), `nvmd_v2.py` (superseded) |
| `decomp/` | classical decompositions and mode generation: `vmd.py`, `generate_modes.py`, `vmd_panel_modes.py`, `decomp_zoo.py` (EWT / EMD / wavelet packet / fixed bank) |
| `experiments/` | entry points: `run_three_arms.py`, `run_spatial_2x2.py`, and the `train_nvmd_*` trainers |
| `analysis/` | evidence producers: `ar_probe.py` (the leakage probe), `benchmark*.py`, `ridge_screen.py`, `interpret_modes.py`, `basis_stability.py` |
| `data/` | panel construction; `data/raw/` holds the source CSVs |
| `report/` | `report_findings.py` regenerates `FINDINGS.md`; `watch_runs.py` streams run completions |
| `scripts/` | shell runners |
| `results/` | result JSONs, one row per run |
| `cache/` | generated modes, safe to delete and regenerate |
| `runs/`, `logs/`, `figures/` | checkpoints, logs, plots |
| `attic/` | superseded code, kept for provenance |

## Running

Entry points are modules, run from the repo root:

```bash
python3 -m decomp.vmd_panel_modes --years 2018,2019      # causal VMD, ~3.3 h, cached
python3 -m decomp.decomp_zoo --methods ewt,emd,wpt,bank  # the other families
python3 -m analysis.basis_stability --n 1500             # cross-window basis drift
./scripts/zoo_then_dose.sh                               # decomposition families, then the churn ladder
./scripts/spatial.sh                                     # horizon x exogenous-window 2x2
python3 -m report.report_findings                        # regenerate FINDINGS.md
```

Both experiment runners **resume**: every `(arm, seed)` already in the result
JSON is skipped, so a kill costs one run rather than the queue. Mode generation
caches one `.npy` per channel-year with an atomic write and validates shape and
NaN pattern on resume.

## Protocol, in one place

Window 96, horizon 1 unless stated. Selection on a validation tail of the train
year with a 96-window embargo; test scored once from those weights. Results
carry both that number and the minimum of test MAE over epochs, because the
older tables in `attic/RESULTS-superseded.md` report the latter. Every arm sees 100% of the
signal: decompositions that do not reconstruct exactly carry a residual channel.

This box has 8 GB of RAM and the desktop stays in use. Run the shell scripts
rather than bare python — they set `nice` and a thread cap, without which epoch
times swing from 91 s to 6451 s under swap pressure.
