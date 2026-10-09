# Pitfalls, measured

Every entry here cost real time on 2026-10-08/09 and every one is a thing I got
wrong, not a thing that was hard. Read it before running on PACE or before
trusting a margin.

## PACE Phoenix

### Connecting, from scratch

```bash
ssh login-phoenix.pace.gatech.edu            # xfeng300, GT VPN must be up
pace-quota                                   # charge account + storage
```

`~/.ssh/config` already carries this, and it is what makes unattended runs
possible:

```
Host login-phoenix.pace.gatech.edu
  HostName login-phoenix.pace.gatech.edu
  User xfeng300
  IdentityFile ~/.ssh/id_ed25519_pace        # no passphrase, this host only
  IdentitiesOnly yes
  ControlMaster auto                         # one login serves many commands
  ControlPath ~/.ssh/cm-%r@%h-%p
  ControlPersist 12h
  ServerAliveInterval 60
```

The dedicated key exists because the older `id_ed25519` is passphrase-protected
and the passphrase is lost; `IdentitiesOnly yes` stops ssh offering it and
hanging on a prompt no script can answer. To rebuild it:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_pace -N "" -C "pace-$USER"
ssh-copy-id -i ~/.ssh/id_ed25519_pace -o IdentitiesOnly=yes \
            -o PubkeyAuthentication=no xfeng300@login-phoenix.pace.gatech.edu
```

That last command needs the GT password typed interactively, and Duo if it is
asked for, so it is the one step that cannot be automated.

Account facts as of 2026-10-09:

| | |
|---|---|
| charge account | `gts-sd111` (Deng), 285.16 credits available |
| cost | a 5-minute A100 job charged **0.02** -- roughly 0.24 credits/GPU-hour |
| home | 20 GB, and a torch env is 5.8 GB, so keep it out |
| scratch | `~/scratch`, 15 TB, **wiped between semesters** |
| project | `/storage/project/r-sd111-0`, 1 TB |
| repo lives at | `~/scratch/sa_market`, env at `~/scratch/.conda/envs/samkt` |

Cost is not the constraint here; queueing is. Pull results home when a sweep
finishes -- anything that only lives on scratch is not a result yet.

### What decides whether a GPU job starts

Measured on one afternoon, same model, same account:

| request | submit -> start |
|---|---|
| no GPU | **3 s** |
| 1 GPU, 8 cores, 2 h walltime | 6 min, then still pending |
| 1 GPU, 8 cores, 15 min | 54 s |
| 1 GPU, 1 core, 5 min | **3 min 51 s** |
| 1 GPU, 1 core, `--qos=embers` | never landed |
| 1 GPU, pinned `--partition=gpu-v100` | never landed |

**The scheduler plans around what you declare, not what you use.** Ask for one
core and twenty minutes. A four-core request came back as eight cores with 64 GB
and a billing weight of 10,261, which is a large hole to wait for, and the CPU
only feeds the GPU here anyway.

Things that did *not* help:

- `embers` has priority 0. It is the free tier and it is *slower*; `inferno`
  landed in under a minute while an identical `embers` job sat for ten.
- Dropping the GPU type from `--gres=gpu:1` does not let a job cross partitions.
  Partitions are defined by node feature, so Slurm still placed it in
  `gpu-v100`. List them: `--partition=gpu-a100,gpu-l40s,gpu-h100,gpu-rtx6000`.
- `interactive-gpu` appears in `sacctmgr show qos` but is not submittable with
  sbatch.
- No GPU node on the cluster is ever `idle`. Every one is `alloc`, `mix` or
  `drain`. Free *cards* on `mix` nodes are what backfill fills; count them from
  each node's `AllocTRES`, not from partition state.

### Do not include gpu-v100

A V100 is compute capability 7.0 and `torch 2.14.1+cu130` ships kernels for
CC >= 7.5 only. A job scheduled there dies in five seconds with "does not
include kernels for this GPU". Three arms were lost that way. A100 / L40S /
H100 / RTX6000 are fine.

### Environment

- `pip install --index-url https://download.pytorch.org/whl/cuXXX torch`
  **replaces** PyPI instead of adding to it, so torch's own build dependencies
  cannot resolve and the install dies on `No matching distribution found for
  flit_core`. Plain `pip install torch` already gives a CUDA wheel on Linux
  x86_64. Use `--extra-index-url` if a specific build is genuinely needed.
- **`source activate <env>` does not switch interpreters in a non-interactive
  shell, and does not say so.** It silently left the base python 3.10 in place
  and pip installed 5.2 GB into a 20 GB home directory. Call the interpreter
  directly: `$HOME/scratch/.conda/envs/<env>/bin/python`.
- Symlink `~/.conda` and `~/.cache` into scratch *before* creating an
  environment. Home is 20 GB; scratch is 15 TB.
- Do not poll `import torch` to decide whether an install finished. The package
  directory exists long before the install is complete, so the import fails on
  a half-written `libtorch_global_deps.so` and a retry loop waits forever on
  something that is already doomed. Wait for the installer's own exit marker.

### SSH

`Server accepts key` followed by `Permission denied` is **not** two-factor
authentication. It means the private key is passphrase-protected and a
non-interactive command cannot answer the prompt. Check with
`head -3 ~/.ssh/id_ed25519 | grep ENCRYPTED` and `ssh-add -l`. A key of its own
with no passphrase, scoped to that host with `IdentitiesOnly yes`, is what makes
unattended runs possible -- and it means filesystem access to the laptop is
login access to the cluster, which is a trade to make deliberately.

`rsync host:'~/path/*'` does not expand `~` remotely. Use a path relative to the
remote home, or scp.

### Job hygiene

- **One result file per job.** Six parallel jobs sharing `--out` race on one
  JSON that the resume path reads at startup.
- **Prediction filenames come from the arm name**, so four arms that differ only
  in a CLI flag all write the same `.npz` and clobber each other. Give each its
  own `--save-preds` directory.
- `sed "s/^/$arm /"` breaks when an arm name contains `/`. Use printf.
- Two background waiters whose conditions both come true submit the job twice.
  Check `squeue` before submitting, not after.
- **Check whether a job has already started before cancelling it.** One was
  killed 63 seconds in, four seconds before the line being waited for.
- Save checkpoints. Nothing did, for the whole session, which left every
  statement about "the learned bands" describing the geometric initialisation.

## Training and measurement

### The objective is not a hyperparameter

Twice, in the same session, the loss was not optimising the reported metric.

1. **Shape.** `smooth_l1_loss(beta=1.0)` on a standardised target is quadratic
   for **86.9%** of residuals -- measured -- so the objective was MSE for the
   bulk while every number reported was an absolute error. The repo had already
   swept it and the ordering is monotone: L1 13.744, beta 0.5 13.909, beta 2.0
   14.067, beta 4.0 14.234.
2. **Space.** Fixing the shape left the space wrong. The loss lives in asinh
   coordinates, the metric is $/MWh, and `dp/dz = w*cosh(z)`, so one unit of
   asinh error is worth 64 $/MWh in the calm band and 1,080 on a spike -- a
   factor of 22 the loss ignored. The model was trained to ignore the rows that
   dominate the metric, and did.

### Every baseline mistake ran the same way

Three separate errors, all of which made the *control* weaker and the model look
better. None was deliberate and that is the point: own-model details get care,
baselines get defaults.

- **Features in a different space from the target.** Raw prices against an
  asinh target: the best possible linear fit scores an MAE of **29,865** against
  persistence's 29.5, because a linear map cannot approximate asinh. LEAR
  transforms inputs and outputs alike for this reason.
- **No prediction clamp.** The networks clamp to the training range; the ridge
  did not, reached **272,650 $/MWh**, and five such rows moved its MAE from a
  median of 38.7 to a mean of 98.8.
- **A grid too coarse to find the optimum.** Decade steps put the ridge penalty
  optimum between 1e4 and 1e5; validation picked 1e5 and test came out 0.7 MAE
  worse than the 3e4 the finer grid found. Half-decade steps, and check that the
  chosen value is interior.

Also: `HistGradientBoostingRegressor(early_stopping=True,
validation_fraction=None)` stops on the **training** loss, and fitting on
`vstack([train, val])` makes the reported validation MAE in-sample. That
combination produced a 3.4% "tree beats ridge" result that vanished under a
clean setup.

### Comparisons that looked like findings

- **A global mean can be settled where nothing has skill.** 115 spike rows are
  0.7% of the test set and **23%** of the total absolute error, and every model
  including the ridge sits near 1,300 $/MWh on them. Negative prices are another
  19.6% of rows and 29% of the error. Segment before concluding.
- **Comparing one arm's selected epoch against another's mid-flight epoch.**
  That produced a "decomposition is worth 14%" claim; both arms at their own
  validation-selected epoch gives 6.8%.
- **Choosing the ordering after seeing the effects.** A per-region pattern read
  r = -0.968 against "correlation with SA1" and r = -0.70 (p = 0.18) against
  neutral connectivity measures -- and the SA1 ordering was picked after looking.
  n = 5.
- **Not removing the common mode first.** Cross-region correlation is 0.99 in
  every band, which looks like "no scale dependence" and means nothing: 96.4% of
  the variance across regions is one common mode. After removing it, regional
  structure is 0.4-1.9% of variance per band and the fast band is 4.6x the
  most-shared one.
- **Early-epoch advantages evaporate.** Three times in a row -- joint at h=1,
  the coupling-init sweep, the FiLM arm -- an arm led by 3-9% for four epochs and
  finished level. Only epochs inside the convergence region compare.

### Things worth knowing about this model

- Zero-initialising the coupling makes the exogenous pathway *identically zero*
  at step 0, so it must be grown from nothing while the own-mode pathway starts
  as a lossless identity. It buys an exact ablation; keep both inits and report
  which one each number came from.
- A partition-of-unity band decomposition is an invertible linear map. Ridge on
  bands and ridge on the raw window agree to **0.0000 $/MWh**. No linear reader
  can ever tell them apart, so a linear control is not a test of the
  decomposition -- the no-decomposition *same-head* control is.
- One seed cannot resolve a 1% margin here. Seed noise is 0.116 MAE where five
  decomposition families span 0.04.
