#!/usr/bin/env python3
"""
The three-arm comparison attic/RESULTS-superseded.md section 12 says has never been run.

Claim 3 -- "spatio-temporal NVMD beats VMD" -- cannot be composed out of
"spatial NVMD beats temporal NVMD" (measured on the compound panel) and
"temporal NVMD beats VMD" (measured on the price-only per-year CSVs).  They
use different data, different pipelines and different MAE scales.  This script
puts every arm on identical rows, windows, head, optimiser and budget:

    arm            information   decomposition
    vmd_price      temporal      univariate VMD on the price
    vmd_panel      spatial       univariate VMD per channel, concatenated
    nvmd_temporal  temporal      joint, coupling frozen at A_k = I
    nvmd_st        spatial       joint, per-band coupling, concat mode

vmd_price vs nvmd_st is claim 3.  vmd_panel is the stronger test: it isolates
*joint* decomposition from merely *having* the exogenous channels.
nvmd_temporal is the free fourth corner of the 2x2.

Protocol notes that differ from the numbers already in attic/RESULTS-superseded.md section 13:

  * Every arm is restricted to window ends at row >= 190 of its year, because
    causal VMD modes do not exist until row W-1 = 95 and a window needs 96 of
    them.  All four arms therefore score the same rows.
  * Model selection is on a held-out tail of the *train* year, never on test.
    Section 13.3 early-stopped on test; both numbers are reported here so the
    two tables can be reconciled.

    python vmd_panel_modes.py --years 2018,2019     # ~1.8 h, once
    python run_three_arms.py --seeds 1,2,3
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from models.nvmd_st import STForecaster

CAL_PREFIX = "cal_"
ARMS = ["vmd_price", "vmd_panel", "nvmd_temporal", "nvmd_st"]

# The stability family.  All price-only (R=1), so the only thing that varies is
# the filter bank, and the 2x2 below is complete:
#
#                     VMD-like centres      geometric centres
#   per-window        vmd_price             --
#   fixed across      fixed_vmdmean         fixed_geo / nvmd_trained
#
# If fixed_vmdmean matches nvmd_trained, neither *learning* nor *band
# allocation* is the mechanism, and cross-window basis stability is.
STABILITY_ARMS = ["vmd_price_res", "fixed_geo", "fixed_vmdmean", "nvmd_trained"]

# The zoo: other decompositions, same causal protocol, produced by
# decomp_zoo.py.  Each is read as K modes plus a residual channel, so every
# arm sees 100% of the signal regardless of whether its method reconstructs
# exactly.  adaptive = the basis is re-solved in every window.
ZOO = {"ewt": "adaptive", "emd": "adaptive", "wpt": "fixed", "bank": "fixed"}

# The dose-response ladder.  Same fixed bank, same everything, with its band
# centres deliberately moved by a chosen amount in every window.  Measured
# churn: 2.2 / 3.7 / 9.2 / 28.2 / 60.9 / 82.0 per cent, so the ladder spans and
# brackets VMD's 31.7 per cent.  If MAE tracks it, the five-method correlation
# becomes a controlled dose-response.
JITTER = {"bankjit0.015": 3.7, "bankjit0.025": 9.2, "bankjit0.04": 28.2,
          "bankjit0.07": 60.9, "bankjit0.12": 82.0}
ZOO.update({k: "jittered" for k in JITTER})

# Measured over 1500 consecutive 2018 windows, K=8, alpha=1000, W=96.
VMD_MEAN_CENTRES = [0.0001, 0.0268, 0.0862, 0.1567,
                    0.2292, 0.3026, 0.3776, 0.4519]


def dump_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    os.replace(tmp, path)


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data
class ArmDataset(Dataset):
    """Windows of `feat`, target = raw price `h` steps past the window end.

    `starts` indexes window start rows, so every arm can be handed the same
    list and the scored rows are identical by construction.
    """

    def __init__(self, feat, target, starts, L, h, layout="TC"):
        self.x = torch.from_numpy(feat.astype(np.float32))
        self.y = torch.from_numpy(target.astype(np.float32))
        self.starts = starts
        self.L, self.h, self.layout = L, h, layout

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        w = self.x[s:s + self.L]                       # (L, C)
        if self.layout == "CT":
            w = w.T                                    # (C, L) for STForecaster
        return w, self.y[s + self.L + self.h - 1].unsqueeze(0)


def load_year(df, year, chans):
    return df[df.SETTLEMENTDATE.dt.year == year][chans].to_numpy(np.float64)


def load_modes(outdir, year, decomp_chans):
    """-> (T, n_chan, K); NaN in the first W-1 rows."""
    arrs = [np.load(os.path.join(outdir, f"{year}_{c}.npy")) for c in decomp_chans]
    return np.stack(arrs, axis=1)


def load_zoo(arm, year, K, W, channel="SA1_price"):
    """-> (T, K) modes from decomp_zoo.py for one method."""
    return np.load(os.path.join(f"cache/decomp_{arm}_K{K}_W{W}", f"{year}_{channel}.npy"))


def with_residual(m, target):
    """K modes + the part of the signal they do not reconstruct.

    VMD leaves 8.5-9.5% of the price standard deviation unmodelled and EWT
    leaves a few percent, while NVMD's masks are a partition of unity and
    leave nothing.  Without this channel the arms are not seeing the same
    information and the comparison measures reconstruction completeness as
    much as basis quality.
    """
    res = target - m.sum(axis=1)
    return np.concatenate([m, res[:, None]], axis=1)


def build_features(arm, raw, modes, cal_idx, price_pos, zoo_modes=None):
    """Per-arm feature matrix over the full year, before normalisation."""
    T = raw.shape[0]
    if arm in ZOO:
        return with_residual(zoo_modes, raw[:, 0])             # (T, K+1)
    if arm in ("fixed_geo", "fixed_vmdmean", "nvmd_trained"):
        return raw[:, [0]]                                     # (T, 1) raw price
    if arm == "vmd_price":
        return modes[:, price_pos, :]                          # (T, K)
    if arm == "vmd_price_res":
        return with_residual(modes[:, price_pos, :], raw[:, 0])  # (T, K+1)
    if arm == "vmd_panel":
        flat = modes.reshape(T, -1)                            # (T, 26K)
        return np.concatenate([flat, raw[:, cal_idx]], axis=1)  # + calendar
    return raw                                                  # (T, 33) NVMD arms


# ---------------------------------------------------------------- models
class ModeLSTM(nn.Module):
    """The same head STForecaster uses, on precomputed modes."""

    def __init__(self, n_in, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return None, None, None, self.head(h[:, -1])


def set_centres(bank, centres):
    """Point a StructuredSpectralNVMD at a given set of band centres.

    `_centres_from` maps softmax(gap_logits) -> cumsum -> rescale to [0, 0.5],
    so the logits that reproduce `centres` are the logs of its successive
    differences.  Bandwidths are rebuilt from the new local gaps, exactly as
    __init__ does for the geometric layout.
    """
    c = torch.tensor(centres, dtype=torch.float32)
    d = torch.diff(c)
    g = torch.cat([d[:1], d]).clamp(min=1e-6)
    with torch.no_grad():
        bank.gap_logits.copy_(torch.log(g))
        c0 = bank._centres_from(bank.gap_logits.unsqueeze(0))[0]
        gap = torch.diff(c0, prepend=c0[:1], append=c0[-1:])
        local = 0.5 * (gap[:-1] + gap[1:])
        bank.bw_min.copy_(0.25 * local)
        bank.log_bw.copy_(torch.log(0.6 * local))


def build_model(arm, n_feat, K, L, device, args):
    if arm.startswith("vmd") or arm in ZOO:
        return ModeLSTM(n_feat).to(device)
    if arm in ("fixed_geo", "fixed_vmdmean", "nvmd_trained"):
        m = STForecaster(R=1, K=K, signal_len=L, adapt=0.0,
                         coupling=False, concat=False)
        if arm == "fixed_vmdmean":
            set_centres(m.decomposer.decomposer, VMD_MEAN_CENTRES)
        return m.to(device)
    return STForecaster(R=n_feat, K=K, signal_len=L, adapt=0.0,
                        coupling=(arm == "nvmd_st"),
                        concat=(arm == "nvmd_st")).to(device)


def build_optimiser(arm, model, args):
    band = {"decomposer.decomposer.gap_logits", "decomposer.decomposer.log_bw"}
    coup = {"decomposer.coupling.delta"}
    if arm.startswith("fixed_"):
        # The whole point of these arms: the bank never moves, so the only
        # thing separating them from VMD is that it is the *same* bank in
        # every window.
        for n, p in model.named_parameters():
            if n in band:
                p.requires_grad_(False)
        return torch.optim.AdamW(
            [{"params": [p for n, p in model.named_parameters()
                         if n not in band | coup],
              "lr": args.lr, "weight_decay": 1e-5}])
    if arm.startswith("vmd") or arm in ZOO:
        groups = [{"params": list(model.parameters()), "lr": args.lr,
                   "weight_decay": 1e-5}]
    else:
        groups = [
            {"params": [p for n, p in model.named_parameters()
                        if n not in band | coup],
             "lr": args.lr, "weight_decay": 1e-5},
            {"params": [p for n, p in model.named_parameters() if n in band],
             "lr": args.band_lr, "weight_decay": 0.0},
        ]
        if arm == "nvmd_st":
            groups.append({"params": [p for n, p in model.named_parameters()
                                      if n in coup],
                           "lr": args.coupling_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


# ---------------------------------------------------------------- loop
@torch.no_grad()
def evaluate(model, loader, device, mu, sd):
    model.eval()
    sae = sse = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)[3]
        pr, yr = p * sd + mu, y * sd + mu
        sae += (pr - yr).abs().sum().item()
        sse += ((pr - yr) ** 2).sum().item()
        n += x.size(0)
    return {"mae": sae / max(n, 1), "rmse": math.sqrt(sse / max(n, 1))}


def run_arm(arm, seed, data, args, device):
    set_seed(seed)
    tr_dl = DataLoader(data["train"], batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(data["val"], batch_size=args.batch, shuffle=False)
    te_dl = DataLoader(data["test"], batch_size=args.batch, shuffle=False)

    model = build_model(arm, data["n_feat"], args.K, args.seq_len, device, args)
    opt = build_optimiser(arm, model, args)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)
    mu, sd = data["y_mu"], data["y_sd"]

    # Two checkpoints, two numbers, both reported.
    #
    #   test_mae        weights chosen by the validation tail of the train year
    #   test_mae_cherry weights chosen by the test year itself, i.e. the
    #                   minimum over epochs -- the statistic section 13.3 and
    #                   benchmark_seeds.py report
    #
    # Neither is a guarantee of generalisation on one test year and one
    # region.  The *gap* between them is the size of the selection effect,
    # which is worth reporting on its own.
    best_val, val_state, sel_ep, stale = float("inf"), None, 0, 0
    best_test, cherry_ep = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            p = model(x)[3]
            # The objective and the headline metric have to agree.  Training on
            # MSE while selecting and reporting MAE lets any arm with spare
            # capacity spend it on the tail -- which is exactly the
            # MAE-worse/RMSE-better signature the spatial arm kept showing.
            if args.loss == "mse":
                loss = F.mse_loss(p, y)
            elif args.loss == "l1":
                loss = F.l1_loss(p, y)
            else:                                   # huber
                loss = F.smooth_l1_loss(p, y, beta=args.huber_beta)
            if arm.startswith("nvmd"):
                dec = model.decomposer.decomposer
                loss = loss + 0.05 * dec.bandwidth_loss(x[:, :1]) \
                            + 1.0 * dec.separation_loss(x[:, :1])
                if arm == "nvmd_st" and args.w_sparse:
                    loss = loss + args.w_sparse * \
                        model.decomposer.coupling.sparsity_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        v = evaluate(model, va_dl, device, mu, sd)
        te = evaluate(model, te_dl, device, mu, sd)
        flag = ""
        if v["mae"] < best_val - 1e-6:
            best_val, sel_ep, stale = v["mae"], ep, 0
            val_state = {k: t_.detach().cpu().clone()
                         for k, t_ in model.state_dict().items()}
            flag = "  <- val-selected"
        else:
            stale += 1
        if te["mae"] < best_test:
            best_test, cherry_ep = te["mae"], ep
            cherry = te
        print(f"  [{arm} s{seed} {ep:02d}/{args.epochs}] "
              f"val {v['mae']:.3f} | test {te['mae']:.3f} "
              f"({time.time()-t0:.0f}s){flag}", flush=True)
        if stale >= args.patience:
            print("  early stop", flush=True)
            break

    model.load_state_dict(val_state)
    t = evaluate(model, te_dl, device, mu, sd)
    return {"arm": arm, "seed": seed, "val_mae": best_val,
            "test_mae": t["mae"], "test_rmse": t["rmse"], "epoch": sel_ep,
            "test_mae_cherry": best_test, "test_rmse_cherry": cherry["rmse"],
            "epoch_cherry": cherry_ep}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_2018_2022.csv")
    ap.add_argument("--modes", default="cache/vmd_panel_K8_a1000_W96")
    ap.add_argument("--train-year", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2019)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--vmd-window", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--band-lr", type=float, default=3e-4)
    ap.add_argument("--coupling-lr", type=float, default=1e-2)
    ap.add_argument("--w-sparse", type=float, default=0.01)
    ap.add_argument("--loss", default="mse", choices=["mse", "huber", "l1"],
                    help="training objective; mse reproduces the original runs")
    ap.add_argument("--huber-beta", type=float, default=1.0,
                    help="Huber crossover, in standardized target units")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arms", default=",".join(ARMS))
    # CPU by default: MPS has no rfft, so the NVMD arms cannot run there, and
    # the arms must share a device.  This also matches how the numbers already
    # in attic/RESULTS-superseded.md were produced.
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=6,
                    help="torch intra-op threads; leave cores for the desktop")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any completed runs in --out and redo them")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="cap windows per split; for validating the pipeline cheaply")
    ap.add_argument("--out", default="results/three_arms_results.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = args.device
    df = pd.read_csv(args.panel, parse_dates=["SETTLEMENTDATE"])
    chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
    decomp = [c for c in chans if not c.startswith(CAL_PREFIX)]
    cal_idx = [i for i, c in enumerate(chans) if c.startswith(CAL_PREFIX)]
    price_pos = decomp.index(chans[0])

    L, W, h = args.seq_len, args.vmd_window, args.horizon
    # A window [s, s+L) needs modes at s .. s+L-1, and causal VMD's first mode
    # is at row W-1, so s >= W-1 for every arm.
    s_min = W - 1
    arms = args.arms.split(",")
    need_modes = any(a.startswith("vmd") for a in arms)
    zoo_needed = [a for a in arms if a in ZOO]

    raw, modes, starts, n_rows = {}, {}, {}, {}
    for tag, year in (("train", args.train_year), ("test", args.test_year)):
        raw[tag] = load_year(df, year, chans)
        modes[tag] = (load_modes(args.modes, year, decomp) if need_modes
                      else np.zeros((len(raw[tag]), len(decomp), args.K)))
        T = raw[tag].shape[0]
        starts[tag] = np.arange(s_min, T - L - (h - 1))
        n_rows[tag] = T

    print(f"device={device} threads={args.threads} | {len(chans)} channels, target={chans[0]}, "
          f"{len(decomp)} decomposed")
    print(f"train {args.train_year}: {n_rows['train']} rows -> "
          f"{len(starts['train'])} windows | "
          f"test {args.test_year}: {n_rows['test']} rows -> "
          f"{len(starts['test'])} windows")
    print(f"window ends at row >= {s_min + L - 1} in both years "
          f"(causal VMD needs {W-1} rows of history before its first mode)\n",
          flush=True)

    # Validation is the tail of the train year, with an L-window embargo so no
    # validation input window overlaps a training target.
    n_val = int(len(starts["train"]) * args.val_frac)
    va_starts = starts["train"][-n_val:]
    tr_starts = starts["train"][:-(n_val + L)]
    if args.max_windows:
        n = args.max_windows
        tr_starts = tr_starts[:n]
        va_starts = va_starts[:max(n // 4, 1)]
        starts["test"] = starts["test"][:max(n // 4, 1)]
        print(f"[--max-windows {n}] pipeline check only, numbers are meaningless\n")

    # Resume: every (arm, seed) already in --out is kept and skipped.  The
    # training stage is hours long, so a kill should cost one run, not all of
    # them.
    if args.max_windows and not args.out.endswith(".check.json"):
        args.out = args.out.replace(".json", "") + ".check.json"
        print(f"[--max-windows] results redirected to {args.out}")
    results = []
    if os.path.exists(args.out) and not args.fresh:
        try:
            results = json.load(open(args.out))
            done = {(r["arm"], r["seed"]) for r in results}
            print(f"resuming from {args.out}: {len(done)} run(s) already done")
        except Exception as e:
            print(f"could not read {args.out} ({e}); starting fresh")
            results = []
    done = {(r["arm"], r["seed"]) for r in results}

    # Build every arm's dataset up front, then loop seeds on the OUTSIDE.  Arm
    # order alone would mean the head-to-head table only exists after all three
    # seeds of the first arm; this way one complete row of all four arms lands
    # after four runs.
    built = {}
    zoo = {a: {tag: load_zoo(a, y, args.K, W)
               for tag, y in (("train", args.train_year), ("test", args.test_year))}
           for a in zoo_needed}

    for arm in arms:
        layout = "CT" if arm.startswith("nvmd") or arm.startswith("fixed") else "TC"
        zt = zoo[arm]["train"] if arm in ZOO else None
        ze = zoo[arm]["test"] if arm in ZOO else None
        f_tr = build_features(arm, raw["train"], modes["train"], cal_idx, price_pos, zt)
        f_te = build_features(arm, raw["test"], modes["test"], cal_idx, price_pos, ze)
        # statistics from the train year only, over rows any arm can use
        stat = f_tr[s_min:]
        mu, sd = stat.mean(0), stat.std(0) + 1e-8
        y_tr_raw = raw["train"][:, 0]
        y_mu = y_tr_raw[s_min:].mean()
        y_sd = y_tr_raw[s_min:].std() + 1e-8
        nf_tr, nf_te = (f_tr - mu) / sd, (f_te - mu) / sd
        y_tr = (y_tr_raw - y_mu) / y_sd
        y_te = (raw["test"][:, 0] - y_mu) / y_sd

        data = {
            "train": ArmDataset(nf_tr, y_tr, tr_starts, L, h, layout),
            "val":   ArmDataset(nf_tr, y_tr, va_starts, L, h, layout),
            "test":  ArmDataset(nf_te, y_te, starts["test"], L, h, layout),
            "n_feat": nf_tr.shape[1], "y_mu": y_mu, "y_sd": y_sd,
        }
        print(f"== {arm}: {data['n_feat']} input channels, "
              f"train {len(tr_starts)} / val {len(va_starts)} / "
              f"test {len(starts['test'])} windows", flush=True)
        built[arm] = data

    for seed in [int(v) for v in args.seeds.split(",")]:
        for arm in arms:
            data = built[arm]
            if (arm, seed) in done:
                prev = next(r for r in results
                            if r["arm"] == arm and r["seed"] == seed)
                print(f"  {arm} seed {seed}: cached, test MAE "
                      f"{prev['test_mae']:.3f}", flush=True)
                continue
            r = run_arm(arm, seed, data, args, device)
            r["n_feat"] = data["n_feat"]
            results.append(r)
            done.add((arm, seed))
            print(f"  -> {arm} seed {seed}: test MAE {r['test_mae']:.3f} "
                  f"(epoch {r['epoch']}, val {r['val_mae']:.3f})\n", flush=True)
            dump_atomic(results, args.out)
            os.system(f"python3 report_three_arms.py --results {args.out} "
                      f"--out THREE_ARMS.md >/dev/null 2>&1")

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 68)
    print(f"{'arm':<16}{'n_in':>6}{'test MAE':>22}{'test RMSE':>16}")
    print("-" * 68)
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        if not rs:
            continue
        m = np.array([r["test_mae"] for r in rs])
        rm = np.array([r["test_rmse"] for r in rs])
        print(f"{arm:<16}{rs[0]['n_feat']:>6}"
              f"{m.mean():>15.3f} +/- {m.std(ddof=1) if len(m)>1 else 0:.3f}"
              f"{rm.mean():>12.3f} +/- {rm.std(ddof=1) if len(rm)>1 else 0:.3f}")
    print("=" * 68)
    dump_atomic(results, args.out)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
