#!/usr/bin/env python3
"""
Is there spatial information, or were we looking in the wrong place?

Two factors, crossed:

    horizon    h=1   the task RESULTS.md calls saturated: persistence scores
                     14.40 and the best model 14.3, so there is ~0.1 MAE of
                     room for anything to win
               h=6   where section 11a measured the spatial coupling gain at
                     -2.21 MAE

    exogenous  back  every channel over the trailing window [i, i+L), which is
                     what PanelWindowDataset has always done
               fwd   the target's own history over [i, i+L), but the exogenous
                     channels shifted forward by h so they cover the interval
                     being predicted -- i.e. treated as a forecast, which is
                     what a real operator has and what makes weather worth
                     carrying at all

A price-only control at each horizon gives the gain that the exogenous panel
actually buys.

The forward condition uses reanalysis values at the target time.  That is an
UPPER BOUND, not a deployable number: real forecasts carry error reanalysis
does not.  It is the right measurement for "is the information there", and the
wrong one for "what would this earn".

Architecture follows the project's own thesis: decompose the *target*, because
it is being extrapolated, and leave the exogenous channels raw, because they
are only being conditioned on.

    python run_spatial_2x2.py --seeds 1,2
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
from torch.utils.data import DataLoader

from models.nvmd_v3 import StructuredSpectralNVMD
from experiments.run_three_arms import ArmDataset, dump_atomic, evaluate, set_seed

CAL = "cal_"


class BankPlusExo(nn.Module):
    """Frozen band decomposition of the target, raw exogenous alongside."""

    def __init__(self, n_exo, K=8, L=96, hidden=128, layers=2):
        super().__init__()
        self.bank = StructuredSpectralNVMD(K=K, signal_len=L, adapt=0.0)
        for p in self.bank.parameters():
            p.requires_grad_(False)
        self.lstm = nn.LSTM(K + n_exo, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):                       # x: (B, 1 + n_exo, L)
        modes, _ = self.bank(x[:, :1])          # (B, K, L)
        z = torch.cat([modes, x[:, 1:]], dim=1) if x.shape[1] > 1 else modes
        h, _ = self.lstm(z.permute(0, 2, 1))
        return None, None, None, self.head(h[:, -1])


def build(df, chans, year, h, mode):
    """-> feature matrix (T, C) with column 0 the target price.

    `mode='fwd'` shifts every exogenous column forward by h, so the window that
    ends at the last observed target step carries exogenous values through the
    step being predicted.
    """
    a = df[df.SETTLEMENTDATE.dt.year == year][chans].to_numpy(np.float64)
    if mode == "price":
        return a[:, :1]
    if mode == "back":
        return a
    exo = np.roll(a[:, 1:], -h, axis=0)
    exo[-h:] = np.nan                            # no future rows exist there
    return np.concatenate([a[:, :1], exo], axis=1)


def run(cfg, seed, data, args, device):
    set_seed(seed)
    dl = {k: DataLoader(v, batch_size=args.batch, shuffle=(k == "train"))
          for k, v in data["ds"].items()}
    model = BankPlusExo(data["n_exo"], args.K, args.seq_len).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    mu, sd = data["y_mu"], data["y_sd"]
    best, state, sel, stale = float("inf"), None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x)[3], y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        v = evaluate(model, dl["val"], device, mu, sd)
        flag = ""
        if v["mae"] < best - 1e-6:
            best, sel, stale = v["mae"], ep, 0
            state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            flag = "  <- selected"
        else:
            stale += 1
        print(f"  [{cfg} s{seed} {ep:02d}/{args.epochs}] val {v['mae']:.3f} "
              f"({time.time()-t0:.0f}s){flag}", flush=True)
        if stale >= args.patience:
            print("  early stop", flush=True); break
    model.load_state_dict(state)
    t = evaluate(model, dl["test"], device, mu, sd)
    return {"cfg": cfg, "seed": seed, "val_mae": best, "test_mae": t["mae"],
            "test_rmse": t["rmse"], "epoch": sel, "n_exo": data["n_exo"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_2018_2022.csv")
    ap.add_argument("--train-year", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2019)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--horizons", default="1,6")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out", default="results/spatial_2x2_results.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.panel, parse_dates=["SETTLEMENTDATE"])
    chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
    L = args.seq_len

    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        print(f"resuming: {len(results)} run(s) done")
    done = {(r["cfg"], r["seed"]) for r in results}

    for h in [int(v) for v in args.horizons.split(",")]:
        for mode in ("price", "back", "fwd"):
            cfg = f"h{h}_{mode}"
            f_tr = build(df, chans, args.train_year, h, mode)
            f_te = build(df, chans, args.test_year, h, mode)
            # a window [s, s+L) plus its target must avoid the NaN tail the
            # forward shift leaves behind
            T_tr, T_te = len(f_tr), len(f_te)
            s_tr = np.arange(0, T_tr - L - h - (h if mode == "fwd" else 0))
            s_te = np.arange(0, T_te - L - h - (h if mode == "fwd" else 0))
            n_val = int(len(s_tr) * args.val_frac)
            tr, va = s_tr[:-(n_val + L)], s_tr[-n_val:]

            mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-8
            y_mu, y_sd = f_tr[:, 0].mean(), f_tr[:, 0].std() + 1e-8
            ntr, nte = (f_tr - mu) / sd, (f_te - mu) / sd
            ytr = (f_tr[:, 0] - y_mu) / y_sd
            yte = (f_te[:, 0] - y_mu) / y_sd

            data = {"ds": {
                "train": ArmDataset(ntr, ytr, tr, L, h, "CT"),
                "val":   ArmDataset(ntr, ytr, va, L, h, "CT"),
                "test":  ArmDataset(nte, yte, s_te, L, h, "CT")},
                "n_exo": f_tr.shape[1] - 1, "y_mu": y_mu, "y_sd": y_sd}
            print(f"== {cfg}: {data['n_exo']} exogenous channels, "
                  f"train {len(tr)} / val {len(va)} / test {len(s_te)}", flush=True)

            for seed in [int(v) for v in args.seeds.split(",")]:
                if (cfg, seed) in done:
                    print(f"  {cfg} s{seed}: cached", flush=True); continue
                r = run(cfg, seed, data, args, device)
                results.append(r); done.add((cfg, seed))
                print(f"  -> {cfg} s{seed}: test MAE {r['test_mae']:.3f}\n", flush=True)
                dump_atomic(results, args.out)

    print("\n" + "=" * 60)
    print(f"{'config':>12}{'test MAE':>20}{'gain vs price-only':>22}")
    print("-" * 60)
    for h in [int(v) for v in args.horizons.split(",")]:
        base = np.mean([r["test_mae"] for r in results if r["cfg"] == f"h{h}_price"] or [np.nan])
        for mode in ("price", "back", "fwd"):
            rs = [r["test_mae"] for r in results if r["cfg"] == f"h{h}_{mode}"]
            if not rs:
                continue
            m = np.mean(rs)
            g = "--" if mode == "price" else f"{m - base:+.3f}"
            sd_ = np.std(rs, ddof=1) if len(rs) > 1 else 0.0
            print(f"{'h%d_%s' % (h, mode):>12}{m:>13.3f} ± {sd_:.3f}{g:>22}")
    print("=" * 60)


if __name__ == "__main__":
    main()
