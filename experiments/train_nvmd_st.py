#!/usr/bin/env python3
"""
Train spatio-temporal NVMD and its exact temporal-only ablation.

`--coupling 0` freezes A_k at the identity, which reduces the model to
temporal-only NVMD with an identical output shape and an identical head.  The
two runs therefore differ in exactly one thing: whether cross-region mixing is
allowed.

    python train_nvmd_st.py --coupling 1 --outdir runs_st_on
    python train_nvmd_st.py --coupling 0 --outdir runs_st_off
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.nvmd_st import STForecaster
from experiments.train_nvmd_v2 import set_seed

REGIONS = ["SA1", "NSW1", "VIC1", "QLD1", "TAS1"]


class PanelWindowDataset(Dataset):
    """x: (R, L) window across regions.  y: next-step target-region value."""

    def __init__(self, arr: np.ndarray, seq_len: int, target: int = 0,
                 mean=None, std=None, horizon: int = 1):
        if mean is None:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-8
        self.mean, self.std = mean, std
        self.data = torch.from_numpy(((arr - mean) / std).astype(np.float32))
        self.L = seq_len
        self.target = target
        # horizon h predicts h steps past the end of the window.  h=1 is the
        # original next-step task, where a persistence baseline scores 14.40 and
        # causal VMD + LSTM scores 14.43 -- i.e. the task is saturated.  The
        # headroom is at h>=6 (naive floor ~33), so the horizon is a knob now.
        self.h = horizon
        self.N = max(0, len(self.data) - seq_len - (horizon - 1))

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.data[i:i + self.L].T                      # (R, L)
        y = self.data[i + self.L + self.h - 1, self.target].unsqueeze(0)
        return x, y


@torch.no_grad()
def evaluate(model, loader, device, mean, std, target=0):
    model.eval()
    sae = sse = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, _, _, p = model(x)
        pr = p * std[target] + mean[target]
        yr = y * std[target] + mean[target]
        sae += (pr - yr).abs().sum().item()
        sse += ((pr - yr) ** 2).sum().item()
        n += x.size(0)
    d = max(n, 1)
    return {"mae": sae / d, "rmse": math.sqrt(sse / d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_2018_2022.csv")
    ap.add_argument("--train-years", default="2018")
    ap.add_argument("--test-years", default="2019")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=1,
                    help="steps ahead to predict (1 = next 30 min). "
                         "h=1 is saturated: persistence scores 14.40. "
                         "Real headroom is h>=6 (naive floor ~33).")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--adapt", type=float, default=0.0)
    ap.add_argument("--coupling", type=int, default=1)
    ap.add_argument("--xfilter", type=int, default=0,
                    help="cross-filter the two streams: K x K cross-band transfer "
                         "+ FiLM gating of own on exogenous (3K head inputs)")
    ap.add_argument("--concat", type=int, default=0,
                    help="keep the target's own modes and append the purely "
                         "exogenous mix (2K head inputs) instead of replacing them")
    ap.add_argument("--w-sparse", type=float, default=0.01,
                    help="L1 on off-diagonal coupling, keeps topology readable")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--band-lr", type=float, default=3e-4)
    ap.add_argument("--coupling-lr", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="./runs_st")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.panel, parse_dates=["SETTLEMENTDATE"])
    # Channels are every non-date column, in file order; column 0 is the target.
    chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
    globals()["REGIONS"] = chans
    tr_y = [int(v) for v in args.train_years.split(",")]
    te_y = [int(v) for v in args.test_years.split(",")]
    tr = df[df.SETTLEMENTDATE.dt.year.isin(tr_y)][chans].to_numpy(float)
    te = df[df.SETTLEMENTDATE.dt.year.isin(te_y)][chans].to_numpy(float)
    print(f"{len(chans)} channels | target = {chans[0]}")

    tr_ds = PanelWindowDataset(tr, args.seq_len, horizon=args.horizon)
    te_ds = PanelWindowDataset(te, args.seq_len, mean=tr_ds.mean,
                               std=tr_ds.std, horizon=args.horizon)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False)
    print(f"coupling={'ON' if args.coupling else 'OFF (A_k = I)'}"
          f"{' | xfilter (own + cross-band exo + gated, 3K head inputs)' if args.xfilter else (' | concat (own + exogenous, 2K head inputs)' if args.concat else '')} | "
          f"h={args.horizon} ({args.horizon*0.5:g}h ahead) | "
          f"train {len(tr_ds)} | test {len(te_ds)} windows")

    model = STForecaster(R=len(chans), K=args.K, signal_len=args.seq_len,
                         adapt=args.adapt, coupling=bool(args.coupling),
                         concat=bool(args.concat),
                         xfilter=bool(args.xfilter)).to(device)

    band = {"decomposer.decomposer.gap_logits", "decomposer.decomposer.log_bw"}
    coup = {"decomposer.coupling.delta"}
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if n not in band | coup], "lr": args.lr, "weight_decay": 1e-5},
        {"params": [p for n, p in model.named_parameters() if n in band],
         "lr": args.band_lr, "weight_decay": 0.0},
    ]
    if args.coupling:
        groups.append({"params": [p for n, p in model.named_parameters()
                                  if n in coup],
                       "lr": args.coupling_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best, best_state, stale = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = n = 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            _, _, _, p = model(x)
            loss = F.mse_loss(p, y)
            dec = model.decomposer.decomposer
            loss = loss + 0.05 * dec.bandwidth_loss(x[:, :1]) \
                        + 1.0 * dec.separation_loss(x[:, :1])
            if args.coupling and args.w_sparse:
                loss = loss + args.w_sparse * model.decomposer.coupling.sparsity_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        m = evaluate(model, te_dl, device, tr_ds.mean, tr_ds.std)
        sched.step()
        print(f"[{ep:02d}/{args.epochs}] loss={tot/max(n,1):.5f} | "
              f"test MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} | {time.time()-t0:.0f}s",
              flush=True)

        if m["mae"] < best:
            best = m["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            print(f"  -> new best {best:.2f}")
            torch.save({"model_state": best_state, "best_mae": best,
                        "args": vars(args), "regions": REGIONS,
                        "norm": {"mean": tr_ds.mean, "std": tr_ds.std}},
                       os.path.join(args.outdir, "best.pt"))
        else:
            stale += 1
            if stale >= args.patience:
                print("early stop")
                break

    print(f"\nBEST test MAE = {best:.2f}")

    if args.coupling and best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        C = model.decomposer.coupling.coupling_strength()      # (K, R, R)
        rows = model.decomposer.decomposer.describe()
        A = model.decomposer.coupling.matrices()               # (K, R, R)

        # Raw |A_k| is not comparable across bands: high-frequency modes carry
        # far less energy, so a large coefficient there moves little signal.
        # Measure instead the share of the target band's output variance that
        # actually comes from other regions.
        dec = model.decomposer.decomposer
        share = np.zeros(C.shape[0])
        with torch.no_grad():
            xb, _ = next(iter(te_dl))
            xb = xb.to(device)
            B, R, L = xb.shape
            m, _ = dec(xb.reshape(B * R, 1, L))
            m = m.reshape(B, R, args.K, L)
            for k in range(args.K):
                oth = sum(A[k, 0, j] * m[:, j, k] for j in range(1, R))
                if args.concat or args.xfilter:
                    # The two head blocks are separate inputs, so the meaningful
                    # quantity is how much of the head's input energy is the
                    # exogenous block rather than the target's own (passthrough,
                    # unscaled) modes.
                    vo, ve = m[:, 0, k].var().item(), oth.var().item()
                    share[k] = ve / max(vo + ve, 1e-12)
                else:
                    own = A[k, 0, 0] * m[:, 0, k]              # (B, L)
                    vo, vt = own.var().item(), (own + oth).var().item()
                    share[k] = max(0.0, 1.0 - vo / max(vt, 1e-12))

        # With 30+ channels the full matrix is unreadable; show the
        # strongest contributors by mean coupling across bands.
        mean_c = C[:, 0, 1:].mean(0)
        top_idx = [int(i) + 1 for i in mean_c.argsort(descending=True)[:6]]
        top = [chans[i] for i in top_idx]
        print("\nLearned coupling into %s, by band (top 6 channels)" % chans[0])
        print(f"  {'band':<6s}{'period(h)':>11s}" +
              "".join(f"{r[:8]:>9s}" for r in top) + f"{'spatial%':>11s}")
        for k in range(C.shape[0]):
            per = rows[k][3]
            ps = f"{per:.1f}" if math.isfinite(per) else "inf"
            print(f"  {k+1:<6d}{ps:>11s}" +
                  "".join(f"{C[k, 0, j]:>9.4f}" for j in top_idx) +
                  f"{100*share[k]:>10.1f}%")
        if args.concat or args.xfilter:
            print("  spatial% = exogenous block's share of the head's input "
                  "variance for that band\n  (own modes are passed through "
                  "unscaled; raw |A_k| is not comparable across bands)")
        else:
            print("  spatial% = share of the band's output variance contributed by "
                  "other regions\n  (energy-weighted; raw |A_k| is not comparable "
                  "across bands)")


if __name__ == "__main__":
    main()
