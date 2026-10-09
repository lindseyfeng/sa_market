#!/usr/bin/env python3
"""Joint multi-region prediction, and the two arms that keep it interpretable.

    single            1 target (SA1), the same 38-channel panel
    joint_nocouple    5 targets, A_k frozen at I
    joint             5 targets, A_k learned

    single -> joint_nocouple   the multi-task effect (5x supervision)
    joint_nocouple -> joint    the spatial effect (gradient on the coupling)

Splitting it this way is the point. Joint training does both at once, and a
single joint-vs-single margin cannot say which one paid. Every arm shares the
input panel, the window list, the bank, the trunk size, the objective and the
seed, so the arm name is the only thing that differs.

Protocol follows run_three_arms exactly: window 96, h=1, asinh around the train
median scaled by the train IQR, Huber, selection on a validation tail of the
train years with an L-window embargo, test scored once from those weights. Each
region is transformed and standardised with its *own* train statistics -- TAS1
and QLD1 differ by 12 $/MWh in mean and 6 in sd, and one shared scale would let
the loss be dominated by whichever region happens to be widest.
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.nvmd_joint import JointForecaster
from experiments.run_three_arms import set_seed, dump_atomic, _mask

ARMS = ["single", "joint_nocouple", "joint"]
REGIONS = ["SA1_price", "NSW1_price", "VIC1_price", "QLD1_price", "TAS1_price"]

# Physical drivers, as auxiliary targets. Calendar channels are deterministic and
# would hand the loss a free ride; the spreads are linear combinations of the
# price targets and carry nothing the levels do not; demand_NEM is the sum of the
# other five. What is left is the state of the system the price responds to.
AUX = ["demand_SA1", "demand_NSW1", "demand_VIC1", "demand_QLD1", "demand_TAS1",
       "wind100_adelaide", "wind100_nsa_wind", "wind100_sesa_wind",
       "wind100_melbourne", "ramp_SA1", "ramp_VIC1", "scarcity_SA1"]


class JointDataset(Dataset):
    """Window (C, ctx) ending at s+L-1 -> targets (T,) `h` steps past it.

    `ctx` > L hands the bank a longer view while the prediction point and the
    scored row are unchanged, so a long-context arm and a short-context arm are
    scored on exactly the same rows and a paired test between them is valid.
    """

    def __init__(self, feat, target, starts, L, h, ctx=None):
        self.x = torch.from_numpy(feat.astype(np.float32))
        self.y = torch.from_numpy(target.astype(np.float32))      # (rows, T)
        self.starts, self.L, self.h = starts, L, h
        self.ctx = ctx or L

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        a = s + self.L - self.ctx
        assert a >= 0, "window start is negative; raise --skip-head"
        return self.x[a:s + self.L].T, self.y[s + self.L + self.h - 1]


class TargetScaler:
    """asinh + standardise per region, with the inverse used for every metric.

    Metrics are only ever taken in $/MWh. A number in the transformed space is
    not comparable with anything in FINDINGS, and asinh is not a monotone
    rescaling of MAE, so reporting there would not even rank arms the same way.
    """

    def __init__(self, y, s_min, transform="asinh", plain_from=None):
        """`plain_from`: columns at or after this index are standardised only.

        asinh is a variance-stabiliser chosen for a target with kurtosis ~1100
        and 11% negative values. Demand and wind are neither, so applying it to
        them would be a claim about their distribution that nobody has checked.
        """
        ref = y[s_min:]
        self.transform = transform
        n = y.shape[1]
        self.asinh_mask = np.ones(n, bool)
        if plain_from is not None:
            self.asinh_mask[plain_from:] = False
        if transform == "asinh":
            self.c = np.where(self.asinh_mask, np.median(ref, axis=0), 0.0)
            iqr = (np.percentile(ref, 75, axis=0)
                   - np.percentile(ref, 25, axis=0)) + 1e-8
            self.w = np.where(self.asinh_mask, iqr, 1.0)
            z = np.where(self.asinh_mask, np.arcsinh((y - self.c) / self.w), y)
        else:
            self.c = np.zeros(n)
            self.w = np.ones(n)
            self.asinh_mask[:] = False
            z = y.copy()
        self.mu = z[s_min:].mean(axis=0)
        self.sd = z[s_min:].std(axis=0) + 1e-8
        # sinh overflows float32 past ~88: an untrained prediction that wanders
        # turns the metric into inf and then nan. Clamp to the range the training
        # target occupies plus a margin; outside it is not a price this market
        # has produced.
        self.lo = z[s_min:].min(axis=0) - 2.0
        self.hi = z[s_min:].max(axis=0) + 2.0

    def forward(self, y):
        z = (np.where(self.asinh_mask, np.arcsinh((y - self.c) / self.w), y)
             if self.transform == "asinh" else y)
        return (z - self.mu) / self.sd

    def inverse(self, p):
        """p: (B, T) standardised -> $/MWh"""
        dev = p.device
        t = lambda v: torch.as_tensor(v, dtype=p.dtype, device=dev)
        z = p * t(self.sd) + t(self.mu)
        if self.transform != "asinh":
            return z
        m = t(self.asinh_mask.astype(np.float32))
        inv = torch.sinh(z.clamp(t(self.lo), t(self.hi))) * t(self.w) + t(self.c)
        return m * inv + (1.0 - m) * z


def deviation(y):
    """Each region's departure from the cross-region mean.

    96.4% of the variance across the five regions is the common NEM mode and
    only 3.6% is regional -- measured on 2021. A model that predicts the common
    mode perfectly and reports the same number for all five regions scores a
    mean MAE of 20.66, against 29.75 for persistence. So a plain Huber on the
    levels can be won outright without representing any spatial structure at
    all, and a joint-vs-coupled margin taken under it would be measuring the
    common mode in both arms.

    This is the term that makes the spatial claim testable: the deviations are a
    deterministic function of the targets, so nothing is added to the inputs --
    the loss is simply reweighted onto the 3.6% that is actually regional.
    `--w-dev 0` recovers the plain objective exactly, which is the ablation.
    """
    return y - y.mean(dim=1, keepdim=True)


def _elem(args):
    """The pointwise loss, matched to the metric.

    The metric is MAE, so the loss should be L1. Huber at beta = 1.0 in
    standardised units is not a mild robustification of that: measured on a
    trained model's residuals, 87% of them fall inside |r| < 1, so the objective
    was quadratic for the bulk of the data while every reported number was an
    absolute error. FINDINGS calls this out as the project's own worst confound
    -- "the objective is a confound, not a hyperparameter" -- and records that
    matching it moved nvmd_st 0.442 MAE and flipped claim 5 from lose-by-0.098
    to win-by-0.334.

    The repo had already measured the ordering and it is monotone in beta:
    L1 13.744, beta 0.5 13.909, beta 2.0 14.067, beta 4.0 14.234
    (results/beta_*.json, nvmd_st, 2019, h=1). L1 wins, so L1 is the default.
    """
    if args.loss == "l1":
        return lambda a, b: F.l1_loss(a, b)
    if args.loss == "mse":
        return lambda a, b: F.mse_loss(a, b)
    return lambda a, b: F.smooth_l1_loss(a, b, beta=args.huber_beta)


def joint_loss(p, y, args, model, arm, dev_sd, n_price):
    """The primary term, plus every regulariser this problem actually argues for.

    Each term is reported as well as summed, because a term whose magnitude is
    never looked at is a term nobody can ablate honestly.
    """
    el = _elem(args)
    pp, yp = p[:, :n_price], y[:, :n_price]
    parts = {"price": el(pp, yp)}

    if args.w_dev:
        # Standardised by the deviations' own train scale, so w_dev = 1 means
        # "get the regional structure as right as the level", not "add 3.6% of a
        # gradient". Without this normalisation the weight would be meaningless.
        d = (deviation(pp) - deviation(yp)) / dev_sd
        parts["dev"] = args.w_dev * el(d, torch.zeros_like(d))

    if args.w_aux and p.shape[1] > n_price:
        parts["aux"] = args.w_aux * el(p[:, n_price:], y[:, n_price:])

    if args.w_bias:
        # The model is known to carry a constant offset: on 2021 nvmd_st ran
        # 5.65 low against AR(48)'s 3.71, which is why `--bias-correct` exists
        # as a post-hoc median shift. Penalising the batch-mean residual pays
        # for it during training instead of patching it afterwards.
        parts["bias"] = args.w_bias * (pp - yp).mean(dim=0).abs().mean()

    if arm == "joint" and args.w_sparse:
        parts["sparse"] = args.w_sparse * model.decomposer.coupling.sparsity_loss()

    dec = model.decomposer.decomposer
    return sum(parts.values()), parts


def evaluate(model, loader, device, scaler, names, n_price=None, cut=None):
    """Metrics in $/MWh, over the price targets only."""
    n_price = len(names) if n_price is None else n_price
    model.eval()
    sae = sse = None
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = (cut(x) if cut else x).to(device)
            y = y.to(device)
            p = model(x)[3]
            pr, yr = scaler.inverse(p)[:, :n_price], scaler.inverse(y)[:, :n_price]
            a = (pr - yr).abs().sum(0)
            s = ((pr - yr) ** 2).sum(0)
            sae = a if sae is None else sae + a
            sse = s if sse is None else sse + s
            n += x.size(0)
    mae = (sae / max(n, 1)).cpu().numpy()
    rmse = np.sqrt((sse / max(n, 1)).cpu().numpy())
    return {"mae": {k: float(v) for k, v in zip(names[:n_price], mae)},
            "rmse": {k: float(v) for k, v in zip(names[:n_price], rmse)},
            "mae_mean": float(mae.mean()), "mae_sa1": float(mae[0])}


def run_arm(arm, seed, data, args, device):
    set_seed(seed)
    _ = device
    nw = args.workers if args.workers is not None else (4 if device.type == "cuda" else 0)
    dkw = dict(num_workers=nw, pin_memory=(device.type == "cuda"),
               persistent_workers=nw > 0)
    dl = {k: DataLoader(data[k], batch_size=args.batch, shuffle=(k == "train"), **dkw)
          for k in ("train", "val", "test")}
    # "joint@0.01" is the joint arm with its off-diagonal coupling seeded at
    # N(0, 0.01). Carrying it in the arm name rather than a global flag lets both
    # initialisations share one dataset build, one window list and one resume
    # file, so the only thing that differs between them is the seeding.
    # Arm names carry their own configuration so one invocation, one dataset
    # build and one resume file cover the whole comparison:
    #   joint              lstm head, zero-init coupling
    #   joint/linear       linear head over the whole band stack
    #   joint/linear@0.01  the same with the coupling seeded
    spec = arm.split("@", 1)
    c_init = float(spec[1]) if len(spec) > 1 else args.coupling_init
    parts = spec[0].split("/", 1)
    tag = parts[1] if len(parts) > 1 else args.head
    bits = tag.split("+")
    head = bits[0] or args.head
    xfilter = "film" in bits
    ctx = next((int(b[3:]) for b in bits if b.startswith("ctx")), args.context)
    base = parts[0]
    # "panel" is the no-decomposition control: the same head reading the raw
    # window. It is the only arm that isolates the decomposition, because a band
    # decomposition is an invertible linear map -- ridge on bands and ridge on
    # the raw window give identical predictions to 0.0000 $/MWh, measured -- so
    # nothing linear can tell them apart and `arx_window` is not this control.
    decompose = base != "panel"
    if arm.startswith("single"):
        # "single" is SA1; "single:VIC1_price" is the single-task baseline for
        # that region. Multi-task results are only interpretable against a
        # single-task baseline *per task*, so every region needs its own.
        reg = arm.split(":", 1)[1] if ":" in arm else data["names"][0]
        j = data["names"].index(reg)
        targets, names, n_price = [data["targets"][j]], [reg], 1
    else:
        targets = data["targets"] + (data["aux_idx"] if args.w_aux else [])
        names = data["names"] + (data["aux_names"] if args.w_aux else [])
        n_price = len(data["names"])
        reg = None
    model = JointForecaster(
        C=data["n_feat"], targets=targets, K=args.K, signal_len=args.seq_len,
        lstm_hidden=args.hidden, adapt=0.0, coupling=(base == "joint"),
        coupling_init=(c_init if base == "joint" else 0.0), head=head,
        decompose=decompose, n_chan=data["n_feat"],
        xfilter=xfilter, context=ctx,
    ).to(device)
    # "joint@0.01" is the joint arm with its off-diagonal coupling seeded at
    # N(0, 0.01). Carrying it in the arm name rather than a global flag lets both
    # initialisations share one dataset build, one window list and one resume
    # file, so the only thing that differs between them is the seeding.
    # Arm names carry their own configuration so one invocation, one dataset
    # build and one resume file cover the whole comparison:
    #   joint              lstm head, zero-init coupling
    #   joint/linear       linear head over the whole band stack
    #   joint/linear@0.01  the same with the coupling seeded
    spec = arm.split("@", 1)
    c_init = float(spec[1]) if len(spec) > 1 else args.coupling_init
    parts = spec[0].split("/", 1)
    tag = parts[1] if len(parts) > 1 else args.head
    bits = tag.split("+")
    head = bits[0] or args.head
    xfilter = "film" in bits
    ctx = next((int(b[3:]) for b in bits if b.startswith("ctx")), args.context)
    base = parts[0]
    # "panel" is the no-decomposition control: the same head reading the raw
    # window. It is the only arm that isolates the decomposition, because a band
    # decomposition is an invertible linear map -- ridge on bands and ridge on
    # the raw window give identical predictions to 0.0000 $/MWh, measured -- so
    # nothing linear can tell them apart and `arx_window` is not this control.
    decompose = base != "panel"
    if arm.startswith("single"):
        scaler = data["scaler_one"][reg]
        dl = {k: DataLoader(data["one"][reg][k], batch_size=args.batch,
                            shuffle=(k == "train"), **dkw)
              for k in ("train", "val", "test")}
    elif args.w_aux:
        scaler = data["scaler_aux"]
        dl = {k: DataLoader(data[k + "_aux"], batch_size=args.batch,
                            shuffle=(k == "train"), **dkw)
              for k in ("train", "val", "test")}
    else:
        scaler = data["scaler"]
    dev_sd = torch.as_tensor(data["dev_sd"], dtype=torch.float32, device=device)
    want_ctx = ctx or args.seq_len
    # Every arm is served the longest window any arm in the run asked for; this
    # arm takes the tail it needs, so the scored rows stay identical across arms.
    cut = (lambda t: t[..., -want_ctx:]) if want_ctx < data["ctx_max"] else (lambda t: t)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    best_val, state, sel_ep, stale = float("inf"), None, 0, 0
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); seen, nb = {}, 0
        for x, y in dl["train"]:
            x, y = cut(x).to(device), y.to(device)
            p = model(x)[3]
            loss, parts = joint_loss(p, y, args, model, arm, dev_sd, n_price)
            if decompose:
                dec = model.decomposer.decomposer
                loss = loss + 0.05 * dec.bandwidth_loss(x[:, :1]) \
                            + 1.0 * dec.separation_loss(x[:, :1])
            for k_, v_ in parts.items():
                seen[k_] = seen.get(k_, 0.0) + float(v_)
            nb += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        # Selection reads the price MAE only. Auxiliary targets are there to
        # shape the representation, not to be forecast; selecting on them would
        # pick the epoch that is best at predicting wind.
        v = evaluate(model, dl["val"], device, scaler, names, n_price, cut)
        te = evaluate(model, dl["test"], device, scaler, names, n_price, cut)
        flag = ""
        if v["mae_mean"] < best_val - 1e-6:
            best_val, sel_ep, stale = v["mae_mean"], ep, 0
            state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            flag = "  <- val-selected"
        else:
            stale += 1
        terms = " ".join(f"{k} {v_/max(nb,1):.3f}" for k, v_ in seen.items())
        hist.append({"epoch": ep, "val": v["mae_mean"], "test": te["mae_mean"],
                     "loss_terms": {k: v_ / max(nb, 1) for k, v_ in seen.items()}})
        print(f"  [{arm} s{seed} {ep:02d}/{args.epochs}] val {v['mae_mean']:.3f} | "
              f"test {te['mae_mean']:.3f} (SA1 {te['mae_sa1']:.3f}) "
              f"| {terms} ({time.time()-t0:.0f}s){flag}", flush=True)
        if stale >= args.patience:
            print("  early stop", flush=True); break

    model.load_state_dict(state)
    t = evaluate(model, dl["test"], device, scaler, names, n_price, cut)

    if args.save_preds:
        # Saved so the evaluation can be done the way the field does it --
        # rMAE against an out-of-sample naive, and a Diebold-Mariano test on the
        # loss differential. Neither can be computed from a summary number, and
        # with seed noise this large a margin without a test is not evidence.
        os.makedirs(args.save_preds, exist_ok=True)
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y in dl["test"]:
                x = cut(x).to(device)
                P.append(scaler.inverse(model(x)[3])[:, :n_price].cpu().numpy())
                Y.append(scaler.inverse(y.to(device))[:, :n_price].cpu().numpy())
        np.savez_compressed(
            os.path.join(args.save_preds,
                         re.sub(r"[^A-Za-z0-9._-]", "_", arm) + f"_s{seed}.npz"),
            pred=np.concatenate(P), truth=np.concatenate(Y),
            start=np.asarray(dl["test"].dataset.starts), names=np.array(names))
    out = {"arm": arm, "seed": seed, "val_mae": best_val, "epoch": sel_ep,
           "test_mae_mean": t["mae_mean"], "test_mae_sa1": t["mae_sa1"],
           "test_mae": t["mae"], "test_rmse": t["rmse"], "history": hist,
           "n_feat": data["n_feat"], "n_targets": len(targets),
           "n_price": n_price,
           "head": tag, "context": ctx, "xfilter": xfilter,
           "loss": {"kind": args.loss, "coupling_init": c_init,
                    "w_dev": args.w_dev, "w_aux": args.w_aux,
                    "w_bias": args.w_bias, "w_sparse": args.w_sparse,
                    "huber_beta": args.huber_beta}}
    out["decompose"] = decompose
    if base == "joint" and decompose:
        out["coupling"] = coupling_report(model, data, names, targets)
    return out


def coupling_report(model, data, names, targets):
    """What the learned coupling actually is, not what it is hoped to be.

    The uncomfortable half of claim 6 is that one channel, ramp_VIC1, carries
    73-83% of the whole exogenous effect. With a single target and 256 live
    parameters there was nothing to stop that. The number to watch here is
    `top1_share`: the fraction of a target row's off-diagonal mass sitting on its
    single largest source channel. If joint training is doing its job the mass
    spreads, and if it is not, this says so in one number per region.
    """
    A = model.decomposer.coupling.matrices().detach()        # (K, C, C)
    chans = data["chans"]
    rep = {}
    for t_i, nm in enumerate(names):
        c = targets[t_i]
        w = A[:, c, :].abs().clone()                         # (K, C)
        w[:, c] = 0.0                                        # drop the self term
        per_chan = w.sum(0)                                  # (C,)
        tot = float(per_chan.sum()) + 1e-12
        order = torch.argsort(per_chan, descending=True)[:6]
        rep[nm] = {
            "offdiag_mass": tot,
            "top1_share": float(per_chan[order[0]] / tot),
            "top3_share": float(per_chan[order[:3]].sum() / tot),
            "top": [[chans[int(j)], float(per_chan[int(j)] / tot)] for j in order],
            # Per-band mass: claim 6 says coupling is scale-dependent, so a flat
            # profile across k would be evidence against it, not for it.
            "per_band": [float(v) for v in w.sum(1)],
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_joint_2018_2022.csv")
    ap.add_argument("--train-year", default="2018,2019,2020")
    ap.add_argument("--test-year", default="2021")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="l1", choices=["l1", "huber", "mse"],
                    help="pointwise loss. l1 by default because the metric is "
                         "MAE and the repo's own beta sweep is monotone: L1 "
                         "13.744 against 14.234 at beta 4.0")
    ap.add_argument("--huber-beta", type=float, default=0.2,
                    help="Huber crossover in standardised units; only used with "
                         "--loss huber. The old default of 1.0 left 87% of "
                         "residuals in the quadratic region")
    ap.add_argument("--w-sparse", type=float, default=0.0,
                    help="L1 on off-diagonal coupling, so the learned topology "
                         "is readable rather than dense")
    ap.add_argument("--w-dev", type=float, default=0.0,
                    help="weight on the regional-deviation term, standardised by "
                         "its own train scale. 0 reproduces the plain objective, "
                         "which a model can win on the common NEM mode alone")
    ap.add_argument("--w-aux", type=float, default=0.0,
                    help="weight on the auxiliary driver targets (demand, wind, "
                         "ramp, scarcity). They never enter selection or any "
                         "reported metric; they exist to shape the trunk and to "
                         "put gradient on their own coupling rows")
    ap.add_argument("--w-bias", type=float, default=0.0,
                    help="penalty on the batch-mean residual per region, against "
                         "the constant offset this model is known to carry")
    ap.add_argument("--aux", default=",".join(AUX))
    ap.add_argument("--context", type=int, default=None,
                    help="steps the bank sees; the head still reads --seq-len of "
                         "them. At window 96 nothing above 20.1 h exists in the "
                         "bank's output, so the 168 h weekly cycle is absent "
                         "from the input. Needs --skip-head >= context - seq_len.")
    ap.add_argument("--head", default="lstm", choices=["lstm", "linear"],
                    help="what reads the bands; the arm name can override it")
    ap.add_argument("--coupling-init", type=float, default=0.0,
                    help="sd of the random seeding of the off-diagonal coupling. "
                         "0 keeps the exact ablation and a dead exogenous pathway "
                         "at step 0; >0 gives the pathway a live start")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; autodetected when omitted")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers; defaults to 4 on cuda, 0 on cpu")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--skip-head", type=int, default=0,
                    help="drop the first N rows of each span before windowing. "
                         "The cached-VMD arms lose W-1=95 rows per year to causal "
                         "history, so --skip-head 95 makes the scored window set "
                         "identical to theirs and the SA1 number comparable with "
                         "FINDINGS. Default 0: this model decomposes in-model and "
                         "needs no history, so by default it is not handicapped.")
    ap.add_argument("--out", default="results/joint.json")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--save-preds", default="preds/joint",
                    help="where to write test predictions, in $/MWh")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    # CUDA when it is there, CPU otherwise. Not MPS: the bank needs rfft and a
    # complex multiply, and CLAUDE.md records that MPS has neither.
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"cuda: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    L, h = args.seq_len, args.horizon

    df = pd.read_csv(args.panel, parse_dates=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
    names = REGIONS
    tgt_idx = [chans.index(c) for c in names]
    aux_names = [c for c in args.aux.split(",") if c] if args.w_aux else []
    missing = [c for c in aux_names if c not in chans]
    if missing:
        raise SystemExit(f"auxiliary channels not in the panel: {missing}")
    aux_idx = [chans.index(c) for c in aux_names]

    raw, starts, n_rows = {}, {}, {}
    for tag, spec in (("train", args.train_year), ("test", args.test_year)):
        a = df[_mask(df, spec)][chans].to_numpy(np.float64)
        T = len(a)
        raw[tag] = a
        starts[tag] = np.arange(args.skip_head, T - L - h + 1)
        n_rows[tag] = T
    assert np.isfinite(raw["train"]).all() and np.isfinite(raw["test"]).all(), \
        "the panel has non-finite values; the joint targets cannot be built"

    print(f"device={device} threads={args.threads} | {len(chans)} channels")
    print(f"targets ({len(names)}): {', '.join(names)}")
    print(f"train {args.train_year}: {n_rows['train']} rows -> {len(starts['train'])} "
          f"windows | test {args.test_year}: {n_rows['test']} rows -> "
          f"{len(starts['test'])} windows"
          + (f"  [--skip-head {args.skip_head}: window set matched to the "
             f"cached-VMD arms]" if args.skip_head else ""))

    s_min = args.skip_head
    n_val = int(len(starts["train"]) * args.val_frac)
    va_starts = starts["train"][-n_val:]
    tr_starts = starts["train"][:-(n_val + L)]
    if args.max_windows:
        n = args.max_windows
        tr_starts, va_starts = tr_starts[:n], va_starts[:max(n // 4, 1)]
        starts["test"] = starts["test"][:max(n // 4, 1)]
        if not args.out.endswith(".check.json"):
            args.out = args.out.replace(".json", "") + ".check.json"
        print(f"[--max-windows {n}] pipeline check only, numbers are meaningless; "
              f"results -> {args.out}")

    mu = raw["train"][s_min:].mean(0)
    sd = raw["train"][s_min:].std(0) + 1e-8
    f_tr, f_te = (raw["train"] - mu) / sd, (raw["test"] - mu) / sd

    sc = TargetScaler(raw["train"][:, tgt_idx], s_min)
    y_tr, y_te = sc.forward(raw["train"][:, tgt_idx]), sc.forward(raw["test"][:, tgt_idx])
    # One window list for every arm in the run. The dataset is built at the
    # largest context any arm asks for, and a shorter-context arm slices the tail
    # of the same window, so every arm is scored on identical rows.
    ctx_max = max([args.context or L] +
                  [int(b[3:]) for a_ in args.arms.split(",")
                   for b in a_.split("/")[-1].split("+") if b.startswith("ctx")])
    if ctx_max > L and args.skip_head < ctx_max - L:
        args.skip_head = ctx_max - L
        print(f"--skip-head raised to {args.skip_head} so a {ctx_max}-step "
              f"context fits; every arm shares this window list")
        s_min = args.skip_head
        for tag_, spec_ in (("train", args.train_year), ("test", args.test_year)):
            starts[tag_] = np.arange(args.skip_head, n_rows[tag_] - L - h + 1)
        n_val = int(len(starts["train"]) * args.val_frac)
        va_starts = starts["train"][-n_val:]
        tr_starts = starts["train"][:-(n_val + L)]

    one, scaler_one = {}, {}
    for j, nm in enumerate(names):
        sc_j = TargetScaler(raw["train"][:, [tgt_idx[j]]], s_min)
        yj_tr = sc_j.forward(raw["train"][:, [tgt_idx[j]]])
        yj_te = sc_j.forward(raw["test"][:, [tgt_idx[j]]])
        scaler_one[nm] = sc_j
        one[nm] = {"train": JointDataset(f_tr, yj_tr, tr_starts, L, h, ctx_max),
                   "val":   JointDataset(f_tr, yj_tr, va_starts, L, h, ctx_max),
                   "test":  JointDataset(f_te, yj_te, starts["test"], L, h, ctx_max)}

    # The deviation term is taken in the standardised space the loss lives in, so
    # its scale is measured there too. Without this the weight would silently
    # depend on how wide the regions happen to be.
    dtr = y_tr[s_min:] - y_tr[s_min:].mean(axis=1, keepdims=True)
    dev_sd = dtr.std(axis=0) + 1e-8
    print(f"regional deviation, standardised space: sd per region "
          f"{np.round(dev_sd, 3).tolist()}")

    # Auxiliary targets ride in the same tensor, after the prices. They are
    # standardised but not asinh'd: demand and wind are not spike-dominated, and
    # a transform chosen for price would be a claim about them nobody has checked.
    sca = ya_tr = ya_te = None
    if aux_idx:
        all_idx = tgt_idx + aux_idx
        sca = TargetScaler(raw["train"][:, all_idx], s_min, transform="asinh",
                           plain_from=len(tgt_idx))
        ya_tr = sca.forward(raw["train"][:, all_idx])
        ya_te = sca.forward(raw["test"][:, all_idx])
        print(f"auxiliary targets ({len(aux_idx)}): {', '.join(aux_names)}")

    data = {
        "train": JointDataset(f_tr, y_tr, tr_starts, L, h, ctx_max),
        "val":   JointDataset(f_tr, y_tr, va_starts, L, h, ctx_max),
        "test":  JointDataset(f_te, y_te, starts["test"], L, h, ctx_max),
        "ctx_max": ctx_max,
        "n_feat": f_tr.shape[1], "targets": tgt_idx, "names": names,
        "scaler": sc, "one": one, "scaler_one": scaler_one, "chans": chans,
        "aux_idx": aux_idx, "aux_names": aux_names, "dev_sd": dev_sd,
        "scaler_aux": sca,
    }
    if aux_idx:
        data.update({
            "train_aux": JointDataset(f_tr, ya_tr, tr_starts, L, h),
            "val_aux":   JointDataset(f_tr, ya_tr, va_starts, L, h),
            "test_aux":  JointDataset(f_te, ya_te, starts["test"], L, h),
        })
    print(f"train {len(tr_starts)} / val {len(va_starts)} / test {len(starts['test'])} windows\n")

    results = []
    if os.path.exists(args.out) and not args.fresh:
        try:
            results = json.load(open(args.out))
            print(f"resuming from {args.out}: {len(results)} run(s) done")
        except Exception as e:
            print(f"could not read {args.out} ({e}); starting fresh")
    done = {(r["arm"], r["seed"]) for r in results}

    arms = args.arms.split(",")
    for seed in [int(v) for v in args.seeds.split(",")]:
        for arm in arms:
            if (arm, seed) in done:
                print(f"  {arm} seed {seed}: cached"); continue
            r = run_arm(arm, seed, data, args, device)
            results.append(r); done.add((arm, seed))
            print(f"  -> {arm} s{seed}: mean MAE {r['test_mae_mean']:.3f}, "
                  f"SA1 {r['test_mae_sa1']:.3f}\n", flush=True)
            dump_atomic(results, args.out)

    print("=" * 76)
    print(f"{'arm':<18}{'targets':>8}{'mean MAE':>11}{'SA1 MAE':>10}   per-region")
    print("-" * 76)
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        if not rs: continue
        mm = np.mean([r["test_mae_mean"] for r in rs])
        s1 = np.mean([r["test_mae_sa1"] for r in rs])
        per = " ".join(f"{k.replace('_price','')} {np.mean([r['test_mae'][k] for r in rs]):.2f}"
                       for k in rs[0]["test_mae"])
        print(f"{arm:<18}{rs[0]['n_targets']:>8}{mm:>11.3f}{s1:>10.3f}   {per}")
    cj = [r for r in results if r["arm"] == "joint" and "coupling" in r]
    if cj:
        print("\nlearned coupling, where each region's exogenous mass sits")
        print(f"  {'region':<12}{'top1':>7}{'top3':>7}   largest sources")
        for nm, d_ in cj[-1]["coupling"].items():
            top = ", ".join(f"{c} {v:.0%}" for c, v in d_["top"][:3])
            print(f"  {nm.replace('_price',''):<12}{d_['top1_share']:>7.0%}"
                  f"{d_['top3_share']:>7.0%}   {top}")


if __name__ == "__main__":
    main()
