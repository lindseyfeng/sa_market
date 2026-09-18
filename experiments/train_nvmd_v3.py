#!/usr/bin/env python3
"""
Train NVMD v3 (structured filter bank) on next-step forecast MSE.

The forecast term decides what the bank is for; the three structural terms
keep it interpretable:

    bandwidth  -- VMD's own criterion, prefer narrowband modes
    separation -- adjacent centres at least ~1 bandwidth apart
    overlap    -- pairwise spectral overlap between distinct bands

Unlike v2 these do not have to fight for ordering, DC coverage, or exact
reconstruction -- those hold by construction, so the weights can stay small.

    python train_nvmd_v3.py \
        --train-csv ../sa_market2/VMD_modes_with_residual_2018_2018.csv \
        --val-csv   ../sa_market2/VMD_modes_with_residual_2019_2019.csv \
        --K 8 --seq-len 96 --epochs 30
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.nvmd_v3 import NVMDv3Forecaster
from experiments.train_nvmd_v2 import RRPWindowDataset, load_rrp, set_seed, eval_epoch


def train_epoch(model, loader, optimizer, device, w, clip_grad=5.0):
    model.train()
    sums = dict(loss=0.0, fc=0.0, bw=0.0, sep=0.0, ovl=0.0)
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, _, _, y_pred = model(x)
        dec = model.decomposer

        fc = F.mse_loss(y_pred, y)
        bw = dec.bandwidth_loss(x)
        sep = dec.separation_loss(x)
        ovl = dec.overlap_loss(x)
        loss = fc + w["bw"] * bw + w["sep"] * sep + w["ovl"] * ovl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        bs = x.size(0)
        n += bs
        for k, v in zip(sums, (loss, fc, bw, sep, ovl)):
            sums[k] += v.item() * bs

    return {k: v / max(n, 1) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="../sa_market2/VMD_modes_with_residual_2018_2018.csv")
    ap.add_argument("--val-csv", default="../sa_market2/VMD_modes_with_residual_2019_2019.csv")
    ap.add_argument("--rrp-col", default="RRP")
    ap.add_argument("--seq-len", type=int, default=96)

    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--adapt", type=float, default=0.5,
                    help="max gap-logit perturbation; 0 = fixed filter bank")
    ap.add_argument("--ratio", type=float, default=1.8,
                    help="geometric band spacing ratio")
    ap.add_argument("--head", choices=["lstm", "linear"], default="lstm",
                    help="linear = predictor-agnostic; forces the modes to "
                         "carry the signal instead of the head")
    ap.add_argument("--tail", type=int, default=48,
                    help="steps of the window fed to a linear head")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--band-lr", type=float, default=3e-2,
                    help="LR for gap_logits/log_bw; they need ~100x the base "
                         "rate to traverse a band position within the run")
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--w-bw", type=float, default=0.05)
    ap.add_argument("--w-sep", type=float, default=1.0)
    ap.add_argument("--w-ovl", type=float, default=0.05)

    ap.add_argument("--outdir", default="./runs_nvmd_v3")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rrp_tr = load_rrp(args.train_csv, args.rrp_col)
    rrp_va = load_rrp(args.val_csv, args.rrp_col)
    tr_ds = RRPWindowDataset(rrp_tr, args.seq_len)
    va_ds = RRPWindowDataset(rrp_va, args.seq_len, mean=tr_ds.mean, std=tr_ds.std)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)
    print(f"Train: {len(tr_ds)} windows | Val: {len(va_ds)} windows")

    model = NVMDv3Forecaster(
        K=args.K, signal_len=args.seq_len, d_model=args.d_model,
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        adapt=args.adapt, head=args.head, tail=args.tail,
    ).to(device)
    model.decomposer.gap_logits.data = (
        torch.arange(args.K, dtype=torch.float32, device=device)
        * math.log(args.ratio)
    )
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")

    # The band parameters need their own, much larger LR.  Measured gradient
    # sign consistency on gap_logits is 1.000 (every batch pushes the same way)
    # and their gradient is ~8x the LSTM's -- so movement is limited purely by
    # step budget: at lr=3e-4 over ~2000 steps the bank can traverse less than
    # one band position (log(ratio)=0.59).  Weight decay is also excluded here,
    # since decaying gap_logits toward 0 flattens the geometric spacing back
    # toward uniform and fights the prior.
    band_names = {"decomposer.gap_logits", "decomposer.log_bw"}
    band_params = [p for n, p in model.named_parameters() if n in band_names]
    rest = [p for n, p in model.named_parameters() if n not in band_names]
    opt = torch.optim.AdamW(
        [
            {"params": rest, "lr": args.lr, "weight_decay": 1e-5},
            {"params": band_params, "lr": args.band_lr, "weight_decay": 0.0},
        ]
    )
    print(f"Band-parameter LR: {args.band_lr:g} ({len(band_params)} tensors), "
          f"base LR: {args.lr:g}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    w = {"bw": args.w_bw, "sep": args.w_sep, "ovl": args.w_ovl}

    best, best_state, stale = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, tr_dl, opt, device, w, args.clip_grad)
        va = eval_epoch(model, va_dl, device, tr_ds.mean, tr_ds.std)
        sched.step()

        print(f"[{ep:03d}/{args.epochs}] loss={tr['loss']:.5f} "
              f"(fc={tr['fc']:.5f} bw={tr['bw']:.4f} sep={tr['sep']:.2e} ovl={tr['ovl']:.4f}) | "
              f"val MAE={va['mae_raw']:.2f} RMSE={va['rmse_raw']:.2f} | {time.time()-t0:.0f}s")

        if ep % 5 == 0 or ep == args.epochs:
            print("      bands (centre / bw / period h): " + "  ".join(
                f"{c:.4f}/{b:.4f}/{p:.1f}" for _, c, b, p in model.decomposer.describe()))

        if va["mae_raw"] < best:
            best = va["mae_raw"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            print(f"  -> new best val MAE = {best:.2f}")
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model_state": best_state, "best_val_mae": best,
                    "args": vars(args), "arch": "v3",
                    "norm": {"mean": tr_ds.mean, "std": tr_ds.std}},
                   os.path.join(args.outdir, "best.pt"))
        print(f"\nSaved best: val MAE = {best:.2f} -> {args.outdir}/best.pt")
        print("\nFinal band table:")
        for k, c, b, p in model.decomposer.describe():
            print(f"  mode {k}: centre={c:.4f}  bw={b:.4f}  period={p:8.2f} h")


if __name__ == "__main__":
    main()
