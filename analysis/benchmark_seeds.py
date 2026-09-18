#!/usr/bin/env python3
"""
Multi-seed wrapper around benchmark.py's train_and_eval.

Single-seed margins in this project have been 0.1-1%, which is inside run-to-run
noise -- the MLP and LSTM rankings flipped meaning at that scale.  This reports
mean +/- std over N seeds and a paired comparison against a reference method,
so a claim can be made (or withheld) on evidence.

    python benchmark_seeds.py \
        --method "Causal VMD" causal_train.csv causal_test.csv \
        --method "NVMD v3"    v3_train.csv     v3_test.csv \
        --ref "Causal VMD" --seeds 5 --window 48 --epochs 30
"""

import argparse
import statistics as st

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analysis.benchmark import (
    ModeWindowDataset, detect_mode_cols, set_seed, train_and_eval,
    LinearPredictor, MLPPredictor, LSTMPredictor, CNNBiLSTMPredictor,
)


def build_factories(skip_cnn: bool):
    f = {
        "Linear": lambda K, W: LinearPredictor(K, W),
        "MLP":    lambda K, W: MLPPredictor(K, W),
        "LSTM":   lambda K, W: LSTMPredictor(K),
    }
    if not skip_cnn:
        try:
            from train.baseline.cnn_bilstm import MRC_BiLSTM  # noqa: F401
            f["CNN-BiLSTM"] = lambda K, W: CNNBiLSTMPredictor(K, W)
        except ImportError:
            print("  (CNN-BiLSTM skipped -- could not import MRC_BiLSTM)")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs=3, action="append", required=True,
                    metavar=("NAME", "TRAIN_CSV", "VAL_CSV"))
    ap.add_argument("--ref", default=None,
                    help="method name to compare others against")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=1,
                    help="steps ahead to predict (1 = next 30 min)")
    ap.add_argument("--target-col", default="RRP")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--skip-cnn-bilstm", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these predictors, e.g. --only CNN-BiLSTM. "
                         "Lets a missing cell be filled without re-running the "
                         "predictors that are already tabulated.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = args.window
    factories = build_factories(args.skip_cnn_bilstm)
    if args.only:
        missing = [p for p in args.only if p not in factories]
        if missing:
            raise SystemExit(f"--only: unknown predictor(s) {missing}; "
                             f"available: {list(factories)}")
        factories = {k: v for k, v in factories.items() if k in args.only}

    # raw[method][predictor] = list of val MAE, one per seed
    raw = {name: {p: [] for p in factories} for name, _, _ in args.method}

    for name, tr_csv, va_csv in args.method:
        df_tr, df_va = pd.read_csv(tr_csv), pd.read_csv(va_csv)
        cols = detect_mode_cols(df_tr)
        K = len(cols)

        tr_ds = ModeWindowDataset(df_tr, cols, args.target_col, W,
                                  horizon=args.horizon)
        va_ds = ModeWindowDataset(df_va, cols, args.target_col, W,
                                  mode_means=tr_ds.mode_means, mode_stds=tr_ds.mode_stds,
                                  target_mean=tr_ds.target_mean, target_std=tr_ds.target_std,
                                  horizon=args.horizon)
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

        for pname, factory in factories.items():
            for s in range(args.seeds):
                set_seed(1000 + s)
                model = factory(K, W).to(device)
                _, va_m = train_and_eval(
                    model, tr_dl, va_dl, device,
                    target_mean=tr_ds.target_mean, target_std=tr_ds.target_std,
                    epochs=args.epochs, lr=args.lr, patience=args.patience,
                )
                raw[name][pname].append(va_m["mae"])
                print(f"  [{name}] {pname:12s} seed {s}: MAE={va_m['mae']:.3f}", flush=True)

    print("\n" + "=" * 78)
    print(f"  Val MAE, mean +/- std over {args.seeds} seeds (AUD/MWh)")
    print("=" * 78)
    preds = list(factories)
    print("  {:<20s}".format("Method") + "".join(f"{p:>18s}" for p in preds))
    print("  " + "-" * (20 + 18 * len(preds)))
    for name in raw:
        cells = []
        for p in preds:
            v = raw[name][p]
            sd = st.stdev(v) if len(v) > 1 else 0.0
            cells.append(f"{st.mean(v):8.2f} +/-{sd:5.2f}")
        print(f"  {name:<20s}" + "".join(f"{c:>18s}" for c in cells))

    if args.ref and args.ref in raw:
        print("\n" + "=" * 78)
        print(f"  Paired difference vs '{args.ref}'  (negative = better)")
        print("=" * 78)
        for name in raw:
            if name == args.ref:
                continue
            cells = []
            for p in preds:
                a = np.array(raw[name][p]); b = np.array(raw[args.ref][p])
                d = a - b
                sd = d.std(ddof=1) if len(d) > 1 else 0.0
                # separated only if the mean gap exceeds the paired spread
                mark = "*" if sd > 0 and abs(d.mean()) > 2 * sd / np.sqrt(len(d)) else " "
                cells.append(f"{d.mean():+7.2f}+/-{sd:4.2f}{mark}")
            print(f"  {name:<20s}" + "".join(f"{c:>18s}" for c in cells))
        print("\n  * = |mean difference| exceeds 2 standard errors; anything")
        print("    unmarked is inside noise and must not be reported as a win.")
    print()


if __name__ == "__main__":
    main()
