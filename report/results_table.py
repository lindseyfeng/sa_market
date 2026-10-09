#!/usr/bin/env python3
"""Emit the section 4 results table as markdown, on one row set.

Every arm is intersected with `panel_lstm`'s scored rows before anything is
averaged, so each number in the table is computed on the same timestamps, and the regime
masks come from the reference's truth rounded to cents so every row is split
the same way. Both matter. The earlier version of this table mixed two
alignments, which put the same arm at two different negative-price MAEs
depending on which table you read; and the stored float32 truth returns a price
of exactly $0.00 as -7.6e-06 in the neural files but as 0.0 in the baseline
files, which moved 79 rows across the negative/calm boundary depending on who
wrote the file.

    python report/results_table.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF = "preds/pace_h6/panel_lstm_s1.npz"
ROWS = [
    ("Naive", "`naive_persist` the last observed price", "preds/baselines_h6/naive_persist_s0.npz"),
    ("", "`naive_week` the same half-hour one week earlier", "preds/baselines_h6/naive_week_s0.npz"),
    ("Linear", "`ar_window` ridge, target window only", "preds/baselines_h6/ar_window_s0.npz"),
    ("", "`var` VAR, five price windows", "preds/baselines_h6/var_s0.npz"),
    ("", "`global_linear` one pooled linear model across regions", "preds/baselines_h6/global_linear_s0.npz"),
    ("", "`arx_window` ridge, raw 96 x 37 window", "preds/baselines_h6/arx_window_s0.npz"),
    ("", "`arx_ctx336` the same ridge on a 336-step window", "preds/baselines_h6/arx_ctx336_s0.npz"),
    ("Trees", "`gbt_own` boosted trees, target window", "preds/baselines_h6/gbt_own_s0.npz"),
    ("", "`gbt_prices` boosted trees, five price windows", "preds/baselines_h6/gbt_prices_s0.npz"),
    ("", "`gbt_pca` boosted trees, prices + 8 exogenous PCs", "preds/baselines_h6/gbt_pca_s0.npz"),
    ("", "`gbt_all` boosted trees, all 3,552 values", "preds/baselines_h6/gbt_all_s0.npz"),
    ("LSTM, no decomposition", "`single_SA1_price` LSTM, single task", "preds/pace_h6/single_SA1_price_s1.npz"),
    ("", "`panel_lstm` LSTM on the raw window -- **the control**", REF),
    ("", "`red37` LSTM on a learned 37-channel projection", "preds/final/red_panel_lstm_red37_x.npz"),
    ("", "`red16` LSTM on a learned 16-channel projection", "preds/final/red_panel_lstm_red16_x.npz"),
    ("", "`red8` LSTM on a learned 8-channel projection", "preds/final/red_panel_lstm_red8_x.npz"),
    ("LSTM, bands", "`joint_nocouple` bands, coupling frozen at identity", "preds/pace_h6/joint_nocouple_s1.npz"),
    ("", "`joint` bands + per-band coupling", "preds/pace_h6/joint_s1.npz"),
    ("", "`joint_lstm_film` the same + FiLM gate", "preds/pace_h6/joint_lstm_film_s1.npz"),
    ("", "`joint_lstm_film_ctx336` the same on a 336-step bank view", "preds/pace_h6/joint_lstm_film_ctx336_s1.npz"),
    ("LSTM, bands + Jacobian weight", "`a0.25` bands + coupling, Jacobian weight 0.25", "preds/jac/a0.25_s1.npz"),
    ("", "`a0.5` the same, 0.5", "preds/jac/a0.5_s1.npz"),
    ("", "`a0.75` the same, 0.75", "preds/jac/a0.75_s1.npz"),
    ("", "`a1.0` the same, 1.0", "preds/jac/a1.0_s1.npz"),
    ("", "`f0.25` FiLM + Jacobian 0.25", "preds/fj/f0.25_s1.npz"),
    ("", "`f0.5` FiLM + Jacobian 0.5", "preds/fj/f0.5_s1.npz"),
    ("", "`f0.75` FiLM + Jacobian 0.75", "preds/fj/f0.75_s1.npz"),
]
SEGS = [("negative", lambda y: y < 0), ("calm", lambda y: (y >= 0) & (y < 100)),
        ("high", lambda y: (y >= 100) & (y < 300)), ("spike", lambda y: y >= 300)]


def ld(f):
    z = np.load(f, allow_pickle=True)
    nm = [str(v) for v in z["names"]]
    i = nm.index("SA1_price") if "SA1_price" in nm else 0
    return z["pred"][:, i], z["truth"][:, i], z["start"]


def main():
    pr, yr, sr = ld(REF)
    out, counts = [], None
    for grp, label, f in ROWS:
        if not os.path.exists(f):
            print(f"MISSING {f}", file=sys.stderr)
            continue
        p, y, st = ld(f)
        common = np.intersect1d(st, sr)
        a, b = np.isin(st, common), np.isin(sr, common)
        yy, pp = y[a], p[a]
        ymask = np.round(yr[b], 2)          # one canonical split for every row
        if counts is None:
            counts = {n: int(m(ymask).sum()) for n, m in SEGS}
        cells = [f"{np.abs(pp - yy)[m(ymask)].mean():.1f}" for _, m in SEGS]
        out.append(f"| {grp} | {label} | {np.abs(pp - yy).mean():.2f} | "
                   f"{np.sqrt(((pp - yy) ** 2).mean()):.1f} | " + " | ".join(cells) + " |")
    hdr = ("| | model | MAE | RMSE | " + " | ".join(f"{n} ({counts[n]:,})" for n, _ in SEGS) + " |")
    print(hdr)
    print("|---|---|---:|---:|" + "---:|" * len(SEGS))
    print("\n".join(out))


if __name__ == "__main__":
    main()
