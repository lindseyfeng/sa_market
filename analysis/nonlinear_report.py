#!/usr/bin/env python3
"""Score the nonlinear-bank sweep, against both references that matter.

Two different questions, two different baselines, and reporting only one of
them is how a result gets oversold:

  vs panel_lstm   what the whole band architecture is worth. This is the
                  number the README headlines.
  vs joint/lstm   what *breaking the linearity* is worth, holding the band
                  architecture fixed. +gate/+env/+bgate differ from joint by
                  the nonlinearity and nothing else, so this is the only
                  comparison that isolates it.

Alongside each MAE, the gate statistics read out of the checkpoint. Every
switch in this sweep is an exact no-op at initialisation, so an arm can train
straight back to the linear bank and still report a respectable MAE -- which
is precisely the reading joint_lstm_film's 38.45 against joint's 38.50 never
got. `gate_c` at zero means the bank is still exactly linear and the MAE says
nothing about nonlinearity at all.

    python -m analysis.nonlinear_report --preds preds/nl6 --models runs/nl6
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.eval_joint import dm

SEGS = [("negative", lambda y: y < 0), ("calm", lambda y: (y >= 0) & (y < 100)),
        ("high", lambda y: (y >= 100) & (y < 300)), ("spike", lambda y: y >= 300)]


def ld(f):
    z = np.load(f, allow_pickle=True)
    nm = [str(v) for v in z["names"]]
    i = nm.index("SA1_price") if "SA1_price" in nm else 0
    return z["pred"][:, i], z["truth"][:, i], z["start"]


def paired(f, ref):
    """Intersect an arm's scored rows with the reference's before averaging."""
    pr, yr, sr = ref
    p, y, st = ld(f)
    common = np.intersect1d(st, sr)
    a, b = np.isin(st, common), np.isin(sr, common)
    # Regimes are cut on the reference truth rounded to cents: the stored
    # float32 returns a price of exactly $0.00 as -7.6e-06 in some files and
    # 0.0 in others, which moves 79 rows across the negative/calm boundary
    # depending only on who wrote the file.
    return p[a], y[a], pr[b], yr[b], np.round(yr[b], 2)


def gate_stats(models_dir, arm, subdir=None):
    """Checkpoint names come from the ARM, and two jobs can share an arm name.

    `+gate+env+bgate` was run twice, once with --w-jacobian 0.25, so both wrote
    `joint_lstm_gate_env_bgate_s1.pt`. They were saved into per-job directories
    precisely so they would not collide -- and then a recursive glob taking
    hits[0] reported one arm's gate statistics under both names anyway. Scope
    the search to the job's own directory when the caller knows it.
    """
    import torch
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", arm)
    root = os.path.join(models_dir, subdir) if subdir else models_dir
    hits = sorted(glob.glob(os.path.join(root, "**", f"{tag}_s*.pt"),
                            recursive=True))
    if not hits:
        return {}
    st = torch.load(hits[0], map_location="cpu", weights_only=False)["state"]
    out = {}
    for key, name in [("decomposer.decomposer.g_c", "gate_c"),
                      ("decomposer.aggregator.film.2.weight", "bgate"),
                      ("decomposer.aggregator.mix.weight", "band_mix")]:
        if key in st:
            t = st[key]
            if name == "band_mix":
                ident = t.new_zeros(t.shape)
                ident[0, 0, 1] = 1.0
                out[name] = float((t - ident).abs().sum())
            else:
                out[name] = float(t.abs().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds/nl6")
    ap.add_argument("--models", default="runs/nl6")
    ap.add_argument("--control", default="preds/pace_h6/panel_lstm_s1.npz")
    ap.add_argument("--linear", default="preds/pace_h6/joint_s1.npz",
                    help="the linear-bank arm the nonlinearity is measured against")
    a = ap.parse_args()

    files = {}
    for f in sorted(glob.glob(os.path.join(a.preds, "**", "*.npz"), recursive=True)):
        files[re.sub(r"_s\d+\.npz$", "", os.path.basename(f))
              + ("" if os.path.dirname(f) == a.preds
                 else f"  [{os.path.basename(os.path.dirname(f))}]")] = f
    if not files:
        raise SystemExit(f"no .npz under {a.preds}")

    refs = {"panel_lstm": ld(a.control), "joint": ld(a.linear)}
    rows = []
    for tag, f in files.items():
        r = {"arm": tag}
        for rname, ref in refs.items():
            p, y, pr, yr, ym = paired(f, ref)
            e, ec = np.abs(p - y), np.abs(pr - yr)
            _, pv, _ = dm(ec, e)
            r[f"mae"] = float(e.mean())
            r[f"rmse"] = float(np.sqrt(((p - y) ** 2).mean()))
            r[f"d_{rname}"] = 100 * (e.mean() - ec.mean()) / ec.mean()
            r[f"p_{rname}"] = pv
            if rname == "joint":
                for n, m in SEGS:
                    r[n] = float(e[m(ym)].mean())
        sub = tag.split("  [")[1][:-1] if "  [" in tag else None
        r.update(gate_stats(a.models, tag.split("  [")[0], sub))
        rows.append(r)
    rows.sort(key=lambda r: r["mae"])

    w = max(len(r["arm"]) for r in rows) + 2
    print(f"{'arm':<{w}}{'MAE':>8}{'RMSE':>8}"
          f"{'vs panel':>10}{'p':>7}{'vs joint':>10}{'p':>7}"
          + "".join(f"{n:>10}" for n, _ in SEGS)
          + f"{'gate_c':>9}{'bgate':>8}{'bandmix':>8}")
    print("-" * (w + 8 * 2 + 10 * 2 + 7 * 2 + 10 * len(SEGS) + 25))
    for r in rows:
        print(f"{r['arm']:<{w}}{r['mae']:8.2f}{r['rmse']:8.1f}"
              f"{r['d_panel_lstm']:+10.1f}{r['p_panel_lstm']:7.3f}"
              f"{r['d_joint']:+10.1f}{r['p_joint']:7.3f}"
              + "".join(f"{r[n]:10.1f}" for n, _ in SEGS)
              + f"{r.get('gate_c', float('nan')):9.4f}"
              + f"{r.get('bgate', float('nan')):8.4f}"
              + f"{r.get('band_mix', float('nan')):8.4f}")
    print("\nvs panel = the whole band architecture; vs joint = the nonlinearity "
          "alone.\ngate_c ~ 0 means the bank trained back to linear and its MAE "
          "is not evidence\nof anything nonlinear. One-sided DM, Andrews AR(1) "
          "bandwidth, no multiplicity\nadjustment. One seed: seed noise here is "
          "0.116 MAE.")


if __name__ == "__main__":
    main()
