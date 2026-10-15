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


REGIONS = ["SA1_price", "NSW1_price", "VIC1_price", "QLD1_price", "TAS1_price"]


def ld(f, region=None):
    """All five regions by default, not SA1.

    The model has five outputs and the headline number has to be the mean over
    all of them. Scoring SA1 alone nearly doubled the apparent effect of +env:
    -3.2% against the control on SA1, -1.82% over the five regions. The reason
    is not noise -- the gain tracks whether a region has a weather channel of
    its own. SA1 -3.2% and VIC1 -2.9% have Adelaide, the two SA wind zones and
    Melbourne; NSW1 -0.6%, QLD1 -1.3% and TAS1 -0.9% have none.
    """
    z = np.load(f, allow_pickle=True)
    nm = [str(v) for v in z["names"]]
    want = [region] if region else REGIONS
    idx = [nm.index(r) for r in want if r in nm] or [0]
    return z["pred"][:, idx], z["truth"][:, idx], z["start"]


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
            k = min(p.shape[1], pr.shape[1])
            p, y, pr, yr = p[:, :k], y[:, :k], pr[:, :k], yr[:, :k]
            e, ec = np.abs(p - y), np.abs(pr - yr)
            _, pv, _ = dm(ec.reshape(-1), e.reshape(-1))
            r["mae"] = float(e.mean())
            r["rmse"] = float(np.sqrt(((p - y) ** 2).mean()))
            r[f"d_{rname}"] = 100 * (e.mean() - ec.mean()) / ec.mean()
            r[f"p_{rname}"] = pv
            if rname == "joint":
                # Regimes are cut on the first region's price, which is the
                # one the regime analysis was built on; the MAE averaged is
                # still over every region.
                for n, m in SEGS:
                    r[n] = float(e[m(ym[:, 0])].mean())
                for j, nmr in enumerate(REGIONS[:k]):
                    r["R_" + nmr.replace("_price", "")] = float(e[:, j].mean())
        sub = tag.split("  [")[1][:-1] if "  [" in tag else None
        r.update(gate_stats(a.models, tag.split("  [")[0], sub))
        rows.append(r)
    rows.sort(key=lambda r: r["mae"])

    w = max(len(r["arm"]) for r in rows) + 2
    regs = [k[2:] for k in rows[0] if k.startswith("R_")]
    print(f"{'arm':<{w}}{'OVERALL':>9}{'RMSE':>8}"
          f"{'vs panel':>10}{'p':>7}{'vs joint':>10}{'p':>7}"
          + "".join(f"{n:>10}" for n, _ in SEGS)
          + "".join(f"{g:>8}" for g in regs)
          + f"{'gate_c':>9}{'bgate':>8}{'bandmix':>8}")
    print("-" * (w + 8 * 2 + 10 * 2 + 7 * 2 + 10 * len(SEGS) + 25))
    for r in rows:
        print(f"{r['arm']:<{w}}{r['mae']:9.2f}{r['rmse']:8.1f}"
              f"{r['d_panel_lstm']:+10.1f}{r['p_panel_lstm']:7.3f}"
              f"{r['d_joint']:+10.1f}{r['p_joint']:7.3f}"
              + "".join(f"{r[n]:10.1f}" for n, _ in SEGS)
              + "".join(f"{r.get('R_' + g, float('nan')):8.2f}" for g in regs)
              + f"{r.get('gate_c', float('nan')):9.4f}"
              + f"{r.get('bgate', float('nan')):8.4f}"
              + f"{r.get('band_mix', float('nan')):8.4f}")
    print("\nOVERALL is the mean over all five regions, which is what the model "
          "predicts.\nScoring SA1 alone reports roughly twice the effect.\n")
    print("vs panel = the whole band architecture; vs joint = the nonlinearity "
          "alone.\ngate_c ~ 0 means the bank trained back to linear and its MAE "
          "is not evidence\nof anything nonlinear. One-sided DM, Andrews AR(1) "
          "bandwidth, no multiplicity\nadjustment. One seed: seed noise here is "
          "0.116 MAE.")


if __name__ == "__main__":
    main()
