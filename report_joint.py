#!/usr/bin/env python3
"""The headline in the first line: did joint prediction help SA1, or hurt it.

Everything else in this file is the explanation of that one sentence. Written so
the answer does not have to be reconstructed from a training log.
"""
import argparse, json, os, re, sys

import numpy as np


def load(path):
    return json.load(open(path)) if os.path.exists(path) else []


def best(rs, arm):
    hits = [r for r in rs if r["arm"] == arm]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/joint.json")
    ap.add_argument("--log", default="logs/joint.log")
    a = ap.parse_args()
    rs = load(a.results)

    stl = best(rs, "single:SA1_price")
    jt = best(rs, "joint")
    nc = best(rs, "joint_nocouple")

    print("=" * 74)
    if stl and jt:
        s, j = stl["test_mae_sa1"], jt["test_mae_sa1"]
        d = (j - s) / s * 100
        verdict = "BETTER" if j < s else ("WORSE" if j > s else "UNCHANGED")
        print(f"HEADLINE: on SA1, joint prediction is {verdict} "
              f"({s:.3f} -> {j:.3f} MAE, {d:+.1f}%)")
        if nc:
            n_ = nc["test_mae_sa1"]
            print(f"          of which multi-task {(n_ - s) / s * 100:+.1f}% "
                  f"and spatial coupling {(j - n_) / max(n_, 1e-9) * 100:+.1f}%")
    else:
        have = sorted({r["arm"] for r in rs})
        need = [x for x in ("single:SA1_price", "joint") if not best(rs, x)]
        print("HEADLINE: not available yet -- still waiting on " + ", ".join(need))
        print(f"          finished so far: {', '.join(have) if have else 'nothing'}")
        if os.path.exists(a.log):
            eps = re.findall(r"\[(\S+) s(\d+) (\d+)/(\d+)\].*?test (\d+\.\d+) "
                             r"\(SA1 (\d+\.\d+)\)", open(a.log).read())
            if eps:
                arm, sd, ep, tot, mm, s1 = eps[-1]
                print(f"          in progress: {arm} epoch {ep}/{tot}, "
                      f"test mean {mm} / SA1 {s1} (not final: selection is on "
                      f"the validation tail, not this number)")
    print("=" * 74)

    if not rs:
        return
    print(f"\n{'arm':<22}{'targets':>8}{'mean MAE':>11}{'SA1 MAE':>10}{'epoch':>7}   per-region")
    print("-" * 94)
    for r in rs:
        per = " ".join(f"{k.replace('_price','')} {v:.2f}" for k, v in r["test_mae"].items())
        print(f"{r['arm']:<22}{r['n_targets']:>8}{r['test_mae_mean']:>11.3f}"
              f"{r['test_mae_sa1']:>10.3f}{r['epoch']:>7}   {per}")

    # Per-region single-task baselines, and the sign of the transfer per region.
    if jt:
        rows = []
        for nm, mtl in jt["test_mae"].items():
            b = best(rs, f"single:{nm}")
            if b:
                rows.append((nm, b["test_mae_sa1"], mtl))
        if rows:
            print(f"\n{'region':<10}{'single-task':>12}{'joint':>10}{'change':>9}")
            for nm, s_, m_ in rows:
                print(f"{nm.replace('_price',''):<10}{s_:>12.3f}{m_:>10.3f}"
                      f"{(m_ - s_) / s_ * 100:>8.1f}%")
            rel = [(m_ - s_) / s_ for _, s_, m_ in rows]
            neg = [nm for nm, s_, m_ in rows if m_ > s_]
            print(f"\ndelta_m = {-np.mean(rel) * 100:+.2f}%  "
                  f"(positive = joint better, averaged over the "
                  f"{len(rows)} region(s) that have a baseline)")
            print(f"negative transfer on {len(neg)} of {len(rows)}"
                  + (f": {', '.join(n.replace('_price','') for n in neg)}" if neg else ""))
        missing = [nm for nm in jt["test_mae"] if not best(rs, f"single:{nm}")]
        if missing:
            print("no single-task baseline yet for: "
                  + ", ".join(n.replace('_price', '') for n in missing))

    # Did the coupling spread, or collapse onto one channel as before?
    if jt and "coupling" in jt:
        print(f"\nlearned coupling -- the question is whether the mass spread or "
              f"collapsed\n  {'region':<10}{'top1':>7}{'top3':>7}   largest sources")
        for nm, d_ in jt["coupling"].items():
            top = ", ".join(f"{c} {v:.0%}" for c, v in d_["top"][:3])
            print(f"  {nm.replace('_price',''):<10}{d_['top1_share']:>7.0%}"
                  f"{d_['top3_share']:>7.0%}   {top}")
        t1 = np.mean([d_["top1_share"] for d_ in jt["coupling"].values()])
        print(f"\n  mean top1 share {t1:.1%}. Uniform over 36 other channels would "
              f"be {1/36:.1%};\n  the single-target model put 73-83% on ramp_VIC1, "
              f"which is the number to beat.")

    if jt and jt.get("history"):
        h = jt["history"][-1].get("loss_terms", {})
        if h:
            print("\nloss terms, last epoch: "
                  + "  ".join(f"{k} {v:.3f}" for k, v in h.items()))
    print("\nMAE here is val-selected and single-seed. For rMAE, a Diebold-Mariano "
          "test\nand negative transfer with significance, run: "
          "python3 -m analysis.eval_joint")


if __name__ == "__main__":
    main()
