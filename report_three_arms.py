#!/usr/bin/env python3
"""Render three_arms_results.json as a markdown log.

Called after every completed run, so THREE_ARMS.md is always current even
while the job is still going.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np

ARM_DESC = {
    "vmd_price":     ("temporal", "univariate VMD on the price", 8),
    "vmd_panel":     ("spatial",  "univariate VMD per channel, concatenated", 215),
    "nvmd_temporal": ("temporal", "joint, coupling frozen at $A_k=I$", 33),
    "nvmd_st":       ("spatial",  "joint, per-band coupling, concat", 33),
}
ORDER = ["vmd_price", "vmd_panel", "nvmd_temporal", "nvmd_st"]


def agg(rs, key):
    v = np.array([r[key] for r in rs])
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return v.mean(), sd, len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="three_arms_results.json")
    ap.add_argument("--out", default="THREE_ARMS.md")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    res = json.load(open(args.results)) if os.path.exists(args.results) else []
    by = {a: [r for r in res if r["arm"] == a] for a in ORDER}
    n_done, n_total = len(res), len(ORDER) * args.seeds

    L = []
    L.append("# Three-arm comparison: does spatio-temporal NVMD beat VMD?\n")
    L.append(f"*generated {datetime.now():%Y-%m-%d %H:%M} -- "
             f"{n_done}/{n_total} runs complete*\n")
    L.append("Settles claim 3 of `RESULTS.md` section 12, which that document "
             "records as **never run**. Every arm sits on identical rows, "
             "windows, head, optimiser and budget, so the numbers below are "
             "the first that may be compared directly.\n")

    L.append("## Result\n")
    L.append("Test MAE on 2019, mean over seeds, lower is better. Two "
             "selection rules, both reported:\n")
    L.append("- **honest** -- weights chosen on the validation tail of the "
             "train year, test scored once from them")
    L.append("- **cherry** -- the minimum over epochs of test MAE, which is "
             "the statistic `RESULTS.md` section 13.3 and "
             "`benchmark_seeds.py` report\n")
    L.append("Neither guarantees generalisation on one test year and one "
             "region. The **gap** between them is the size of the selection "
             "effect, and is a result in its own right.\n")
    L.append("| arm | information | decomposition | $n_{in}$ | test MAE (honest) | test MAE (cherry) | gap | seeds |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    base = None
    for a in ORDER:
        info, dec, nin = ARM_DESC[a]
        if not by[a]:
            L.append(f"| `{a}` | {info} | {dec} | {nin} | *pending* | | | 0 |")
            continue
        m, ms, n = agg(by[a], "test_mae")
        if a == "vmd_price":
            base = m
        has_c = all("test_mae_cherry" in r for r in by[a])
        if has_c:
            c, cs, _ = agg(by[a], "test_mae_cherry")
            gap = f"{m - c:+.3f}"
            cstr = f"{c:.3f} ± {cs:.3f}"
        else:
            gap, cstr = "--", "*not recorded*"
        L.append(f"| `{a}` | {info} | {dec} | {nin} | "
                 f"**{m:.3f}** ± {ms:.3f} | {cstr} | {gap} | {n} |")

    if base is not None:
        L.append("\n### Margin over `vmd_price`, the claim-3 baseline\n")
        L.append("| arm | Δ MAE | relative |")
        L.append("|---|---:|---:|")
        for a in ORDER[1:]:
            if not by[a]:
                continue
            m, _, _ = agg(by[a], "test_mae")
            L.append(f"| `{a}` | {m-base:+.3f} | {(m-base)/base*100:+.2f}% |")
        L.append("\nNegative is better than the price-only VMD baseline.\n")

    L.append("\n## Protocol\n")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| panel | `compound_2018_2022.csv`, 33 channels, target `SA1_price` |")
    L.append("| split | train 2018, test 2019, no overlap |")
    L.append("| VMD | causal, window 96, K=8, alpha=1000 (the *best-by-test* "
             "config of section 10, i.e. the most baseline-favourable choice) |")
    L.append("| decomposed | the 26 non-calendar channels; calendar enters "
             "`vmd_panel` raw |")
    L.append("| window | 96 steps, horizon 1 |")
    L.append("| scored rows | window ends at row >= 190 in **both** years, for "
             "**every** arm, because causal VMD has no mode before row 95 |")
    L.append("| head | LSTM 128x2 bidirectional, dropout 0.1, then 128 -> 1 |")
    L.append("| budget | AdamW, lr 3e-4, cosine to 1e-6, clip 5.0, batch 256, "
             "30 epochs, patience 10 |")
    L.append("| selection | best MAE on the **validation tail of 2018** "
             "(last 15%, with a 96-window embargo); test scored once, from "
             "those weights |")
    L.append("| seeds | 1, 2, 3 |")

    L.append("\n## Why this had to be run\n")
    L.append("`RESULTS.md` section 12 states the problem plainly: every VMD "
             "comparison in that document used *temporal-only* NVMD on the "
             "price-only per-year CSVs, while every spatial result was "
             "NVMD-against-NVMD on the compound panel. Different data, "
             "different pipeline, different MAE scale. \"Spatial beats "
             "temporal NVMD\" and \"temporal NVMD beats VMD\" cannot be "
             "composed into \"spatial NVMD beats VMD\".\n")
    L.append("`vmd_price` vs `nvmd_st` is claim 3. `vmd_panel` is the stronger "
             "test, because it hands classical VMD the same exogenous panel "
             "and so isolates *joint* decomposition from merely *having* the "
             "channels. `nvmd_temporal` is the fourth corner of the 2x2.\n")

    L.append("\n## Relation to section 13.3\n")
    L.append("That table read OFF 14.180, MIX 14.635, CONCAT 14.035, "
             "XFILTER 14.015 over 2 seeds. It reported the **cherry** "
             "statistic and did not align rows with any VMD arm. The cherry "
             "column above is the like-for-like comparison; the honest column "
             "is not.\n")
    L.append("The same selection rule is used by `benchmark_seeds.py`, whose "
             "`va_dl` is built from the *test* CSV, so sections 2 and 8 report "
             "the cherry statistic too. Section 10's ridge screen chose its "
             "lambda on a train-year tail and is unaffected, as are claims 1 "
             "and 2, which do not depend on epoch selection at all.\n")
    L.append("The rule was applied **symmetrically** to every method, so it is "
             "not favouritism. It biases a *comparison* only through variance: "
             "a minimum over ~30 test evaluations rewards whichever arm has "
             "the noisier epoch-to-epoch test curve, and `nvmd_st` carries an "
             "extra 8x33x33 coupling tensor that `nvmd_temporal` does not.\n")

    L.append("\n## Per-run detail\n")
    L.append("| arm | seed | val MAE | test MAE (honest) | ep | "
             "test MAE (cherry) | ep | test RMSE (honest) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ORDER:
        for r in sorted(by[a], key=lambda r: r["seed"]):
            c = r.get("test_mae_cherry")
            cs = f"{c:.3f}" if c is not None else "--"
            ce = r.get("epoch_cherry", "--")
            L.append(f"| `{a}` | {r['seed']} | {r['val_mae']:.3f} | "
                     f"{r['test_mae']:.3f} | {r['epoch']} | {cs} | {ce} | "
                     f"{r['test_rmse']:.3f} |")

    L.append("\n## Caveats to carry forward\n")
    L.append("- Two years, one hub, one target. Nothing here speaks to "
             "other ISOs.")
    L.append("- The weather channels are **reanalysis, not forecast**, so any "
             "spatial margin is an **upper bound**. This caveat is "
             "load-bearing: claim 1 of this project is itself an accusation "
             "of leakage.")
    L.append("- `merge_asof(direction=\"nearest\", +/-60min)` in the panel "
             "build lets a :30 settlement take a :00 reading up to 30 minutes "
             "ahead.")
    L.append("- MVMD (Rehman & Aftab 2019) extends VMD to joint multi-channel "
             "decomposition. `vmd_panel` is per-channel VMD, not MVMD, so "
             "\"VMD cannot use spatial information\" remains **not** a "
             "defensible sentence.")

    open(args.out, "w").write("\n".join(L) + "\n")
    print(f"{args.out}: {n_done}/{n_total} runs")


if __name__ == "__main__":
    main()
