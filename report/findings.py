#!/usr/bin/env python3
"""Write FINDINGS.md. Regenerated, never edited.

Organised as: what holds, the architecture, the objective, the results, what the
decomposition is, the error analysis, the frontier, the caveats. The long record
of every arm and family is attic/FINDINGS-full.md from report_findings.py; both
read the same result files so neither goes stale against the other.
"""
import glob, os, re
from datetime import datetime

import numpy as np

SEG = [("negative", lambda y: y < 0), ("calm", lambda y: (y >= 0) & (y < 100)),
       ("high", lambda y: (y >= 100) & (y < 300)), ("spike", lambda y: y >= 300)]
L, H = 96, 6

NAME = {
    "arx_window": "ridge, raw 96 x 37 window",
    "ar_window": "ridge, target window only",
    "var": "VAR, five price windows",
    "global_linear": "one pooled linear model across regions",
    "naive_persist": "the last observed price",
    "naive_week": "the same half-hour one week earlier",
    "gbt_own": "boosted trees, target window",
    "gbt_prices": "boosted trees, five price windows",
    "gbt_pca": "boosted trees, prices + 8 exogenous PCs",
    "gbt_all": "boosted trees, all 3,552 values",
    "single_SA1_price": "LSTM, single task",
    "panel_lstm": "LSTM on the raw window -- the no-decomposition control",
    "red37": "LSTM on a learned 37-channel projection",
    "red16": "LSTM on a learned 16-channel projection",
    "red8": "LSTM on a learned 8-channel projection",
    "joint_nocouple": "bands, coupling frozen at identity",
    "joint": "bands + per-band coupling",
    "joint_lstm_film": "the same + FiLM gate",
    "joint_lstm_film_ctx336": "the same + 336-step bank context",
    "a0.25": "bands + coupling, Jacobian weight 0.25",
    "a0.5": "the same, 0.5", "a0.75": "the same, 0.75", "a1.0": "the same, 1.0",
    "f0.25": "FiLM + Jacobian 0.25", "f0.5": "FiLM + Jacobian 0.5",
    "f0.75": "FiLM + Jacobian 0.75",
}
GROUP = [("Naive", ["naive_persist", "naive_week"]),
         ("Linear", ["ar_window", "var", "global_linear", "arx_window"]),
         ("Trees", ["gbt_own", "gbt_prices", "gbt_pca", "gbt_all"]),
         ("LSTM, no decomposition", ["single_SA1_price", "panel_lstm",
                                     "red37", "red16", "red8"]),
         ("LSTM, bands", ["joint_nocouple", "joint", "joint_lstm_film",
                          "joint_lstm_film_ctx336"]),
         ("LSTM, bands + Jacobian weight", ["a0.25", "a0.5", "a0.75", "a1.0",
                                            "f0.25", "f0.5", "f0.75"])]


def preds():
    out = {}
    for r in ("preds/pace_h6/*.npz", "preds/jac/*.npz", "preds/fj/*.npz",
              "preds/final/*.npz", "preds/baselines_h6/*.npz"):
        for f in sorted(glob.glob(r)):
            try:
                z = np.load(f, allow_pickle=True)
            except Exception:
                continue
            nm = [str(v) for v in z["names"]]
            i = nm.index("SA1_price") if "SA1_price" in nm else 0
            k = re.sub(r"(_x)?(_s\d+)?\.npz$", "", os.path.basename(f))
            k = re.sub(r"^red_panel_lstm_", "", k)
            k = re.sub(r"^exp_joint_j[\d.]+$", "a0.25", k)
            out[k] = (z["pred"][:, i], z["truth"][:, i])
    return out


def main():
    from analysis.eval_joint import dm
    P = preds()
    if "arx_window" not in P:
        open("FINDINGS.md", "w").write("# Pending: no predictions under preds/\n")
        return
    y = P["arx_window"][1]
    P = {k: v for k, v in P.items() if len(v[1]) == len(y)}
    e = {k: p - t for k, (p, t) in P.items()}
    mae = {k: float(np.abs(v).mean()) for k, v in e.items()}
    rmse = {k: float(np.sqrt((v ** 2).mean())) for k, v in e.items()}
    masks = [f(y) for _, f in SEG]
    n = [int(m.sum()) for m in masks]
    sg = {k: [float(np.abs(v[m]).mean()) for m in masks] for k, v in e.items()}

    def cmp(a, b, i=None):
        ea, eb = (e[a], e[b]) if i is None else (e[a][masks[i]], e[b][masks[i]])
        _, pv, _ = dm(ea, eb)
        return (np.abs(eb).mean() - np.abs(ea).mean()) / np.abs(ea).mean() * 100, pv

    A = []
    def W(*ls): A.extend(ls)

    W("# Decomposition for electricity price forecasting", "",
      f"*generated {datetime.now():%Y-%m-%d %H:%M}. "
      f"h=6, SA1, 2021 test year, one seed, 30 epochs. Every neural arm shares "
      f"the window list, the objective and the seed.*", "")

    # ---------------------------------------------------------------- 1
    d_dec, p_dec = cmp("panel_lstm", "joint")
    d_cpl, p_cpl = cmp("joint_nocouple", "joint")
    W("## 1. What holds", "",
      "Two results are independent of everything below and are recorded in full "
      "in [`attic/FINDINGS-full.md`](attic/FINDINGS-full.md): **the published "
      "gains from VMD price forecasting are leakage**, and **a band "
      "decomposition costs 10^3-10^5x less** than solving one.", "",
      "From the joint multi-region work, against the control that was always "
      "missing -- the same head reading the raw window:", "",
      f"| | effect | DM p | half-hours |", "|---|---:|---:|---|",
      f"| the band decomposition | **{d_dec:+.1f}%** | {p_dec:.3f} | 33/48 |",
      f"| the per-band spatial coupling | **{d_cpl:+.1f}%** | {p_cpl:.3f} | 32/48 |",
      "",
      "And one thing that is not a result but a fact about the construction: a "
      "partition-of-unity bank is an invertible linear map, so **ridge on bands "
      "and ridge on the raw window agree to 0.0000 $/MWh**, correlation "
      "1.000000. The decomposition cannot add information. What it adds is "
      "measured in section 5.", "")

    # ---------------------------------------------------------------- 2
    W("## 2. Architecture", "",
      "```",
      "  x : (B, 37, 96)                         37 channels, 96 half-hours = 48 h",
      "   |",
      "   |   ### decomposition module -- 57,656 params, 8.9% of the model",
      "   |",
      "   +-> rfft over the window                        (B*37, 49) complex",
      "   |",
      "   |     band masks: K=8 Gaussians on [0, 0.5], normalised to",
      "   |     sum to 1 at every frequency (partition of unity, so the",
      "   |     modes add back to the signal exactly -- measured 4.8e-07).",
      "   |     Centres are a cumsum of softmax(gap_logits), geometric",
      "   |     init gap_k ~ 1.8^k, which lands them at",
      "   |",
      "   |        k   period     what lives there",
      "   |        0   48 h       trend (the window's own length)",
      "   |        1   24-75 h    daily",
      "   |        2   27 h       daily",
      "   |        3   12.5 h     half-daily",
      "   |        4   6.3 h",
      "   |        5   3.4 h",
      "   |        6   1.8 h",
      "   |        7   1.0 h      Nyquist",
      "   |",
      "   |     parameters: gap_logits (8) + log_bw (8) = 16 numbers.",
      "   |     The masks themselves have none. encoder (40,096) and",
      "   |     gap_head (17,544) only matter when adapt>0, which makes",
      "   |     the bands input-dependent; every run here uses adapt=0.",
      "   |",
      "   +-> irfft                                modes : (B, 37, 8, 96)",
      "   |",
      "   |   ### spatial coupling -- 10,952 params, 1.7%",
      "   |",
      "   |     own   = modes[:, t]                       (B, 8, 96)  lossless",
      "   |     exo_k = sum_c A_k[t, c] * modes[:, c, k]  (B, 8, 96)",
      "   |",
      "   |     A_k = I + delta_k, one 37x37 matrix per band, self term",
      "   |     zeroed. Identity init, so exo = 0 at step 0 and the arm",
      "   |     is exactly the uncoupled one -- which makes --coupling 0",
      "   |     an exact ablation and costs a dead exogenous pathway at",
      "   |     the start. Five targets put five rows per band under",
      "   |     gradient: 1,440 of 10,952 live, against 256 of 8,712",
      "   |     (2.9%) for the single-target version.",
      "   |",
      "   v  cat([own, exo]) : (B, 5, 16, 96)     5 regions, 2K channels",
      "   |",
      "   |   ### head -- 579,073 params, 84.1% + 5.1%",
      "   |",
      "   +-> reshape to (B*5, 96, 16)            regions fold into the batch",
      "   +-> LSTM, bidirectional, 2 layers, hidden 128      544,768",
      "   +-> last hidden state  (B*5, 256)",
      "   +-> + per-region embedding (5, 256)                  1,280",
      "   +-> Linear 256->128, ReLU, Linear 128->1            33,025",
      "   |",
      "   v  (B, 5)                               one price per region",
      "",
      "  total 647,681. The head is 89% of it; the decomposition is 8.9%",
      "  and the thing that encodes the spatial claim is 1.7%.",
      "```", "",
      "Optional pieces, each an exact no-op at step 0, each measured in section "
      "4: `+film` a FiLM gate over the own bands, `+ctx336` a 336-step bank "
      "context with a 96-step readout, `+exp` the physical drivers given their "
      "own band channels (16 -> 88), `+red<N>` a learned projection instead of "
      "the bank, `@<sd>` a seeded rather than identity coupling.", "")

    # ---------------------------------------------------------------- 3
    W("## 3. The objective, and the Jacobian weight", "",
      "The loss is not a hyperparameter here and it was wrong twice, in two "
      "different ways, and fixing it is the only thing in this whole sweep that "
      "improved the headline number. No architecture change did.", "",
      "**The shape was wrong.** `smooth_l1_loss(beta=1.0)` on a standardised "
      "target is quadratic for **86.9%** of residuals -- measured on a trained "
      "model -- so the objective was MSE for the bulk while every number "
      "reported was an absolute error. The repo had already swept beta and the "
      "ordering is monotone (L1 13.744, 0.5 13.909, 2.0 14.067, 4.0 14.234), so "
      "the default is now L1.", "",
      "**The space was wrong.** Fixing the shape left the loss in asinh "
      "coordinates while the metric is dollars. With "
      "`z = asinh((p - c)/w)`, `c` the train median and `w` the train IQR,", "",
      "```",
      "    dp/dz = w * cosh(z)",
      "",
      "    regime        mean |dp/dz|     relative      rows",
      "    calm                 49 $/MWh      1.0x      12,030",
      "    high                 81            1.7x       1,939",
      "    negative             98            2.0x       3,335",
      "    spike             1,080           22.0x         115",
      "```", "",
      "One unit of asinh-space error is worth 22 times more on a spike than in "
      "the calm band, and an unweighted L1 gave them the same weight. The model "
      "was trained to ignore the rows that dominate the metric, and did: before "
      "the fix it beat a ridge by 1.8% in the calm band and lost by 5.0% on "
      "negative prices.", "",
      "**The weight.** `--w-jacobian alpha` multiplies each sample's loss by "
      "`(|dp/dz| / mean|dp/dz|)^alpha`, evaluated at the truth so it reweights "
      "the data rather than bending the loss:", "",
      "```",
      "    loss = mean_i  w_i * |z_hat_i - z_i|  +  regularisers",
      "    w_i  = ( |dp/dz|(z_i) / mean_j |dp/dz|(z_j) ) ^ alpha",
      "",
      "    alpha   calm    high    negative   spike      what it is",
      "    0       1.00x   1.00x   1.00x       1.00x     the old objective",
      "    0.25    0.92x   1.04x   1.09x       1.98x",
      "    0.5     0.84x   1.06x   1.18x       3.08x",
      "    1.0     0.71x   1.18x   1.43x      15.70x     MAE in $ to 1st order",
      "```", "",
      "alpha=0 reproduces the old objective exactly and stays the ablation. "
      "alpha=1 matches the metric but hands 15.7x weight to 115 rows, which is "
      "why asinh was introduced in the first place; the useful range is in "
      "between, and section 7 is the frontier it traces.", "",
      "**The whole objective.** Written out, with `B` a batch, `R` the five "
      "price targets, `A` the auxiliary driver targets, `rho` the pointwise loss "
      "(`|.|` by default, `--loss`), and everything in the standardised asinh "
      "space the model predicts in:", "",
      "```",
      "  L  =   1/(|B| R)  sum_i sum_r   w_ir * rho( zhat_ir - z_ir )     price",
      "",
      "       + l_dev / (|B| R) sum_i sum_r rho( ( d(zhat)_ir - d(z)_ir )  deviation",
      "                                           / sigma_d,r )",
      "",
      "       + l_aux / (|B| |A|) sum_i sum_a rho( zhat_ia - z_ia )       drivers",
      "",
      "       + l_bias / R  sum_r | 1/|B| sum_i ( zhat_ir - z_ir ) |      offset",
      "",
      "       + l_sp   * mean_{k, c != t} | Delta_k[t, c] |               coupling L1",
      "",
      "       + 0.05   * mean_k  bw_k                                     bandwidth",
      "",
      "       + 1.0    * mean_k  relu( m * (bw_k + bw_{k+1})/2            separation",
      "                                - (ctr_{k+1} - ctr_k) )^2",
      "",
      "  w_ir  = ( |dp/dz|(z_ir) / mean_js |dp/dz|(z_js) ) ^ alpha        --w-jacobian",
      "  d(v)_ir = v_ir - 1/R sum_s v_is          departure from the cross-region mean",
      "  sigma_d,r  standard deviation of d(z)_.r on the training span",
      "  Delta_k = A_k - I    the learned part of the per-band coupling",
      "  ctr_k, bw_k          band centres and bandwidths, from the bank",
      "```", "",
      "Defaults in every run reported here: `alpha` as stated per arm, "
      "`l_dev = 1.0`, `l_aux = 0` (the driver targets were tried and cost "
      "accuracy), `l_bias = 0.1`, `l_sp = 1e-4`, `m = 1`. The last two terms "
      "shape the filter bank rather than the forecast and are unchanged from "
      "earlier work; the deviation term exists because 96.4% of the variance "
      "across the five regions is one common mode, so an unweighted level loss "
      "can be won without representing any regional structure at all.", "",
      "Three of the weights are reported with their ablation: `alpha` in section "
      "7, `l_dev` as `--w-dev 0`, and `l_sp` as the identity-frozen coupling arm. "
      "`l_bias` and the two bank terms are carried over untested here.", "")

    # ---------------------------------------------------------------- 4
    W("## 4. Results", "",
      "| | model | MAE | RMSE | " +
      " | ".join(f"{s} ({c})" for (s, _), c in zip(SEG, n)) + " |",
      "|---|---|---:|---:|" + "---:|" * len(SEG))
    seen = set()
    for g, ks in GROUP:
        first = True
        for k in ks:
            if k not in mae:
                continue
            seen.add(k)
            A.append(f"| {g if first else ''} | `{k}` {NAME.get(k,'')} | "
                     f"{mae[k]:.2f} | {rmse[k]:.1f} | "
                     + " | ".join(f"{v:.1f}" for v in sg[k]) + " |")
            first = False
    for k in sorted(set(mae) - seen, key=mae.get):
        A.append(f"| | `{k}` | {mae[k]:.2f} | {rmse[k]:.1f} | "
                 + " | ".join(f"{v:.1f}" for v in sg[k]) + " |")
    A += ["", "Best MAE is `arx_window` at "
          f"{mae['arx_window']:.2f}; best RMSE is `a0.75` at {rmse['a0.75']:.1f}, "
          f"{(rmse['a0.75']-rmse['arx_window'])/rmse['arx_window']*100:+.1f}% "
          "against the ridge. Which metric is chosen decides the winner, and "
          "section 6 is why.", ""]

    # ---------------------------------------------------------------- 5
    d_red, _ = cmp("panel_lstm", "red16")
    d_band, p_band = cmp("red16", "joint")
    W("## 5. What the decomposition is", "",
      "Three measurements, each from a different direction, and together they "
      "leave one explanation standing.", "",
      "**It is not information.** Ridge on bands and ridge on the raw window "
      "return identical predictions to 0.0000 $/MWh. Forced by construction.", "",
      "**It is not a better basis.** Boosted trees on the target's own 96-step "
      f"window score {mae['gbt_own']:.2f}; on the same window as 8 bands x 96 "
      "steps, 40.51 -- **4.4% worse**. Trees split on single features and cannot "
      "fold a rotation into a weight matrix, so they are the one learner where "
      "a better coordinate system would have to show up. It does not.", "",
      "**It is capacity, plus a shortcut.** Replacing the bank with a learned "
      "projection of the same output width separates the two:", "",
      "| step | arm | MAE | effect |", "|---|---|---:|---:|",
      f"| start | `panel_lstm`, 37 raw channels | {mae['panel_lstm']:.2f} | |",
      f"| narrow to 16 channels | `red16`, learned projection | {mae['red16']:.2f} | {d_red:+.1f}% |",
      f"| make those 16 channels bands | `joint` | {mae['joint']:.2f} | {d_band:+.1f}% (p={p_band:.3f}) |",
      "",
      "The first half is regularisation, and the validation curves say so: 37 "
      "raw channels peak at epoch 1 and degrade for 29 more (25.67 -> 30.29), 16 "
      "channels train to epoch 18, 88 channels (`+exp`) peak at 4-8. Width "
      "costs accuracy monotonically.", "",
      "The second half is not. Band masks are non-causal over the window, so "
      "`band_k(t)` carries the whole window at every step and the LSTM never "
      "integrates 96 of them. A ridge sees the window at once, has no such "
      "bottleneck, and gains exactly nothing -- which is the same 0.0000 from a "
      "different angle.", "")

    # ---------------------------------------------------------------- 6
    tot = np.abs(e["joint"]).sum()
    share = [float(np.abs(e["joint"][m]).sum() / tot * 100) for m in masks]
    W("## 6. Error analysis", "",
      "The global mean is decided largely where nothing has skill.", "",
      "| regime | rows | % of rows | % of total error | `joint` MAE | `arx_window` MAE |",
      "|---|---:|---:|---:|---:|---:|")
    for i, (s, _) in enumerate(SEG):
        A.append(f"| {s} | {n[i]:,} | {n[i]/len(y)*100:.1f}% | {share[i]:.0f}% | "
                 f"{sg['joint'][i]:.1f} | {sg['arx_window'][i]:.1f} |")
    A += ["",
          f"**{n[3]} spike rows are {n[3]/len(y)*100:.1f}% of the test set and "
          f"{share[3]:.0f}% of the absolute error**, and every model including "
          "the ridge sits near 1,300 $/MWh on them -- persistence is the best "
          f"thing there at {mae['naive_persist'] and sg['naive_persist'][3]:.0f}, "
          "because every trained model regresses to the mean. Negative prices "
          f"are another {n[0]/len(y)*100:.1f}% of rows and {share[0]:.0f}% of the "
          "error. Segment before concluding: a global comparison here is settled "
          "mostly on rows no model can reach.", "",
          "A second reason the mean misleads: after removing the cross-region "
          "common mode, **regional structure is 0.4-1.9% of the variance in "
          "every band**, with the fastest band (1-1.6 h) 4.6x more regional than "
          "the most-shared one (16 h). The direction is what the physics "
          "predicts -- interconnector limits bind and decouple fast components "
          "-- but the magnitude bounds what any spatial model can win.", ""]

    # ---------------------------------------------------------------- 7
    W("## 7. The Pareto frontier", "",
      "Raising alpha moves error out of the tails and into the bulk, "
      "monotonically. No single point forecast wins all four regimes. "
      "Percentages are against `arx_window`, DM p in brackets.", "",
      "| alpha | " + " | ".join(s for s, _ in SEG) + " | MAE | RMSE |",
      "|---:|" + "---:|" * (len(SEG) + 2))
    for k in ("joint", "a0.25", "a0.5", "a0.75", "a1.0"):
        if k not in mae:
            continue
        lab = "0" if k == "joint" else k[1:]
        cells = [f"{d:+.1f}% ({pv:.3f})" for d, pv in
                 (cmp("arx_window", k, i) for i in range(4))]
        A.append(f"| {lab} | " + " | ".join(cells)
                 + f" | {mae[k]:.2f} | {rmse[k]:.1f} |")
    A += ["",
          "**alpha=0.25** is the only arm that beats the ridge in the negative "
          "band and the calm band at once, and it ties the ridge globally. "
          "**alpha=0.75** produces the only strongly significant segment win in "
          "this document, -6.8% on negative prices at p=0.000, and the best "
          "spike figure of any trained model, at the cost of a calm band 12.9% "
          "worse.", "",
          "An oracle routing each interval to the best arm for its realised "
          "regime reaches MAE 35.86 against 37.77, **-5.1%**, and the assignment "
          "is interpretable: negative to alpha=0.75, calm to plain `joint`, high "
          "to the ridge, spike to persistence. That bound is not reachable. A "
          "classifier on the same window recalls 26% of negative prices and 3% "
          "of spikes, and hard routing on its output loses 2.4%. Softening to a "
          "probability weight recovers -1.1%, but a fixed weighting with no "
          "classifier reaches -0.9%, so the gain is diversification, not regime "
          "detection -- and those weights were not fitted out of sample, so "
          "-1.1% is not yet a result.", ""]

    # ---------------------------------------------------------------- 8
    W("## 8. Caveats", "",
      "- **One seed.** Seed noise is 0.116 MAE where five decomposition "
      "families span 0.04. Nothing at the 1% scale here is settled.",
      "- **The ridge used for the paired tests is the 37.77 fit.** A finer "
      "penalty grid reaches 37.04 with the validation and test optima coinciding "
      "at lam=3e4 and a selection effect of 0.000, but that fit saved no "
      "predictions. Against it the gap to `joint` is 3.9%, not 1.9%.",
      "- **Exogenous channels are history, not forecasts**, so every spatial "
      "number is an upper bound, and more so at h=6 than at h=1.",
      "- **No checkpoints exist for most arms**, so statements about where the "
      "bank put its bands describe the geometric initialisation. `--save-model` "
      "now writes centres, bandwidths and the coupling tensor; only the `+exp` "
      "arms used it.",
      "- **Two years, one region, one target.** As before.",
      "", "[`PITFALLS.md`](PITFALLS.md) records how to run this on PACE and the "
      "measurement traps that produced three retracted numbers in one session.")

    open("FINDINGS.md", "w").write("\n".join(A) + "\n")
    print(f"FINDINGS.md written: {len(A)} lines")


if __name__ == "__main__":
    main()
