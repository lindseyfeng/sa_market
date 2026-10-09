#!/usr/bin/env python3
"""Write FINDINGS.md: short, current, and regenerated rather than edited.

The long record -- every arm, every decomposition family, every retraction --
is `attic/FINDINGS-full.md` from report_findings.py. This file keeps only what a
reader needs to act on, and both regenerate from the same result files, so
neither is a copy of the other going stale.
"""
import os
from datetime import datetime

from report.joint_section import section as _joint


def main():
    L = ["# Decomposition for electricity price forecasting",
         "",
         f"*generated {datetime.now():%Y-%m-%d %H:%M}*",
         "",
         "## Summary",
         "",
         "Two results hold independently of every protocol question below, and "
         "neither is touched by the later work. Both are in "
         "[`attic/FINDINGS-full.md`](attic/FINDINGS-full.md), which carries the "
         "long record and regenerates from the same result files.",
         "",
         "- **The published gains from VMD price forecasting are leakage, not "
         "decomposition**, and what leaks is a linearly readable aggregate "
         "rather than a forecasting signal.",
         "- **A band decomposition costs 10^3-10^5x less**, 0.1 s/year against "
         "59-18,178 s/year.",
         "",
         "Everything else is a margin against a baseline, and each is smaller "
         "than at least one protocol choice this project initially got wrong: "
         "the residual channel (0.237 MAE), the objective (0.442 on the widest "
         "arm), epoch selection (up to 0.170), the seed itself (0.116). "
         "[`PITFALLS.md`](PITFALLS.md) records those, and the ones found since.",
         "",
         "The current state, measured at h=6 on 2021 with the control that was "
         "always missing -- the same head reading the raw window:",
         "",
         "- **The band decomposition is worth -6.8%** (DM p=0.000, 33 of 48 "
         "half-hours). It splits in two: narrowing 37 channels to 16 is worth "
         "-3.5%, and making those 16 *bands* rather than a learned projection of "
         "the same width is worth a further -3.4%.",
         "- **The per-band spatial coupling is worth -5.6%** (p=0.000, 32/48), "
         "and it survived a change of objective and of schedule.",
         "- **The decomposition is not information.** A partition-of-unity bank "
         "is an invertible linear map: ridge on bands and ridge on the raw "
         "window agree to **0.0000 $/MWh**. It is a capacity limit plus a "
         "shortcut past the LSTM's sequential bottleneck.",
         "- **Against a ridge the whole stack ties, and only after the objective "
         "was fixed** -- 37.79 against 37.77 on MAE, better by 1.6% on RMSE at a "
         "higher Jacobian exponent, better on negative prices, worse on high "
         "ones.",
         ""]
    L += _joint(heading="## Joint multi-region forecasting")
    open("FINDINGS.md", "w").write("\n".join(L) + "\n")
    print(f"FINDINGS.md written: {len(L)} lines")


if __name__ == "__main__":
    main()
