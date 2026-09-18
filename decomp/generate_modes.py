#!/usr/bin/env python3
"""
Generate per-timestep mode CSVs from different decomposition methods.

Output format (identical for all methods, compatible with existing VMD CSVs):
    Mode_1, Mode_2, ..., Mode_K, Residual, RRP, SETTLEMENTDATE
    where  sum(Mode_1..Mode_K) + Residual = RRP  at every row.

Usage:
    # ---- NVMD v2 (fast, no leakage by construction) ----
    python generate_modes.py nvmd \
        --model runs_nvmd_v2/best.pt \
        --csv SA_prices_combined_2018_2021.csv \
        --output nvmd_modes_2018_2021.csv

    # ---- Causal VMD (slow, but leak-free) ----
    python generate_modes.py causal-vmd \
        --csv SA_prices_combined_2018_2021.csv \
        --output causal_vmd_2018_2021.csv \
        --K 12 --window 96

Why causal VMD?
    The existing vmd_data_leakage.py runs VMD on the ENTIRE period at once,
    so mode values at time t are computed using data from t+1 ... end-of-period.
    Causal VMD processes each window [t-W, t] independently — mode values at
    time t never see future data.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------

def load_signal(csv_path, rrp_col="RRP", date_col="SETTLEMENTDATE"):
    df = pd.read_csv(csv_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[rrp_col])
    rrp = df[rrp_col].to_numpy(dtype=np.float32)
    dates = df[date_col].values if date_col in df.columns else None
    return rrp, dates


# -------------------------------------------------------------------
# NVMD mode generation
# -------------------------------------------------------------------

@torch.no_grad()
def generate_nvmd(args):
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    m_args = ckpt["args"]
    norm = ckpt["norm"]
    K = m_args["K"]
    L = m_args["seq_len"]
    d_model = m_args["d_model"]

    if ckpt.get("arch") == "v3":
        from models.nvmd_v3 import StructuredSpectralNVMD
        model = StructuredSpectralNVMD(K=K, signal_len=L, d_model=d_model,
                                       adapt=m_args.get("adapt", 0.5),
                                       ratio=m_args.get("ratio", 1.8))
    else:
        from models.nvmd_v2 import AdaptiveSpectralNVMD
        model = AdaptiveSpectralNVMD(K=K, signal_len=L, d_model=d_model)

    state = {k[len("decomposer."):]: v
             for k, v in ckpt["model_state"].items()
             if k.startswith("decomposer.")}
    model.load_state_dict(state)
    model.eval()

    # v3 masks are a partition of unity, so the modes already sum to the signal
    # exactly -- there is no residual channel to carve out.
    self_consistent = ckpt.get("arch") == "v3"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    rrp, dates = load_signal(args.csv, args.rrp_col, args.date_col)
    mean, std = norm["mean"], norm["std"]
    rrp_norm = (rrp - mean) / (std + 1e-8)

    # The decomposer is an rFFT -> mask -> irFFT, i.e. a *circular* convolution:
    # the window's last sample wraps against its first, so edge error peaks at
    # index L-1 -- exactly the sample we emit.  Reserve the final P slots of the
    # window for a causal pad so the emission point sits P samples inside the
    # boundary.  Real data per window is R = L - P samples ending at time t;
    # no future sample is ever read.  P=0 reproduces the old behaviour.
    P = args.edge_pad
    R = L - P
    if R < 2:
        raise ValueError(f"--edge-pad {P} leaves only {R} real samples of L={L}")

    T = len(rrp)
    num_windows = T - R + 1
    all_modes = []
    pos_energy = np.zeros(L, dtype=np.float64)   # |imf| profile across the window
    n_seen = 0

    t0 = time.time()
    bs = args.batch_size
    for start in range(0, num_windows, bs):
        end = min(start + bs, num_windows)
        windows = np.stack([rrp_norm[i : i + R] for i in range(start, end)])
        if P > 0:
            windows = np.pad(windows, ((0, 0), (0, P)), mode=args.pad_mode)
        batch = torch.from_numpy(windows).unsqueeze(1).to(device)   # (B, 1, L)
        imfs, _ = model(batch)                                       # (B, K, L)
        modes_at_t = imfs[:, :, R - 1].cpu().numpy() * std           # (B, K)
        all_modes.append(modes_at_t)

        pos_energy += imfs.abs().sum(dim=1).sum(dim=0).cpu().numpy()
        n_seen += imfs.shape[0]

    all_modes = np.concatenate(all_modes, axis=0)   # (num_windows, K)
    elapsed = time.time() - t0

    # Boundary-artifact diagnostic: mean |imf| at the emission point vs the
    # window interior.  A ratio >> 1 means the circular wrap is still leaking in.
    pos_energy /= max(n_seen, 1)
    interior = pos_energy[L // 4 : R - 1].mean() if R - 1 > L // 4 else pos_energy[: R - 1].mean()
    print(f"  Edge pad: P={P} ({args.pad_mode}), emitting window index {R-1} of {L}")
    print(f"  |imf| at emission point: {pos_energy[R-1]:.4f} | "
          f"interior mean: {interior:.4f} | ratio: {pos_energy[R-1] / (interior + 1e-8):.3f}")

    valid_start = R - 1
    valid_rrp = rrp[valid_start:]

    if self_consistent:
        # All K modes are real bands; the last carries no leftover.
        spectral = all_modes
        residual = valid_rrp - spectral.sum(axis=1)   # float error only
        col_names = [f"Mode_{i+1}" for i in range(K)] + ["Residual"]
    else:
        # Mode_1 .. Mode_{K-1} and Residual = RRP - sum(Mode_1..Mode_{K-1})
        spectral = all_modes[:, : K - 1]
        residual = valid_rrp - spectral.sum(axis=1)
        col_names = [f"Mode_{i+1}" for i in range(K - 1)] + ["Residual"]

    out = pd.DataFrame(
        np.column_stack([spectral, residual]), columns=col_names
    )
    out["RRP"] = valid_rrp
    if dates is not None:
        out["SETTLEMENTDATE"] = dates[valid_start:]

    out.to_csv(args.output, index=False)
    recon_err = np.abs(out[col_names].sum(axis=1).values - valid_rrp).mean()
    print(f"NVMD modes saved: {len(out)} rows, {K} total modes "
          f"→ {args.output}  ({elapsed:.1f}s)")
    print(f"  Reconstruction error: {recon_err:.6f} AUD/MWh")


# -------------------------------------------------------------------
# Causal VMD mode generation (truly leak-free)
# -------------------------------------------------------------------

def _vmd_one_window(t, rrp, window, K, alpha):
    """Worker: run VMD on [t-window, t), return modes at last step."""
    from vmdpy import VMD as _VMD

    chunk = rrp[t - window : t]
    pad_len = min(20, window // 4)
    chunk_pad = np.pad(chunk, (0, pad_len), mode="edge")
    try:
        u, _, omega = _VMD(chunk_pad, alpha, 0.0, K, 0, 1, 1e-7)
        u = u[:, :window]
        order = omega[-1].argsort()
        u = u[order]
        return t, u[:, -1]          # modes at last time step
    except Exception:
        return t, np.full(K, np.nan)


def generate_causal_vmd(args):
    from joblib import Parallel, delayed

    rrp, dates = load_signal(args.csv, args.rrp_col, args.date_col)
    T = len(rrp)
    W = args.window
    K = args.K

    num_windows = T - W + 1
    print(f"Causal VMD: {num_windows} windows, K={K}, W={W}, alpha={args.alpha}")
    print(f"  Using {args.n_jobs} workers.  This may take a while ...")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_vmd_one_window)(t, rrp, W, K, args.alpha)
        for t in range(W, T + 1)
    )
    elapsed = time.time() - t0

    mode_array = np.full((T, K), np.nan, dtype=np.float32)
    for t, modes in results:
        mode_array[t - 1] = modes

    valid = ~np.isnan(mode_array).any(axis=1)
    col_names = [f"Mode_{i+1}" for i in range(K)]

    out = pd.DataFrame(mode_array, columns=col_names)
    out["RRP"] = rrp
    mode_sum = out[col_names].sum(axis=1)
    out["Residual"] = rrp - mode_sum
    if dates is not None:
        out["SETTLEMENTDATE"] = dates
    out = out[valid].reset_index(drop=True)

    n_failed = int((~valid[W - 1 :]).sum())
    out.to_csv(args.output, index=False)

    recon_err = np.abs(
        out[col_names + ["Residual"]].sum(axis=1).values - out["RRP"].values
    ).mean()
    print(f"\nCausal VMD saved: {len(out)} rows, {K} modes + Residual "
          f"→ {args.output}  ({elapsed:.1f}s)")
    print(f"  Reconstruction error: {recon_err:.6f} AUD/MWh")
    if n_failed:
        print(f"  {n_failed} windows failed VMD and were dropped.")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate per-timestep mode CSVs (NVMD or causal VMD)"
    )
    sub = p.add_subparsers(dest="method")

    # ---- nvmd sub-command ----
    nv = sub.add_parser("nvmd", help="Generate modes from trained NVMD v2")
    nv.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    nv.add_argument("--csv", required=True, help="Input CSV with RRP column")
    nv.add_argument("--output", required=True, help="Output mode CSV")
    nv.add_argument("--batch-size", type=int, default=512)
    nv.add_argument("--rrp-col", default="RRP")
    nv.add_argument("--date-col", default="SETTLEMENTDATE")
    nv.add_argument("--edge-pad", type=int, default=0,
                    help="Causal right-pad slots reserved inside the window so "
                         "the emitted sample sits off the circular boundary. "
                         "0 = old behaviour; try seq_len//8.")
    nv.add_argument("--pad-mode", default="edge", choices=["edge", "reflect"],
                    help="How to fill the reserved pad slots.")

    # ---- causal-vmd sub-command ----
    cv = sub.add_parser("causal-vmd",
                        help="Leak-free per-window VMD")
    cv.add_argument("--csv", required=True, help="Input CSV with RRP column")
    cv.add_argument("--output", required=True, help="Output mode CSV")
    cv.add_argument("--K", type=int, default=12)
    cv.add_argument("--alpha", type=float, default=2000)
    cv.add_argument("--window", type=int, default=96)
    cv.add_argument("--n-jobs", type=int, default=-1)
    cv.add_argument("--rrp-col", default="RRP")
    cv.add_argument("--date-col", default="SETTLEMENTDATE")

    args = p.parse_args()
    if args.method == "nvmd":
        generate_nvmd(args)
    elif args.method == "causal-vmd":
        generate_causal_vmd(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
