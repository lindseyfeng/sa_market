#!/usr/bin/env python3
"""
Causal per-timestep decompositions, one file per method, same convention as
`vmd_panel_modes.py`: window [t-W, t), modes read at the last step, K channels,
first W-1 rows NaN.

The point is not another decomposition. It is to place several of them on one
axis -- how much the basis moves between adjacent windows -- and see whether
that, rather than decomposition quality, predicts downstream error.

    adaptive, re-solved every window:  vmd, ewt, emd
    fixed basis, identical everywhere: wpt, bank

    python decomp_zoo.py --methods ewt,emd,wpt,bank --years 2018,2019
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
K_DEFAULT = 8


# ---------------------------------------------------------------- methods
def _ewt(chunk, K):
    from ewtpy import EWT1D
    ewt, _, _ = EWT1D(chunk, N=K)
    return ewt.T                                     # (K, L)


def _emd(chunk, K):
    """EMD gives a variable number of IMFs; fold the tail into the last slot
    so the K channels still sum to the signal exactly."""
    from PyEMD import EMD
    imfs = EMD().emd(chunk, max_imf=K)
    out = np.zeros((K, len(chunk)))
    n = min(len(imfs), K)
    out[:n] = imfs[:n]
    out[K - 1] += chunk - out.sum(0)                 # residue + any extra IMFs
    return out


def _wpt(chunk, K):
    """Wavelet packet at level 3 -> 8 fixed frequency bands, reconstructed
    separately so they sum to the signal."""
    import pywt
    level = int(np.log2(K))
    out = np.zeros((K, len(chunk)))
    nodes = [n.path for n in pywt.WaveletPacket(chunk, "db4", "symmetric",
                                                maxlevel=level).get_level(level, "freq")]
    for i, path in enumerate(nodes):
        wp = pywt.WaveletPacket(chunk, "db4", "symmetric", maxlevel=level)
        for p in nodes:
            if p != path:
                wp[p].data = np.zeros_like(wp[p].data)
        out[i] = wp.reconstruct(update=False)[:len(chunk)]
    return out


def _bank_centres(K):
    ratio = 1.8
    g = ratio ** np.arange(K)
    c = np.concatenate([[0.0], np.cumsum(g)[:-1]])
    return c / c[-1] * 0.5


def _apply_bank(chunk, c):
    """Gaussian bands at centres `c`, normalised to a partition of unity."""
    L = len(chunk)
    f = np.fft.rfftfreq(L)
    gap = np.diff(c, prepend=c[0], append=c[-1])
    bw = 0.6 * 0.5 * (gap[:-1] + gap[1:]) + 1e-6
    m = np.exp(-0.5 * ((f[None, :] - c[:, None]) / bw[:, None]) ** 2)
    m /= m.sum(0, keepdims=True).clip(1e-8)
    return np.fft.irfft(np.fft.rfft(chunk)[None, :] * m, n=L, axis=-1)


def _bankjit(chunk, K, scale, widx):
    """The fixed bank, with its centres deliberately moved in every window.

    This is the dose-response control.  VMD's centres wander because the
    optimisation re-solves per window; here the wander is injected directly,
    at a magnitude we choose, with everything else held identical.  The
    perturbation is seeded by the window index, so it is a property of the
    *basis* -- the same window always gets the same bank -- rather than noise
    that a model could average away over epochs.
    """
    c = _bank_centres(K)
    gap = np.diff(c, prepend=c[0], append=c[-1])
    local = 0.5 * (gap[:-1] + gap[1:])
    rng = np.random.default_rng(widx)
    c = np.sort(np.clip(c + scale * local * rng.standard_normal(K), 0.0, 0.5))
    return _apply_bank(chunk, c)


def _bank(chunk, K):
    """Fixed Gaussian filter bank, geometric centres -- the same bank the
    `fixed_geo` arm uses.  Included so its drift can be measured on the same
    footing as the adaptive methods (it should be exactly zero)."""
    return _apply_bank(chunk, _bank_centres(K))


METHODS = {"ewt": _ewt, "emd": _emd, "wpt": _wpt, "bank": _bank}


def spectral_centroid(modes):
    """(K, L) -> (K,) energy-weighted mean frequency of each mode."""
    F = np.abs(np.fft.rfft(modes, axis=-1)) ** 2
    f = np.fft.rfftfreq(modes.shape[-1])
    return (F * f).sum(-1) / F.sum(-1).clip(1e-12)


def resolve(meth):
    """`bankjit:0.35` -> a callable with that jitter scale baked in."""
    if meth.startswith("bankjit:"):
        scale = float(meth.split(":", 1)[1])
        return lambda chunk, K, widx: _bankjit(chunk, K, scale, widx)
    fn = METHODS[meth]
    return lambda chunk, K, widx: fn(chunk, K)


def _one(t, sig, W, K, fn, pad):
    chunk = np.pad(sig[t - W:t], (0, pad), mode="edge")
    try:
        m = fn(chunk, K, t)[:, :W]
        cen = spectral_centroid(m)
        order = np.argsort(cen)                      # the usual rank-sort
        return m[order][:, -1], cen[order]
    except Exception:
        return np.full(K, np.nan), np.full(K, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/compound_2018_2022.csv")
    ap.add_argument("--channel", default="SA1_price")
    ap.add_argument("--methods", default="ewt,emd,wpt,bank")
    ap.add_argument("--years", default="2018,2019")
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.panel, parse_dates=["SETTLEMENTDATE"])
    W, K = args.window, args.K
    pad = min(20, W // 4)

    for meth in args.methods.split(","):
        outdir = f"cache/decomp_{meth.replace(':', '')}_K{K}_W{W}"
        os.makedirs(outdir, exist_ok=True)
        for year in [int(y) for y in args.years.split(",")]:
            mp = os.path.join(outdir, f"{year}_{args.channel}.npy")
            cp = os.path.join(outdir, f"{year}_{args.channel}_centroids.npy")
            if os.path.exists(mp) and os.path.exists(cp):
                print(f"{meth} {year}: cached", flush=True)
                continue
            sig = df[df.SETTLEMENTDATE.dt.year == year][args.channel].to_numpy(float)
            T = len(sig)
            t0 = time.time()
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(_one)(t, sig, W, K, resolve(meth), pad)
                for t in range(W, T + 1))
            modes = np.full((T, K), np.nan, np.float32)
            cents = np.full((T, K), np.nan, np.float32)
            for t, (m, c) in zip(range(W, T + 1), res):
                modes[t - 1], cents[t - 1] = m, c
            np.save(mp, modes)
            np.save(cp, cents)
            bad = int(np.isnan(modes[W - 1:]).any(1).sum())
            print(f"{meth} {year}: {T} rows, {time.time()-t0:.0f}s, "
                  f"{bad} failed -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
