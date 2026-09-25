#!/usr/bin/env python3
r"""Functional band ablation: intervene, do not read the weights.

The coupling row A_k[t,:] is not identifiable -- flipping A_k -> -A_k and the
downstream weights W -> -W leaves the forecast unchanged, which is why the two
seeds' raw couplings correlate at -0.9 on five bands.  Zeroing a contribution
and measuring the damage is invariant to that symmetry.

  band k      : exo_k        <- 0        ->  dMAE_k
  channel c   : A_k[t,c]     <- 0 for all k  ->  dMAE_c
  cell (k,c)  : A_k[t,c]     <- 0            ->  dMAE_{k,c}
"""
import sys, json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.nvmd_st import STForecaster
from experiments.train_nvmd_st import PanelWindowDataset

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", default=["runs/runs_cat_s1/best.pt",
                                              "runs/runs_cat_s2/best.pt"])
ap.add_argument("--panel", default="data/raw/compound_2018_2022.csv")
ap.add_argument("--train-years", default="2018")
ap.add_argument("--test-years", default="2019")
ap.add_argument("--seq-len", type=int, default=96)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--out", default="results/band_ablation.json")
a = ap.parse_args()
torch.set_num_threads(4)

df = pd.read_csv(a.panel, parse_dates=["SETTLEMENTDATE"])
chans = [c for c in df.columns if c != "SETTLEMENTDATE"]
tr = df[df.SETTLEMENTDATE.dt.year.isin([int(x) for x in a.train_years.split(",")])][chans].to_numpy(float)
te = df[df.SETTLEMENTDATE.dt.year.isin([int(x) for x in a.test_years.split(",")])][chans].to_numpy(float)
tr_ds = PanelWindowDataset(tr, a.seq_len)
te_ds = PanelWindowDataset(te, a.seq_len, mean=tr_ds.mean, std=tr_ds.std)
dl = DataLoader(te_ds, batch_size=a.batch, shuffle=False)
mu, sd_ = tr_ds.mean[0], tr_ds.std[0]
print(f"{len(chans)} channels, {len(te_ds)} test windows, target = {chans[0]}")

@torch.no_grad()
def mae(model):
    """p and y are both (B, 1).  Squeezing only one of them broadcasts to
    (B, B) and inflates the error by ~3 orders of magnitude -- the bug that
    invalidated the first run of this script."""
    e = 0.0; n = 0
    for x, y in dl:
        p = model(x)[3]
        assert p.shape == y.shape, (p.shape, y.shape)
        e += ((p - y).abs() * sd_).sum().item(); n += len(y)
    return e / n

def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = STForecaster(R=len(chans), K=8, signal_len=a.seq_len,
                     adapt=0.0, coupling=True, concat=True)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, ck["best_mae"]

# band centres -> a physical label
def periods(model):
    with torch.no_grad():
        c, _ = model.decomposer.decomposer.bands()
    c = c[0].numpy()
    return ["DC" if v < 1e-9 else f"{0.5/v:.1f}h" for v in c]

OUT = {}
for path in a.ckpt:
    model, rec = load(path)
    K, R = 8, len(chans)
    base = mae(model)
    labs = periods(model)
    print(f"\n=== {path}   recorded {rec:.3f}   re-scored {base:.3f}")

    delta = model.decomposer.coupling.delta.data.clone()

    # --- whole band
    band = []
    for k in range(K):
        model.decomposer.coupling.delta.data = delta.clone()
        model.decomposer.coupling.delta.data[k, 0, :] = 0.0      # row t, band k
        model.decomposer.coupling.delta.data[k, 0, 0] = -1.0     # kill the identity too
        band.append(mae(model) - base)
    model.decomposer.coupling.delta.data = delta.clone()
    print("  band ablation, dMAE (higher = the band's exogenous term matters more)")
    for k in range(K):
        bar = "#" * max(0, int(40 * band[k] / max(max(band), 1e-9)))
        print(f"    k={k+1} {labs[k]:>6}  {band[k]:+7.4f}  {bar}")

    # --- whole channel, all bands
    chan = []
    for c in range(1, R):
        model.decomposer.coupling.delta.data = delta.clone()
        model.decomposer.coupling.delta.data[:, 0, c] = 0.0
        chan.append(mae(model) - base)
    model.decomposer.coupling.delta.data = delta.clone()
    order = np.argsort(-np.array(chan))[:8]
    print("  channel ablation, top 8")
    for i in order:
        print(f"    {chans[i+1]:24} {chan[i]:+7.4f}")

    OUT[path] = dict(base=base, recorded=rec, labels=labs,
                     band=band, chan=chan, chan_names=chans[1:])

Path("results").mkdir(exist_ok=True)
json.dump(OUT, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")

if len(a.ckpt) == 2:
    x, y = (np.array(OUT[p]["band"]) for p in a.ckpt)
    u, v = (np.array(OUT[p]["chan"]) for p in a.ckpt)
    print(f"\nseed agreement (this is the test the raw weights failed)")
    print(f"  band dMAE, corr    {np.corrcoef(x, y)[0,1]:+.3f}")
    print(f"  channel dMAE, corr {np.corrcoef(u, v)[0,1]:+.3f}")
