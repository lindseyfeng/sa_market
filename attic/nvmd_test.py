# nvmd_test_timed.py
import time
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nvmd_autoencoder import NVMD_Autoencoder
import os  # at the top of file if not already

import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt


# -----------------------------
# 1) Dataset: RRP window -> KxL modes over the window (+ residual window)
# -----------------------------
class RRP2ModesTestDataset(Dataset):
    def __init__(self, csv_file: str, K: int = 12, window_size: int = 1024, year: int = 2022):
        super().__init__()
        df = pd.read_csv(csv_file)
        if 'SETTLEMENTDATE' not in df.columns:
            raise ValueError("CSV missing 'SETTLEMENTDATE'")
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], errors='coerce')
        df = df[df['SETTLEMENTDATE'].dt.year == year].reset_index(drop=True)

        req_modes = [f"Mode_{i+1}" for i in range(K)]
        for c in ['RRP', *req_modes]:
            if c not in df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        self.K = K
        self.L = window_size
        self.rrp   = df['RRP'].to_numpy(np.float32)          # (T,)
        self.modes = df[req_modes].to_numpy(np.float32)      # (T,K)
        self.has_resid = 'Residual' in df.columns
        self.resid = df['Residual'].to_numpy(np.float32) if self.has_resid else None

        T = len(self.rrp)
        if T < window_size:
            raise ValueError(f"Series too short: T={T} < window_size={window_size}")
        self.n = T - window_size + 1
        print(f"[Dataset] year={year}  T={T}  windows(n)={self.n}  K={K}  L={window_size}  residual={self.has_resid}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # inputs
        x_win = self.rrp[idx : idx + self.L]                       # (L,)

        # targets over the FULL window: (L, K) -> we'll transpose to (K, L)
        y_win_LK = self.modes[idx : idx + self.L, :]               # (L, K)
        y_win = np.transpose(y_win_LK, (1, 0)).copy()              # (K, L)

        # rrp/residual over the FULL window
        rrp_win   = self.rrp[idx : idx + self.L]                   # (L,)
        if self.has_resid:
            resid_win = self.resid[idx : idx + self.L]             # (L,)
        else:
            resid_win = np.zeros_like(rrp_win, dtype=np.float32)

        # also provide last-step for quick sanity checks (optional)
        rrp_last   = rrp_win[-1]
        resid_last = resid_win[-1]

        return (
            torch.from_numpy(x_win),               # (L,)
            torch.from_numpy(y_win),               # (K, L)
            torch.from_numpy(rrp_win),             # (L,)
            torch.from_numpy(resid_win),           # (L,)
            torch.tensor(rrp_last,   dtype=torch.float32),  # ()
            torch.tensor(resid_last, dtype=torch.float32),  # ()
        )

def main():
    # -----------------------------
    # 2) Setup
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_file_test = "VMD_modes_with_residual_2021_2022.csv"
    K = 12
    window_size = 1024
    batch_size = 8
    num_workers = 0

    plot_dir = "dumps"
    os.makedirs(plot_dir, exist_ok=True)
    _plot_pred = None   # will hold (K, L) for one window
    _plot_true = None   # will hold (K, L) for the same window


    ds = RRP2ModesTestDataset(csv_file_test, K=K, window_size=window_size, year=2022)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=(device == "cuda"), drop_last=False)

    xb, yb, rrp_win_b, resid_win_b, rrp_last_b, resid_last_b = next(iter(ld))
    print(f"[Check] x_win {tuple(xb.shape)}  y_win {tuple(yb.shape)}  "
          f"rrp_win {tuple(rrp_win_b.shape)}  resid_win {tuple(resid_win_b.shape)}")
    # Expect: x_win (B,L), y_win (B,K,L), rrp_win (B,L), resid_win (B,L)

    # -----------------------------
    # 3) Model & checkpoint
    # -----------------------------
    base = 64  # must match training
    model = NVMD_Autoencoder(in_ch=1, base=base, K=K, signal_len=window_size).to(device)

    ckpt_path = "nvmd_rrp2modes_ep1.pth"
    try:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[Error] Could not load checkpoint '{ckpt_path}': {e}")
        sys.exit(1)

    model.eval()
    print(f"[Model] device={device}  base={base}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # sequence losses
    mae = torch.nn.L1Loss(reduction="sum")  # we'll normalize manually

    mode_l1_sum = 0.0          # sum over all (B*K*L) elements
    rrp_l1_sum  = 0.0          # sum over all (B*L) elements
    elem_mode   = 0            # total elements for modes
    elem_rrp    = 0            # total elements for rrp
    batches = 0
    total_fwd_time = 0.0

    print_every = 50
    if device == "cuda":
        torch.cuda.synchronize()
    t_wall0 = time.perf_counter()

    # -----------------------------
    # 4) Timed inference & SEQUENCE metrics
    # -----------------------------
    with torch.inference_mode():
        for i, (x_win, y_win, rrp_win, resid_win, rrp_last, resid_last) in enumerate(ld, start=1):
            B = x_win.shape[0]
            L = x_win.shape[1]
            x = x_win.unsqueeze(1).to(device, non_blocking=True)   # (B,1,L)
            y = y_win.to(device, non_blocking=True)                # (B,K,L)
            rrp_win = rrp_win.to(device, non_blocking=True)        # (B,L)
            resid_win = resid_win.to(device, non_blocking=True)    # (B,L)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model(x)

            # REQUIRE (B,K,L)
            if isinstance(out, (tuple, list)):
                pred = out[0]
            else:
                pred = out
            if pred.dim() != 3:
                raise RuntimeError(f"Model must output (B,K,L); got {tuple(pred.shape)}")
            if pred.shape[0] != B or pred.shape[1] != y.shape[1] or pred.shape[2] != L:
                raise RuntimeError(f"Output shape mismatch. pred={tuple(pred.shape)} vs target={tuple(y.shape)} / L={L}")

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # sequence losses
            loss_modes_sum = mae(pred, y)                               # sum over B*K*L
            pred_rrp_win = pred.sum(dim=1) + resid_win                  # (B,L)
            loss_rrp_sum = mae(pred_rrp_win, rrp_win)                   # sum over B*L

            total_fwd_time += (t1 - t0)
            mode_l1_sum += float(loss_modes_sum.item())
            rrp_l1_sum  += float(loss_rrp_sum.item())
            elem_mode   += int(np.prod(y.shape))        # B*K*L
            elem_rrp    += int(np.prod(rrp_win.shape))  # B*L
            batches += 1

            # ---- collect rows for CSV (only first N windows) ----
            if _plot_pred is None:
                _plot_pred = pred[0].detach().cpu().numpy()  # (K, L)
                _plot_true = y[0].detach().cpu().numpy()     # (K, L)


            if (i % print_every == 0) or (i == len(ld)):
                print(f"[{i:05d}/{len(ld):05d}]  "
                      f"avg_mode_L1={mode_l1_sum/max(elem_mode,1):.6f}  "
                      f"avg_rrp_L1={rrp_l1_sum/max(elem_rrp,1):.6f}  "
                      f"avg_time/batch={total_fwd_time/batches:.4f}s")
                # last-step sanity check (optional)
                if i == 50:
                    b = 0   # pick first element in the batch
                    k = 0   # pick first mode/channel
                    # show first 10 timesteps only
                    print("last-step pred_rrp:", pred[b, k, :10].detach().cpu().numpy(),
                        "  true:", y[b, k, :10].detach().cpu().numpy())

    t_wall1 = time.perf_counter()

    # -----------------------------
    # 5) Report — means per element
    # -----------------------------
    print("\n=== Summary ===")
    print(f"Average mode L1 per element (B*K*L): {mode_l1_sum/max(elem_mode,1):.6f}")
    print(f"Average RRP L1 per element  (B*L):   {rrp_l1_sum/max(elem_rrp,1):.6f}")
    print(f"Forward-pass total time: {total_fwd_time:.2f}s  |  per-batch: {total_fwd_time/max(batches,1):.5f}s")
    print(f"Wall-clock elapsed (incl. dataloader): {t_wall1 - t_wall0:.2f}s")

        # ---- write CSVs ----
    if _plot_pred is not None:
        K_cur, L_cur = _plot_pred.shape
        for k in range(K_cur):
            plt.figure()
            plt.plot(_plot_true[k], label="target")
            plt.plot(_plot_pred[k], label="pred")
            plt.xlabel("timestep")
            plt.ylabel("amplitude")
            plt.title(f"Mode {k+1} — prediction vs target")
            plt.legend()
            out_path = os.path.join(plot_dir, f"mode_{k+1:02d}_pred_vs_true.png")
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close()
        print(f"[Plot] Saved per-mode figures to {plot_dir}/mode_XX_pred_vs_true.png")


if __name__ == "__main__":
    main()
