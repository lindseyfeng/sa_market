# nvmd_train_rrp2modes.py
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Your model (B,1,L) -> (B,K,L)
from nvmd_autoencoder import NVMD_Autoencoder


# -----------------------------
# 1) Dataset: RRP window -> K-vector at last timestep
# -----------------------------
class RRP2ModesDataset(Dataset):
    """
    Each sample:
      - Input: sliding window of RRP of length L, shape (L,)
      - Target: K-mode vector at the LAST index, shape (K,)
    CSV columns required: 'RRP', 'Mode_1'..'Mode_K'
    """
    def __init__(self, csv_file: str, K: int = 12, window_size: int = 1024):
        super().__init__()
        df = pd.read_csv(csv_file)

        self.K = K
        self.window_size = window_size

        self.price = df['RRP'].to_numpy(dtype=np.float32)                     # (T,)
        mode_cols = [f"Mode_{i+1}" for i in range(K)]
        self.modes = df[mode_cols].to_numpy(dtype=np.float32)                 # (T, K)

        T = len(self.price)
        if T < window_size:
            raise ValueError(f"Series length {T} < window_size {window_size}")
        self.n_samples = T - window_size + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # Input window: (L,)
        x_win = self.price[idx : idx + self.window_size]
        # Target: modes at the LAST timestep in the window: (K,)
        y_win = self.modes[idx : idx + self.window_size, :].T  
        return torch.from_numpy(x_win), torch.from_numpy(y_win)

    # Convenience: expose raw arrays for sanity checks
    @property
    def raw_price(self):
        return self.price

    @property
    def raw_modes(self):
        return self.modes


# -----------------------------
# 2) Sanity checks for DataLoader / indexing
# -----------------------------

@torch.no_grad()
def preview_first_n(dataset: RRP2ModesDataset, n: int = 10):
    """
    Manually print the first n dataset samples (no shuffle):
      - window indices [i : i+L)
      - first/last few values of the window
      - full target K-vector at the last timestep (i+L-1)
      - alignment checks vs raw arrays
    """
    L = dataset.window_size
    K = dataset.K
    price = dataset.raw_price
    modes = dataset.raw_modes

    n = min(n, len(dataset))
    print(f"[Preview] Showing first {n} samples (dataset order, no shuffle).")
    for i in range(n):
        x_win, y_vec = dataset[i]            # tensors: (L,), (K,)
        x_win_np = x_win.numpy()
        y_vec_np = y_vec.numpy()

        # alignment checks
        assert np.array_equal(x_win_np, price[i:i+L]), f"Window mismatch at sample {i}"
        exp_y = modes[i:i+L]
        assert np.array_equal(y_vec_np, exp_y.T), f"Target mismatch at sample {i}"

        # pretty print (trim long arrays)
        head = np.array2string(x_win_np[:10], precision=5, separator=", ")
        tail = np.array2string(x_win_np[-10:], precision=5, separator=", ")
        ystr = np.array2string(y_vec_np, precision=5, separator=", ")

        print(f"\n--- Sample {i} ---")
        print(f"window idx range: [{i} : {i+L})  (last idx = {i+L-1})")
        print(f"x_win[:10]: {head}")
        print(f"x_win[-10:]: {tail}")
        print(f"y_vec (K={K}) @ last idx {i+L-1}: {ystr}")
    print("[Preview] Done.\n")



# -----------------------------
# 3) Train
# -----------------------------
def main():
    # ---- Config
    csv_file    = "VMD_modes_with_residual_2018_2021.csv"  # contains RRP + Mode_1..Mode_K
    K           = 12
    window_len  = 16
    batch_size  = 8
    epochs      = 15
    lr          = 1e-3
    num_workers = 0
    lambda_cons = 0.1   # last-timestep sum(modes) ≈ last RRP
    lambda_rec  = 1.2   # optional whole-window reconstruction: sum_k modes ≈ RRP; set >0 to enable
    max_grad_norm = 1.0 # gradient clipping

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")

    # ---- Data
    dataset = RRP2ModesDataset(csv_file=csv_file, K=K, window_size=window_len)

    # Important: shuffle=True only permutes sample order, not the raw arrays.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    # ---- One-time dataloader sanity checks
    preview_first_n(dataset)

    # ---- Model (single-channel input, K outputs across time)
    model = NVMD_Autoencoder(in_ch=1, base=64, K=K, signal_len=window_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps  = len(loader) * epochs
    warmup_steps = max(1, len(loader))  # ~1 epoch warmup
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5),
        ],
        milestones=[warmup_steps],
    )

    loss_fn = torch.nn.HuberLoss(delta=1.0)  # robust L1-ish

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ---- Training
    model.train()
    t0 = time.time()
    global_step = 0
    for ep in range(1, epochs + 1):
        running = 0.0
        for i, (x, y) in enumerate(loader, start=1):
            # x: (B, L) -> (B, 1, L)
            x = x.unsqueeze(1).to(device, non_blocking=True)      # (B, 1, L)
            y = y.to(device, non_blocking=True)                   # (B, K)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(x)                # (B, K, L)
                loss_main = loss_fn(out, y)

                if lambda_rec > 0:
                    sum_modes_win = out.sum(dim=1)        # (B, L)
                    rrp_win = x.squeeze(1)                # (B, L)
                    loss_rec = torch.nn.functional.l1_loss(sum_modes_win, rrp_win)
                else:
                    loss_rec = torch.tensor(0.0, device=device)

                loss = loss_main +  lambda_rec * loss_rec

            scaler.scale(loss).backward()
            # optional grad clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            global_step += 1
            if i % 20 == 0:
                print(
                    f"Epoch {ep} | Batch {i:04d} "
                    f"| loss: {running/20:.6f} "
                    f"| main: {loss_main.item():.6f} rec: {loss_rec.item():.6f} "
                    f"| lr: {scheduler.get_last_lr()[-1]:.2e}"
                )
                running = 0.0

        ckpt = f"nvmd_rrp2modes_ep{ep}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"[Info] Saved: {ckpt}")

    print(f"[Info] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
