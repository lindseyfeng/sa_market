# train_12_mode_models.py
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset: RRP window -> full K-vector (we'll wrap per-mode)
# -----------------------------
class RRP2ModesDataset(Dataset):
    """
    Each sample:
      - Input: RRP window of length L, shape (L,)
      - Target: K-mode vector at the LAST timestep, shape (K,)
    CSV must contain: 'RRP' and 'Mode_1'..'Mode_K'
    """
    def __init__(self, csv_file: str, K: int = 12, window_size: int = 1024):
        df = pd.read_csv(csv_file)
        self.K = K
        self.window_size = window_size
        self.price = df['RRP'].to_numpy(dtype=np.float32)                # (T,)
        mode_cols = [f"Mode_{i+1}" for i in range(K)]
        self.modes = df[mode_cols].to_numpy(dtype=np.float32)            # (T, K)
        T = len(self.price)
        if T < window_size:
            raise ValueError(f"Series length {T} < window_size {window_size}")
        self.n = T - window_size + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        x_win = self.price[idx:idx+self.window_size]                 # (L,)
        y_vec = self.modes[idx+self.window_size-1, :]                # (K,)
        return torch.from_numpy(x_win), torch.from_numpy(y_vec)

class ModeTargetDataset(Dataset):
    """ Wraps RRP2ModesDataset to return only one mode (scalar) as target. """
    def __init__(self, base_ds: RRP2ModesDataset, mode_index: int):
        self.base = base_ds
        self.m = mode_index  # 0-based
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y_vec = self.base[idx]   # x:(L,), y_vec:(K,)
        return x, y_vec[self.m]     # scalar target ()


# -----------------------------
# Model: Encoder-Decoder with last-timestep scalar head
# -----------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, p=None, use_bn=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        if p is None: p = k // 2  # "same" for stride=1
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act = act
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ModeScalarED(nn.Module):
    """
    Encoder-Decoder (down x2 three times, up x2 three times),
    then a linear head on the LAST timestep feature -> scalar prediction for one mode.
    """
    def __init__(self, in_ch=1, base=64, signal_len=1024):
        super().__init__()
        # Encoder: L -> L/2 -> L/4 -> L/8
        self.enc1 = ConvBlock1D(in_ch,   base,   k=7, s=1)  # L
        self.enc2 = ConvBlock1D(base,    base*2, k=7, s=2)  # L/2
        self.enc3 = ConvBlock1D(base*2,  base*4, k=5, s=2)  # L/4
        self.enc4 = ConvBlock1D(base*4,  base*8, k=5, s=2)  # L/8
        # Bottleneck
        self.bott = nn.Sequential(
            ConvBlock1D(base*8, base*8, k=3, s=1),
            ConvBlock1D(base*8, base*8, k=3, s=1),
        )
        # Decoder: L/8 -> L/4 -> L/2 -> L
        self.up1  = nn.ConvTranspose1d(base*8, base*4, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock1D(base*4, base*4, k=5, s=1)
        self.up2  = nn.ConvTranspose1d(base*4, base*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock1D(base*2, base*2, k=5, s=1)
        self.up3  = nn.ConvTranspose1d(base*2, base,   kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock1D(base,   base,   k=7, s=1)

        # Last-timestep scalar head
        self.head = nn.Linear(base, 1)

        # Init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.1)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B,1,L)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        h  = self.bott(e4)
        d1 = self.dec1(self.up1(h))
        d2 = self.dec2(self.up2(d1))
        d3 = self.dec3(self.up3(d2))     # (B, base, L)
        last_feat = d3[:, :, -1]         # (B, base)
        y = self.head(last_feat).squeeze(-1)  # (B,) scalar per sample
        return y


# -----------------------------
# Training utilities
# -----------------------------
def train_single_mode(model, loader, device, epochs=5, lr=1e-3, wd=1e-4, print_every=20):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Huber is smoother than L1 but still raw-units; use L1 if you prefer
    loss_fn = nn.HuberLoss(delta=1.0)

    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        t0 = time.time()
        for i, (x, y_scalar) in enumerate(loader, 1):
            x = x.to(device)                 # (B, L), raw
            y = y_scalar.to(device)          # (B,), raw
            print(i)
            print(x.unsqueeze(1))

            opt.zero_grad(set_to_none=True)
            pred = model(x.unsqueeze(1))     # (B,)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            if i % print_every == 0:
                print(f"  Epoch {ep} | Batch {i:04d} | Loss: {running/print_every:.6f}")
                running = 0.0
        print(f"  Epoch {ep} done in {time.time()-t0:.1f}s")


# -----------------------------
# Train 12 per-mode models
# -----------------------------
def main():
    # Config
    csv_file    = "VMD_modes_with_residual_2018_2021.csv"
    K           = 12
    window_len  = 1024
    batch_size  = 8
    epochs      = 1
    lr          = 1e-3
    weight_decay= 1e-4
    num_workers = 0
    base_width  = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Info] Device:", device)

    base_ds = RRP2ModesDataset(csv_file=csv_file, K=K, window_size=window_len)

    for mode_idx in range(K):  # 0..11
        print(f"\n=== Training model for Mode_{mode_idx+1} ===")
        ds_m = ModeTargetDataset(base_ds, mode_idx)
        loader = DataLoader(
            ds_m, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, pin_memory=(device == "cuda")
        )

        model = ModeScalarED(in_ch=1, base=base_width, signal_len=window_len)
        train_single_mode(model, loader, device, epochs=epochs, lr=lr, wd=weight_decay, print_every=20)

        # Save each mode model
        ckpt = f"mode{mode_idx+1:02d}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"[Info] Saved {ckpt}")

if __name__ == "__main__":
    csv_file = "VMD_modes_with_residual_2018_2021.csv"
    K = 12
    window_size = 1024

    # Base dataset
    base_ds = RRP2ModesDataset(csv_file, K=K, window_size=window_size)

    # Wrap for each mode
    for mode_idx in range(K):
        ds_m = ModeTargetDataset(base_ds, mode_idx)
        loader = DataLoader(ds_m, batch_size=1, shuffle=False)
        x, y = next(iter(loader))   # first sample
        print(f"Mode_{mode_idx+1}:")
        print("  Input window shape:", x.shape)    # expect torch.Size([1, L])
        print("  First 5 RRP values:", x[0, :5])   # sanity check input
        print("  Target scalar:", y.item())        # should be one number
        print("-"*40)
    # main()
