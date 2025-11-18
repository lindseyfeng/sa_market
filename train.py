#!/usr/bin/env python3
"""
Joint NVMD + Transformer with *structured* coupling:

- Dataset:
    x_raw:      (1, L)   raw RRP window [t, ..., t+L-1]
    imfs_true:  (K, L)   VMD IMFs over same window (for supervision)
    rrp_next:   (1,)     RRP at time t+L

- Model: NVMDTransformerCross
    x_raw --NVMD--> imfs_ref (B,K,L), recon_ref (B,1,L), priors
      |                     |
      |                     v
      |           mode branch Transformer (time × K)
      v
    raw branch Transformer (time × 1)
      |
      +-- cross-attention (raw queries, mode keys/values) -->
           fused rep -> RRP head

- Loss:
    loss = w_pred   * MSE(rrp_hat, rrp_next)
         + w_imf    * relRMSE(imfs_ref, imfs_true)
         + w_rrp    * L1(recon_ref, x_raw)
         + w_smooth * smooth_loss
         + w_ortho  * ortho_loss
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
#   NVMD pieces (same as your HybridSpectralNVMD, inlined)
# ============================================================

class SpectralDecomposer(nn.Module):
    def __init__(self, K: int, signal_len: int):
        super().__init__()
        self.K = K
        self.L = signal_len
        self.F = signal_len // 2 + 1
        self.logits = nn.Parameter(torch.zeros(K, self.F))  # (K,F)

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert L == self.L, f"Expected signal_len={self.L}, got {L}"

        Xf = torch.fft.rfft(x, dim=-1)  # (B,1,F), complex

        masks = F.softmax(self.logits, dim=0)  # (K,F)
        masks_exp = masks.unsqueeze(0).expand(B, -1, -1)  # (B,K,F)

        Xf_exp = Xf.expand(-1, self.K, -1)                # (B,K,F)
        Xf_modes = Xf_exp * masks_exp.to(Xf.dtype)        # (B,K,F)

        imfs_lin = torch.fft.irfft(Xf_modes, n=self.L, dim=-1)  # (B,K,L)
        recon_lin = imfs_lin.sum(dim=1, keepdim=True)           # (B,1,L)
        return imfs_lin, recon_lin

    def spectral_smoothness_loss(self):
        masks = F.softmax(self.logits, dim=0)  # (K,F)
        diff = masks[:, 1:] - masks[:, :-1]    # (K,F-1)
        return (diff ** 2).mean()

    def orthogonality_loss(self):
        masks = F.softmax(self.logits, dim=0)  # (K,F)
        K, Ffreq = masks.shape
        loss = 0.0
        cnt = 0
        for i in range(K):
            mi = masks[i]
            for j in range(i+1, K):
                mj = masks[j]
                num = (mi * mj).sum()
                den = mi.norm() * mj.norm() + 1e-8
                loss = loss + (num / den)
                cnt += 1
        if cnt > 0:
            loss = loss / cnt
        return loss


class ModewiseRefiner(nn.Module):
    def __init__(self, K: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=K,
            out_channels=K,
            kernel_size=kernel_size,
            padding=padding,
            groups=K,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            in_channels=K,
            out_channels=K,
            kernel_size=kernel_size,
            padding=padding,
            groups=K,
            bias=True,
        )
        self.act = nn.GELU()
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, imfs_lin: torch.Tensor) -> torch.Tensor:
        x = imfs_lin
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


class HybridSpectralNVMD(nn.Module):
    def __init__(self, K: int = 13, signal_len: int = 256):
        super().__init__()
        self.K = K
        self.L = signal_len
        self.spectral = SpectralDecomposer(K=K, signal_len=signal_len)
        self.refiner  = ModewiseRefiner(K=K)

    def forward(self, x_raw: torch.Tensor):
        imfs_lin, recon_lin = self.spectral(x_raw)           # (B,K,L), (B,1,L)
        imfs_refined = self.refiner(imfs_lin)                # (B,K,L)
        recon_refined = imfs_refined.sum(dim=1, keepdim=True)
        return imfs_refined, recon_refined, imfs_lin, recon_lin


# ============================================================
#                     Dataset (raw + IMFs)
# ============================================================

class JointDecompPredictDataset(Dataset):
    """
    For each start index i:

      x_raw:      (1, L)   raw RRP window [i, ..., i+L-1]
      imfs_true:  (K, L)   VMD IMFs over same window
      rrp_next:   (1,)     raw RRP at time i+L
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 256,
        rrp_col: str = "RRP",
        K: int = 13,
    ):
        super().__init__()
        self.L = seq_len
        self.rrp_col = rrp_col
        self.K = K

        if rrp_col not in df.columns:
            raise ValueError(f"RRP column '{rrp_col}' not in dataframe")

        mode_cols = [f"Mode_{i}" for i in range(1, K)] + ["Residual"]
        for c in mode_cols:
            if c not in df.columns:
                raise ValueError(f"Missing mode column '{c}' in dataframe")
        self.mode_cols = mode_cols

        rrp = df[rrp_col].to_numpy(dtype=np.float32)        # (T,)
        imfs = df[mode_cols].to_numpy(dtype=np.float32)     # (T,K)

        self.rrp = torch.from_numpy(rrp)                    # (T,)
        self.imfs = torch.from_numpy(imfs).transpose(0, 1)  # (K,T)

        T = self.rrp.shape[0]
        # need rrp[i+L] to exist
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        x_raw = self.rrp[i:i+L].unsqueeze(0)     # (1,L)
        imfs_true = self.imfs[:, i:i+L]          # (K,L)
        rrp_next = self.rrp[i+L].unsqueeze(0)    # (1,)
        return x_raw, imfs_true, rrp_next


# ============================================================
#                  Positional Encoding helper
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (L,d)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1,L,d)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


# ============================================================
#        NVMD + dual-branch Transformers + cross-attention
# ============================================================

class NVMDTransformerCross(nn.Module):
    """
    More structured combination of NVMD & Transformer:

      1) NVMD decomposes x_raw:
            x_raw (B,1,L) -> imfs_ref (B,K,L), recon_ref (B,1,L)
      2) Raw branch:
            raw_seq = x_raw^T -> (B,L,1) -> proj -> Transformer
      3) Mode branch:
            modes_seq = imfs_ref^T -> (B,L,K) -> proj -> Transformer
      4) Cross attention:
            raw tokens query mode tokens (Q=raw, K/V=mode)
      5) Head:
            concat(raw_last, cross_last) -> MLP -> RRP next prediction.
    """
    def __init__(
        self,
        K: int = 13,
        seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers_raw: int = 2,
        num_layers_mode: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.seq_len = seq_len

        # NVMD decomposer
        self.decomposer = HybridSpectralNVMD(K=K, signal_len=seq_len)

        # Raw branch: 1-channel → d_model
        self.raw_proj = nn.Linear(1, d_model)
        self.raw_pos  = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        raw_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.raw_encoder = nn.TransformerEncoder(raw_layer, num_layers=num_layers_raw)

        # Mode branch: K-channels → d_model
        self.mode_proj = nn.Linear(K, d_model)
        self.mode_pos  = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        mode_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.mode_encoder = nn.TransformerEncoder(mode_layer, num_layers=num_layers_mode)

        # Cross-attention: raw queries, mode keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # Head on [raw_last, cross_last]
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_raw: torch.Tensor, return_details: bool = False):
        """
        x_raw: (B,1,L)

        Return:
          if return_details:
             (rrp_hat, imfs_ref, recon_ref, smooth_loss, ortho_loss)
          else:
             rrp_hat
        """
        B, C, L = x_raw.shape
        assert C == 1
        assert L == self.seq_len

        # ---- NVMD ----
        imfs_ref, recon_ref, imfs_lin, recon_lin = self.decomposer(x_raw)  # (B,K,L),(B,1,L),...

        # ---- Raw branch ----
        raw_seq = x_raw.permute(0, 2, 1)           # (B,L,1)
        raw_h = self.raw_proj(raw_seq)             # (B,L,d)
        raw_h = self.raw_pos(raw_h)
        raw_h = self.raw_encoder(raw_h)            # (B,L,d)

        # ---- Mode branch ----
        mode_seq = imfs_ref.permute(0, 2, 1)       # (B,L,K)
        mode_h = self.mode_proj(mode_seq)          # (B,L,d)
        mode_h = self.mode_pos(mode_h)
        mode_h = self.mode_encoder(mode_h)         # (B,L,d)

        # ---- Cross-attention: raw queries, mode keys/values ----
        # MultiheadAttention expects (B,L,d) for batch_first=True
        cross_out, _ = self.cross_attn(
            query=raw_h,   # (B,L,d)
            key=mode_h,    # (B,L,d)
            value=mode_h,  # (B,L,d)
        )                 # (B,L,d)

        # take last time-step
        raw_last   = raw_h[:, -1, :]        # (B,d)
        cross_last = cross_out[:, -1, :]    # (B,d)

        fused = torch.cat([raw_last, cross_last], dim=-1)  # (B,2d)
        rrp_hat = self.head(fused)                         # (B,1)

        # priors from spectral masks
        smooth_loss = self.decomposer.spectral.spectral_smoothness_loss()
        ortho_loss  = self.decomposer.spectral.orthogonality_loss()

        if return_details:
            return rrp_hat, imfs_ref, recon_ref, smooth_loss, ortho_loss
        else:
            return rrp_hat


# ============================================================
#                     Training / Eval Epoch
# ============================================================

def run_epoch(
    model: NVMDTransformerCross,
    loader: DataLoader,
    device: str,
    optimizer=None,
    w_pred: float = 1.0,
    w_imf: float = 0.5,
    w_rrp: float = 0.1,
    w_smooth: float = 0.01,
    w_ortho: float = 0.01,
    max_grad_norm: float | None = 10.0,
):
    """
    Joint training/eval epoch.

    loss = w_pred   * MSE(rrp_hat, rrp_next)
         + w_imf    * relRMSE(imfs_ref, imfs_true)
         + w_rrp    * L1(recon_ref, x_raw)
         + w_smooth * smooth_loss
         + w_ortho  * ortho_loss
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0
    eps = 1e-8

    for x_raw, imfs_true, rrp_next in loader:
        x_raw     = x_raw.to(device)      # (B,1,L)
        imfs_true = imfs_true.to(device)  # (B,K,L)
        rrp_next  = rrp_next.to(device)   # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        rrp_hat, imfs_ref, recon_ref, smooth_loss, ortho_loss = model(
            x_raw,
            return_details=True,
        )

        mse = F.mse_loss(rrp_hat, rrp_next)
        mae = F.l1_loss(rrp_hat, rrp_next)

        # IMF relRMSE
        delta = imfs_ref - imfs_true              # (B,K,L)
        num = (delta ** 2).sum(dim=(1, 2))        # (B,)
        den = (imfs_true ** 2).sum(dim=(1, 2)) + eps
        rel_rmse = torch.sqrt(num / den)          # (B,)
        loss_imf = rel_rmse.mean()

        # RRP recon
        loss_rrp = F.l1_loss(recon_ref, x_raw)

        loss = (
            w_pred   * mse
          + w_imf    * loss_imf
          + w_rrp    * loss_rrp
          + w_smooth * smooth_loss
          + w_ortho  * ortho_loss
        )

        if is_train:
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
#                          Utilities
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
#                            MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train-csv", type=str, default="VMD_modes_with_residual_2018_2021.csv")
    ap.add_argument("--val-csv",   type=str, default="VMD_modes_with_residual_2021_2022.csv")
    ap.add_argument("--rrp-col",   type=str, default="RRP")
    ap.add_argument("--seq-len",   type=int, default=256)
    ap.add_argument("--K",         type=int, default=13)

    # model hparams
    ap.add_argument("--d-model",        type=int, default=128)
    ap.add_argument("--n-heads",        type=int, default=4)
    ap.add_argument("--num-layers-raw", type=int, default=2)
    ap.add_argument("--num-layers-mode",type=int, default=2)
    ap.add_argument("--dim-ff",         type=int, default=256)
    ap.add_argument("--dropout",        type=float, default=0.1)

    # training
    ap.add_argument("--batch",         type=int,   default=256)
    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight-decay",  type=float, default=1e-2)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--num-workers",   type=int,   default=0)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)

    # loss weights
    ap.add_argument("--w-pred",   type=float, default=1.0)
    ap.add_argument("--w-imf",    type=float, default=0.5)
    ap.add_argument("--w-rrp",    type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=0.01)
    ap.add_argument("--w-ortho",  type=float, default=0.01)

    # I/O
    ap.add_argument("--out", type=str, default="nvmd_transformer_cross.pt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # data
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = JointDecompPredictDataset(
        df_tr,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        K=args.K,
    )
    va_ds = JointDecompPredictDataset(
        df_va,
        seq_len=args.seq_len,
        rrp_col=args.rrp_col,
        K=args.K,
    )

    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train windows: {len(tr_ds)}, Val windows: {len(va_ds)}")

    # model
    model = NVMDTransformerCross(
        K=args.K,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers_raw=args.num_layers_raw,
        num_layers_mode=args.num_layers_mode,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")

    for ep in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(
            model,
            tr_dl,
            device,
            optimizer=optimizer,
            w_pred=args.w_pred,
            w_imf=args.w_imf,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )
        va_mse, va_mae = run_epoch(
            model,
            va_dl,
            device,
            optimizer=None,
            w_pred=args.w_pred,
            w_imf=args.w_imf,
            w_rrp=args.w_rrp,
            w_smooth=args.w_smooth,
            w_ortho=args.w_ortho,
            max_grad_norm=args.max_grad_norm,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | "
            f"val MSE={va_mse:.4f} MAE={va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(
                {
                    "epoch": ep,
                    "val_mae": best_val_mae,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"  → Saved new best checkpoint with val MAE={best_val_mae:.4f} to {args.out}")


if __name__ == "__main__":
    main()
