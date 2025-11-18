#!/usr/bin/env python3
"""
Full training script for SharedRepresentationPredictor with NVMD decomposer
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import your existing NVMD module
from train_nvmd import HybridSpectralNVMD


# ============================================================
# RRP Prediction Dataset
# ============================================================

class RRPWindowDataset(Dataset):
    """
    Dataset for RRP prediction with next-step target.
    Input: window x_raw[t ... t+L-1]
    Target: RRP[t+L]
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 64, rrp_col: str = "RRP"):
        super().__init__()
        self.L = seq_len

        if rrp_col not in df.columns:
            raise ValueError(f"RRP column '{rrp_col}' not in dataframe")

        rrp_np = df[rrp_col].to_numpy(dtype=np.float32)
        self.rrp = torch.from_numpy(rrp_np)  # (T,)

        T = self.rrp.shape[0]
        # Need rrp[t+L] to exist
        self.N = max(0, T - self.L - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        L = self.L
        # window [i, ..., i+L-1]
        x_raw = self.rrp[i:i+L].unsqueeze(0)       # (1, L)
        # next-step RRP at time t+L
        rrp_next = self.rrp[i + L].unsqueeze(0)    # (1,)
        return x_raw, rrp_next


# ============================================================
# Per-mode temporal encoder (improved FixedMultiModeTransformerRRP)
# ============================================================

class FixedMultiModeTransformerRRP(nn.Module):
    """
    Encodes each IMF time series into a per-mode embedding, then applies
    a Transformer over modes (K tokens).
    """
    def __init__(self, K, d_model, n_heads, num_layers, dim_ff, dropout):
        super().__init__()
        self.K = K
        self.d_model = d_model

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Temporal Conv1d to summarize each mode's time series
        # Input for each mode: (B*K, 1, L) -> (B*K, C, L) -> pooled -> (B*K, d_model)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),
        )

        # Positional / mode embedding (K tokens)
        self.pos_encoding = nn.Parameter(torch.randn(1, K, d_model))

        # Transformer encoder over modes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, imfs):
        """
        imfs: (B, K, L) -> (B, K, d_model)
        """
        B, K, L = imfs.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"

        # Flatten modes into batch dimension: (B*K, 1, L)
        x = imfs.view(B * K, 1, L)
        x = self.temporal_conv(x)          # (B*K, d_model, L)
        x = x.mean(dim=-1)                 # global average pool over time -> (B*K, d_model)

        # Reshape back to (B, K, d_model)
        imf_features = x.view(B, K, self.d_model)

        # Add learnable mode positions
        imf_features = imf_features + self.pos_encoding  # (1,K,d_model) broadcast

        transformed = self.transformer(imf_features)      # (B, K, d_model)
        return self.output_proj(transformed)              # (B, K, d_model)


# ============================================================
# Shared Representation Predictor
# ============================================================

class SharedRepresentationPredictor(nn.Module):
    """
    Uses:
      - IMF-based per-window features from NVMD
      - Global spectral priors from decomposer (masks, center freqs, bandwidths)
    to build spectral-aware attention over modes and predict RRP_next.
    """
    def __init__(self, decomposer, d_model, n_heads, num_layers, dim_ff, dropout):
        super().__init__()
        self.decomposer = decomposer
        self.K = decomposer.K
        self.L = decomposer.L
        self.F = decomposer.spectral.F

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model

        # Encode spectral masks (K,F) -> (K,d_model)
        self.freq_embedding = nn.Linear(self.F, d_model)

        # Encode center frequencies ω_k -> (K, d_model//2)
        self.center_freq_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
        )

        # Encode bandwidths σ_k -> (K, d_model//4)
        self.bandwidth_encoder = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model // 4),
        )

        # Mode-wise temporal encoder + Transformer over modes
        self.mode_aware_transformer = FixedMultiModeTransformerRRP(
            self.K, d_model, n_heads, num_layers, dim_ff, dropout
        )

        # Combined feature dimension (before projection)
        combined_dim_exact = d_model * 2 + d_model // 2 + d_model // 4
        self.combined_dim_exact = combined_dim_exact

        # Ensure embed_dim for MultiheadAttention is divisible by n_heads
        if combined_dim_exact % n_heads != 0:
            attn_dim = (combined_dim_exact // n_heads) * n_heads
            print(f"[SharedRepresentationPredictor] Projecting combined features "
                  f"from {combined_dim_exact} to {attn_dim} to fit n_heads={n_heads}")
            self.combine_proj = nn.Linear(combined_dim_exact, attn_dim)
        else:
            attn_dim = combined_dim_exact
            self.combine_proj = None

        self.attn_dim = attn_dim

        # Spectral-aware attention over modes
        self.spectral_attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection: pooled over modes -> scalar
        self.output_proj = nn.Sequential(
            nn.Linear(self.attn_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, imfs):
        """
        imfs: (B, K, L) from NVMD (refined IMFs)
        """
        B, K, L = imfs.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"
        assert L == self.L, f"Expected L={self.L}, got {L}"

        device = imfs.device

        # 1. Global spectral priors from decomposer (no grad here)
        with torch.no_grad():
            spectral_masks = self.decomposer.spectral.masks().to(device)      # (K, F)
            center_freqs = self.decomposer.spectral.omega.to(device)          # (K,)
            bandwidths = self.decomposer.spectral.log_sigma.exp().to(device)  # (K,)

        # 2. Frequency-domain mask features (same for all windows, per mode)
        freq_features = self.freq_embedding(spectral_masks)  # (K, d_model)
        freq_features = freq_features.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)

        # 3. Center frequency encoding
        center_features = self.center_freq_encoder(
            center_freqs.unsqueeze(-1)  # (K, 1)
        ).unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model//2)

        # 4. Bandwidth encoding
        bandwidth_features = self.bandwidth_encoder(
            bandwidths.unsqueeze(-1)  # (K, 1)
        ).unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model//4)

        # 5. IMF temporal features (per window, per mode)
        imf_features = self.mode_aware_transformer(imfs)  # (B, K, d_model)

        # 6. Combine all features
        combined_features = torch.cat(
            [imf_features, freq_features, center_features, bandwidth_features],
            dim=-1,
        )  # (B, K, combined_dim_exact)

        # 7. Optional projection to attn_dim
        if self.combine_proj is not None:
            combined_features = self.combine_proj(combined_features)  # (B, K, attn_dim)

        # 8. Spectral-aware attention over modes (self-attention)
        attended_features, _ = self.spectral_attention(
            combined_features, combined_features, combined_features
        )  # (B, K, attn_dim)

        # 9. Pool over modes and predict scalar
        pooled = attended_features.mean(dim=1)  # (B, attn_dim)
        output = self.output_proj(pooled)       # (B, 1)
        return output


# ============================================================
# Training / Evaluation
# ============================================================

def run_epoch(
    decomposer: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer=None,
    freeze_decomposer: bool = True,
    max_grad_norm: float = 10.0,
    decomp_grad_scale: float = 1.0,
    w_decomp_rrp: float = 0.0,
    w_decomp_smooth: float = 0.0,
    w_decomp_ortho: float = 0.0,
):
    """
    One training/evaluation epoch.
    If optimizer is None -> evaluation (no gradients).
    """
    is_train = optimizer is not None

    if is_train:
        if freeze_decomposer:
            decomposer.eval()
            for p in decomposer.parameters():
                p.requires_grad = False
        else:
            decomposer.train()
            for p in decomposer.parameters():
                p.requires_grad = True
        predictor.train()
    else:
        # Evaluation: always freeze decomposer & predictor
        decomposer.eval()
        predictor.eval()
        for p in decomposer.parameters():
            p.requires_grad = False

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    for x_raw, rrp_next in loader:
        x_raw = x_raw.to(device)        # (B,1,L)
        rrp_next = rrp_next.to(device)  # (B,1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Decide grad context for decomposer
        if is_train and not freeze_decomposer:
            ctx = torch.enable_grad()
        else:
            ctx = torch.no_grad()

        # Forward through NVMD decomposer
        with ctx:
            imfs_ref, recon_ref, imfs_lin, recon_lin = decomposer(x_raw)

        # If decomposer is frozen, detach IMFs completely
        if freeze_decomposer or not is_train:
            imfs_ref = imfs_ref.detach()

        # Forward through predictor
        rrp_next_hat = predictor(imfs_ref)   # (B, 1)

        # Prediction metrics
        mse = F.mse_loss(rrp_next_hat, rrp_next)
        mae = F.l1_loss(rrp_next_hat, rrp_next)

        if is_train:
            loss = mse

            # Optional decomposer-side regularization in joint training
            if not freeze_decomposer:
                loss_rrp_recon = F.l1_loss(recon_ref, x_raw)
                loss_smooth = decomposer.spectral.spectral_smoothness_loss()
                loss_ortho = decomposer.spectral.orthogonality_loss()

                loss = (
                    loss
                    + w_decomp_rrp * loss_rrp_recon
                    + w_decomp_smooth * loss_smooth
                    + w_decomp_ortho * loss_ortho
                )

            loss.backward()

            # Scale decomposer gradients if trainable
            if not freeze_decomposer and decomp_grad_scale != 1.0:
                with torch.no_grad():
                    for p in decomposer.parameters():
                        if p.grad is not None:
                            p.grad.mul_(decomp_grad_scale)

            # Gradient clipping
            if freeze_decomposer:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(decomposer.parameters()) + list(predictor.parameters()),
                    max_grad_norm,
                )

            optimizer.step()

        bs = x_raw.size(0)
        n_samples += bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs

    denom = max(n_samples, 1)
    return total_mse / denom, total_mae / denom


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(decomposer, predictor, args, epoch, val_mae, filename, stage):
    """Save training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "decomposer_state": decomposer.state_dict(),
            "predictor_state": predictor.state_dict(),
            "val_mae": val_mae,
            "args": vars(args),
            "stage": stage,
            "timestamp": time.time(),
        },
        filename,
    )
    print(f"  → Saved checkpoint: {filename} (val MAE: {val_mae:.4f}, stage={stage})")


def check_data_distribution(train_csv, val_csv, rrp_col="RRP"):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    train_rrp = df_train[rrp_col]
    val_rrp = df_val[rrp_col]

    print(f"Train RRP - Mean: {train_rrp.mean():.2f}, Std: {train_rrp.std():.2f}")
    print(f"Val   RRP - Mean: {val_rrp.mean():.2f}, Std: {val_rrp.std():.2f}")
    print(f"Train min/max: {train_rrp.min():.2f}/{train_rrp.max():.2f}")
    print(f"Val   min/max: {val_rrp.min():.2f}/{val_rrp.max():.2f}")

    if abs(train_rrp.mean() - val_rrp.mean()) > train_rrp.std() * 0.5:
        print("WARNING: Significant distribution shift between train and val!")


# ============================================================
# Main Training Script
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Shared Representation Predictor with NVMD"
    )

    # Data
    parser.add_argument("--train-csv", type=str, required=True,
                        help="Training CSV file")
    parser.add_argument("--val-csv", type=str, required=True,
                        help="Validation CSV file")
    parser.add_argument("--rrp-col", type=str, default="RRP",
                        help="RRP column name")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length")

    # NVMD decomposer
    parser.add_argument("--K", type=int, default=13,
                        help="Number of modes")
    parser.add_argument("--decomposer-ckpt", type=str, required=True,
                        help="Pretrained decomposer checkpoint")

    # Shared predictor hyperparameters
    parser.add_argument("--d-model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of transformer layers")
    parser.add_argument("--dim-ff", type=int, default=256,
                        help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--warmup-epochs", type=int, default=20,
                        help="Epochs with frozen decomposer")
    parser.add_argument("--joint-epochs", type=int, default=100,
                        help="Epochs of joint training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--max-grad-norm", type=float, default=10.0,
                        help="Gradient clipping norm")

    # Decomposer training parameters (joint stage)
    parser.add_argument("--decomp-grad-scale", type=float, default=0.5,
                        help="Decomposer gradient scaling (joint stage)")
    parser.add_argument("--w-decomp-rrp", type=float, default=0.1,
                        help="Weight for RRP reconstruction loss (joint)")
    parser.add_argument("--w-decomp-smooth", type=float, default=0.01,
                        help="Weight for spectral smoothness loss (joint)")
    parser.add_argument("--w-decomp-ortho", type=float, default=0.01,
                        help="Weight for orthogonality loss (joint)")

    # I/O
    parser.add_argument("--out-dir", type=str, default="./checkpoints",
                        help="Output directory")
    parser.add_argument("--experiment", type=str, default="shared_rep_predictor",
                        help="Experiment name")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, f"{args.experiment}.pt")

    # Check distribution
    print("\nChecking train/val RRP distribution:")
    check_data_distribution(args.train_csv, args.val_csv, rrp_col=args.rrp_col)

    # Load data
    print("\nLoading data...")
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)

    tr_ds = RRPWindowDataset(df_tr, seq_len=args.seq_len, rrp_col=args.rrp_col)
    va_ds = RRPWindowDataset(df_va, seq_len=args.seq_len, rrp_col=args.rrp_col)

    pin = device == "cuda"
    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    print(f"Train windows: {len(tr_ds)}, Val windows: {len(va_ds)}")

    # Load pretrained decomposer
    print("\nLoading pretrained decomposer...")
    decomposer = HybridSpectralNVMD(K=args.K, signal_len=args.seq_len).to(device)

    dec_ckpt = torch.load(args.decomposer_ckpt, map_location="cpu")
    if "model_state" in dec_ckpt:
        dec_state = dec_ckpt["model_state"]
    elif "decomposer_state" in dec_ckpt:
        dec_state = dec_ckpt["decomposer_state"]
    else:
        dec_state = dec_ckpt

    missing_d, unexpected_d = decomposer.load_state_dict(dec_state, strict=False)
    print(f"Loaded decomposer: missing={len(missing_d)}, unexpected={len(unexpected_d)}")

    # Initialize predictor
    print("\nInitializing shared representation predictor...")
    predictor = SharedRepresentationPredictor(
        decomposer=decomposer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    best_val_mae = float("inf")

    # ========================================================
    # Stage 1: Warmup (frozen decomposer)
    # ========================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Warmup (frozen decomposer, train predictor only)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.warmup_epochs + 1):
        tr_mse, tr_mae = run_epoch(
            decomposer,
            predictor,
            tr_dl,
            device,
            optimizer=optimizer,
            freeze_decomposer=True,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=1.0,
            w_decomp_rrp=0.0,
            w_decomp_smooth=0.0,
            w_decomp_ortho=0.0,
        )

        va_mse, va_mae = run_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
            optimizer=None,  # eval
            freeze_decomposer=True,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=1.0,
            w_decomp_rrp=0.0,
            w_decomp_smooth=0.0,
            w_decomp_ortho=0.0,
        )

        print(
            f"[Warmup {epoch:03d}/{args.warmup_epochs}] "
            f"Train MSE: {tr_mse:.4f}, MAE: {tr_mae:.4f} | "
            f"Val MSE: {va_mse:.4f}, MAE: {va_mae:.4f}"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            save_checkpoint(
                decomposer,
                predictor,
                args,
                epoch,
                best_val_mae,
                checkpoint_path,
                stage="warmup",
            )

    # ========================================================
    # Stage 2: Joint training (decomposer + predictor)
    # ========================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Joint training (decomposer + predictor)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        list(decomposer.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.joint_epochs + 1):
        tr_mse, tr_mae = run_epoch(
            decomposer,
            predictor,
            tr_dl,
            device,
            optimizer=optimizer,
            freeze_decomposer=False,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=args.decomp_grad_scale,
            w_decomp_rrp=args.w_decomp_rrp,
            w_decomp_smooth=args.w_decomp_smooth,
            w_decomp_ortho=args.w_decomp_ortho,
        )

        # For validation, we run with decomposer frozen (no grad) for simplicity
        va_mse, va_mae = run_epoch(
            decomposer,
            predictor,
            va_dl,
            device,
            optimizer=None,
            freeze_decomposer=True,
            max_grad_norm=args.max_grad_norm,
            decomp_grad_scale=1.0,
            w_decomp_rrp=0.0,
            w_decomp_smooth=0.0,
            w_decomp_ortho=0.0,
        )

        print(
            f"[Joint {epoch:03d}/{args.joint_epochs}] "
            f"Train MSE: {tr_mse:.4f}, MAE: {tr_mae:.4f} | "
            f"Val MSE: {va_mse:.4f}, MAE: {va_mae:.4f}"
        )

        # Optional: inspect decomposer spectral statistics
        if epoch % 10 == 0:
            with torch.no_grad():
                centers = decomposer.spectral.omega.detach().cpu().numpy()
                bandwidths = decomposer.spectral.log_sigma.exp().detach().cpu().numpy()
            print("  Centers (rad):", centers)
            print("  Bandwidths:", bandwidths)

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            save_checkpoint(
                decomposer,
                predictor,
                args,
                epoch,
                best_val_mae,
                checkpoint_path,
                stage="joint",
            )

    print(f"\nTraining completed! Best validation MAE: {best_val_mae:.4f}")
    print(f"Final model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
