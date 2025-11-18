import torch
import torch.nn as nn
import torch.nn.functional as F


class NVMDModeEncoderTail(nn.Module):
    """
    Per-mode temporal encoder that focuses on the tail of each IMF:

      imfs_ref: (B, K, L)  --per-mode conv-->  (B, K, d_model)

    We:
      - apply Conv1d over time for each mode (via B*K trick),
      - keep only the last `last_k` time steps,
      - average over those last_k steps.

    This keeps recent dynamics instead of global mean over the whole window.
    """
    def __init__(self, K, L, d_model, last_k: int = 8):
        super().__init__()
        self.K = K
        self.L = L
        self.d_model = d_model
        self.last_k = min(last_k, L)

        self.temporal = nn.Sequential(
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

    def forward(self, imfs_ref: torch.Tensor) -> torch.Tensor:
        """
        imfs_ref: (B, K, L)
        returns:  (B, K, d_model)
        """
        B, K, L = imfs_ref.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"
        assert L == self.L, f"Expected L={self.L}, got {L}"

        # Treat each mode as a separate 1D signal
        x = imfs_ref.view(B * K, 1, L)           # (B*K, 1, L)
        x = self.temporal(x)                     # (B*K, d_model, L)

        # Focus on last_k timesteps
        if self.last_k < L:
            x_tail = x[:, :, -self.last_k:]      # (B*K, d_model, last_k)
        else:
            x_tail = x                           # (B*K, d_model, L)

        # Average only over the tail
        x_tail = x_tail.mean(dim=-1)             # (B*K, d_model)
        x_tail = x_tail.view(B, K, self.d_model) # (B, K, d_model)
        return x_tail

class NVMDTransformerPredictor(nn.Module):
    """
      - Input: imfs_ref (B, K, L) from HybridSpectralNVMD
      - Encoder:
          * Tail-focused temporal encoder per mode -> (B, K, d_model)
          * Spectral priors (masks, μ_k, σ_k) -> per-mode bias embedding (K, d_model)
          * Add priors as bias: mode_feats += prior_emb
      - Transformer over K mode tokens
      - Pool over modes -> scalar prediction (RRP_next)

    Usage:
      predictor = NVMDTransformerPredictorV2(decomposer, d_model, ...)
      rrp_next_hat = predictor(imfs_ref)
    """
    def __init__(
        self,
        decomposer,          # HybridSpectralNVMD instance
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        last_k: int = 16,
    ):
        super().__init__()
        self.decomposer = decomposer
        self.K = decomposer.K
        self.L = decomposer.L
        self.F = decomposer.spectral.F

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model

        # 1) Tail-focused per-mode encoder (A)
        self.mode_encoder = NVMDModeEncoderTail(
            K=self.K,
            L=self.L,
            d_model=d_model,
            last_k=last_k,
        )

        # 2) Spectral priors → per-mode bias embedding (B)
        #    We'll produce a (K, d_model) tensor and add to mode_feats.

        # Masks: (K, F) -> (K, d_model)
        self.freq_proj = nn.Linear(self.F, d_model)

        # Center frequencies μ_k: (K, 1) -> (K, d_model)
        self.center_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Bandwidth σ_k: (K, 1) -> (K, d_model)
        self.bandwidth_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Optional learned mode index embedding (since K is small & fixed)
        self.mode_index_embed = nn.Embedding(self.K, d_model)

        # 3) Transformer over modes (K tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 4) Output head: pooled (over modes) -> scalar
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, imfs_ref: torch.Tensor) -> torch.Tensor:
        """
        imfs_ref: (B, K, L)
        returns:  rrp_next_hat (B, 1)
        """
        B, K, L = imfs_ref.shape
        assert K == self.K and L == self.L, \
            f"Expected (K={self.K}, L={self.L}), got (K={K}, L={L})"

        device = imfs_ref.device

        # 1) Tail-focused temporal features per mode
        mode_feats = self.mode_encoder(imfs_ref)  # (B, K, d_model)

        # 2) Spectral priors from decomposer (global, typically no grad)
        with torch.no_grad():
            masks = self.decomposer.spectral.masks().to(device)           # (K, F)
            center, bandwidth = self.decomposer.spectral.mask_stats()     # (K,),(K,)
            center = center.to(device)
            bandwidth = bandwidth.to(device)

        # (K, F) -> (K, d_model)
        freq_emb = self.freq_proj(masks)                      # (K, d_model)

        # (K, 1) -> (K, d_model)
        center_emb = self.center_proj(center.unsqueeze(-1))   # (K, d_model)
        bw_emb     = self.bandwidth_proj(bandwidth.unsqueeze(-1))  # (K, d_model)

        # Mode index embedding (0..K-1)
        mode_indices = torch.arange(self.K, device=device)
        mode_idx_emb = self.mode_index_embed(mode_indices)    # (K, d_model)

        # Combine priors into a single per-mode bias embedding
        prior_emb = freq_emb + center_emb + bw_emb + mode_idx_emb  # (K, d_model)

        # Broadcast to batch and add as bias (B)
        prior_emb = prior_emb.unsqueeze(0).expand(B, -1, -1)       # (B, K, d_model)
        mode_feats = mode_feats + prior_emb                        # (B, K, d_model)

        # 3) Transformer over modes
        z = self.transformer(mode_feats)                           # (B, K, d_model)

        # 4) Pool over modes and predict
        pooled = z.mean(dim=1)                                     # (B, d_model)
        out = self.output_head(pooled)                             # (B, 1)
        return out
