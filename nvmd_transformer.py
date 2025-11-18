import torch
import torch.nn as nn
import torch.nn.functional as F


class NVMDModeEncoder(nn.Module):
    """
    Per-mode temporal encoder: imfs_ref (B,K,L) -> mode embeddings (B,K,d_model)
    using Conv1d + GELU + global pooling.
    """
    def __init__(self, K, L, d_model):
        super().__init__()
        self.K = K
        self.L = L
        self.d_model = d_model

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

    def forward(self, imfs_ref):
        """
        imfs_ref: (B,K,L)
        returns:  (B,K,d_model)
        """
        B, K, L = imfs_ref.shape
        assert K == self.K and L == self.L

        # (B,K,L) -> (B*K,1,L)
        x = imfs_ref.view(B * K, 1, L)
        x = self.temporal(x)           # (B*K,d_model,L)
        x = x.mean(dim=-1)             # global pool over time -> (B*K,d_model)
        x = x.view(B, K, self.d_model) # (B,K,d_model)
        return x


class NVMDTransformerPredictor(nn.Module):
    """
    NVMD-aware predictor:

      - Uses NVMD decomposer's spectral masks & stats as per-mode priors.
      - Encodes each mode's time series via NVMDModeEncoder.
      - Concatenates temporal + spectral features per mode.
      - Applies Transformer over K mode tokens.
      - Pools and predicts RRP_next.

    API: rrp_next_hat = predictor(imfs_ref)
         where imfs_ref is (B,K,L) and decomposer is stored inside.
    """
    def __init__(
        self,
        decomposer,     # HybridSpectralNVMD instance
        d_model=128,
        n_heads=4,
        num_layers=3,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.decomposer = decomposer
        self.K = decomposer.K
        self.L = decomposer.L
        self.F = decomposer.spectral.F

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model

        # Temporal encoder for imfs_ref
        self.mode_encoder = NVMDModeEncoder(self.K, self.L, d_model)

        # Spectral mask embedding: (K,F) -> (K,d_model)
        self.freq_embedding = nn.Linear(self.F, d_model)

        # Center frequency μ_k -> (K, d_model//2)
        self.center_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
        )

        # Bandwidth σ_k -> (K, d_model//4)
        self.bandwidth_encoder = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model // 4),
        )

        # Total feature dim before projection
        combined_dim_exact = d_model * 2 + d_model // 2 + d_model // 4
        if combined_dim_exact % n_heads != 0:
            attn_dim = (combined_dim_exact // n_heads) * n_heads
            print(
                f"[NVMDTransformerPredictor] Projecting from {combined_dim_exact} "
                f"to {attn_dim} to fit n_heads={n_heads}"
            )
            self.combine_proj = nn.Linear(combined_dim_exact, attn_dim)
        else:
            attn_dim = combined_dim_exact
            self.combine_proj = None

        self.attn_dim = attn_dim

        # Mode-wise Transformer over K tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: pooled -> scalar
        self.output_head = nn.Sequential(
            nn.Linear(attn_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, imfs_ref):
        """
        imfs_ref: (B,K,L) from decomposer
        """
        B, K, L = imfs_ref.shape
        assert K == self.K and L == self.L
        device = imfs_ref.device

        # 1) Temporal mode features
        mode_feats = self.mode_encoder(imfs_ref)  # (B,K,d_model)

        # 2) Spectral priors from decomposer (global, no grad by default)
        with torch.no_grad():
            masks = self.decomposer.spectral.masks().to(device)           # (K,F)
            center, bandwidth = self.decomposer.spectral.mask_stats()     # (K,),(K,)
            center = center.to(device)
            bandwidth = bandwidth.to(device)

        freq_feats = self.freq_embedding(masks)                      # (K,d_model)
        freq_feats = freq_feats.unsqueeze(0).expand(B, -1, -1)       # (B,K,d_model)

        center_feats = self.center_encoder(center.unsqueeze(-1))     # (K,d_model//2)
        center_feats = center_feats.unsqueeze(0).expand(B, -1, -1)   # (B,K,d_model//2)

        bw_feats = self.bandwidth_encoder(bandwidth.unsqueeze(-1))   # (K,d_model//4)
        bw_feats = bw_feats.unsqueeze(0).expand(B, -1, -1)           # (B,K,d_model//4)

        # 3) Combine
        combined = torch.cat([mode_feats, freq_feats, center_feats, bw_feats], dim=-1)
        # (B,K,combined_dim_exact)

        if self.combine_proj is not None:
            combined = self.combine_proj(combined)                   # (B,K,attn_dim)

        # 4) Transformer over modes
        z = self.transformer(combined)                              # (B,K,attn_dim)

        # 5) Pool and predict
        pooled = z.mean(dim=1)                                      # (B,attn_dim)
        out = self.output_head(pooled)                              # (B,1)
        return out
