#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for batch_first inputs.

    Expects x of shape (B, L, d_model) and adds a fixed positional bias
    before dropout.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)      # even dims
        pe[:, 1::2] = torch.cos(position * div_term)      # odd dims

        # store as (1, max_len, d_model) for easy broadcasting to (B, L, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]  # (1, L, d_model) -> broadcast
        return self.dropout(x)


class EnhancedNVMDTransformer(nn.Module):
    """
    Enhanced NVMD-aware predictor.

    The ONLY job: given a raw RRP window x_raw (B,1,L),
    predict the next-step RRP (B,1).

    Pipeline:
      x_raw (B,1,L)
        └─ decomposer(x_raw) → imfs (B,K,L), recon (B,1,L)
           └─ multi-scale Conv1d over raw / imfs / recon → (B,d_model,L)
              └─ Transformer over time (L tokens, d_model)
                 └─ attention pooling over time → (B,d_model)
                    └─ MLP → (B,1) next-step price
    """
    def __init__(
        self,
        decomposer,             # HybridSpectralNVMD instance
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        use_multi_scale: bool = True,
    ):
        super().__init__()
        self.decomposer = decomposer
        self.K = decomposer.K
        self.use_multi_scale = use_multi_scale

        assert d_model % 4 == 0, "d_model must be divisible by 4 for the multi-scale channel split."
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        # ------------------------------------------------
        # Multi-scale feature extraction
        # ------------------------------------------------
        if use_multi_scale:
            # raw:   1 channel  -> d_model/4
            # imfs:  K channels -> d_model/2
            # recon: 1 channel  -> d_model/4
            self.raw_proj = nn.Conv1d(
                in_channels=1,
                out_channels=d_model // 4,
                kernel_size=3,
                padding=1,
            )

            # IMF branch: depthwise (per-mode) then pointwise (mix modes)
            self.imf_depthwise = nn.Conv1d(
                in_channels=self.K,
                out_channels=self.K,
                kernel_size=3,
                padding=1,
                groups=self.K,         # each mode has its own filter
            )
            self.imf_pointwise = nn.Conv1d(
                in_channels=self.K,
                out_channels=d_model // 2,
                kernel_size=1,
            )

            self.recon_proj = nn.Conv1d(
                in_channels=1,
                out_channels=d_model // 4,
                kernel_size=3,
                padding=1,
            )
        else:
            # Only IMFs, all mixed into a d_model-dimensional sequence
            self.imf_proj = nn.Conv1d(
                in_channels=self.K,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
            )

        # Normalize combined features before positional encoding
        self.feature_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------
        # Positional encoding + Transformer encoder (over time)
        # ------------------------------------------------
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # (B, L, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ------------------------------------------------
        # Multi-head attention pooling over time
        # ------------------------------------------------
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # query/key/value: (B, L, d_model)
        )

        # ------------------------------------------------
        # Output head: ONLY next-step RRP
        # ------------------------------------------------
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_ff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff // 2, 1),
        )

    def _multi_scale_features(
        self,
        x_raw: torch.Tensor,
        imfs: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build (B, d_model, L) feature map from raw, imfs, recon.
        """
        if self.use_multi_scale:
            # Raw branch
            raw_features = self.raw_proj(x_raw)              # (B, d_model/4, L)
            raw_features = F.gelu(raw_features)

            # IMF branch: per-mode temporal filtering, then mixing modes
            imf_features = self.imf_depthwise(imfs)          # (B, K, L)
            imf_features = F.gelu(imf_features)
            imf_features = self.imf_pointwise(imf_features)  # (B, d_model/2, L)
            imf_features = F.gelu(imf_features)

            # Reconstruction branch
            recon_features = self.recon_proj(recon)          # (B, d_model/4, L)
            recon_features = F.gelu(recon_features)

            # Concatenate along channel dimension
            features = torch.cat(
                [raw_features, imf_features, recon_features], dim=1
            )                                                # (B, d_model, L)
        else:
            imf_features = self.imf_proj(imfs)               # (B, d_model, L)
            features = F.gelu(imf_features)

        return features

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B, 1, L)  raw RRP window

        Returns:
          rrp_next_hat: (B, 1)  predicted next-step price
        """
        B, C, L = x_raw.shape
        assert C == 1, f"Expected 1 channel for x_raw, got {C}"

        # 1) NVMD decomposition
        # imfs:  (B, K, L)
        # recon: (B, 1, L)
        imfs, recon, _, _ = self.decomposer(x_raw)

        # 2) Multi-scale features
        features = self._multi_scale_features(x_raw, imfs, recon)  # (B, d_model, L)

        # 3) Reorder to (B, L, d_model) for Transformer
        features = features.transpose(1, 2)                        # (B, L, d_model)

        # 4) Normalize + positional encoding
        features = self.feature_norm(features)
        features = self.pos_encoding(features)                     # (B, L, d_model)

        # 5) Transformer over time
        encoded = self.transformer(features)                       # (B, L, d_model)

        # 6) Attention pooling over time
        pool_query = encoded.mean(dim=1, keepdim=True)             # (B, 1, d_model)
        pooled, _ = self.attention_pool(
            pool_query,  # query: (B, 1, d_model)
            encoded,     # key:   (B, L, d_model)
            encoded,     # value: (B, L, d_model)
        )
        pooled = pooled.squeeze(1)                                 # (B, d_model)

        # 7) Final scalar prediction
        rrp_next_hat = self.output_head(pooled)                    # (B, 1)
        return rrp_next_hat
