import torch
import torch.nn as nn
import torch.nn.functional as F


class IMFRepresentation(nn.Module):
    """
    Rich per-mode representation for imfs_ref:

      Input:  imfs_ref (B, K, L)
      Output: mode_feats (B, K, d_model)

    For each mode k, we compute:
      - Conv-based embedding over the tail (last_k steps)
      - Tail statistics: mean, std, slope, last value
      - Relative energy share across modes

    Then combine into a d_model-dim vector.
    """
    def __init__(self, K, L, d_model: int, last_k: int = 32):
        super().__init__()
        self.K = K
        self.L = L
        self.d_model = d_model
        self.last_k = min(last_k, L)

        # 1) Temporal conv encoder over tail
        # We apply the same conv to every (sample, mode) signal.
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

        # 2) MLP to embed scalar stats into d_model and fuse with conv features
        # Stats per mode: mean, std, slope, last_value, relative_energy → 5 scalars
        stats_dim = 5
        hidden = max(d_model // 2, 16)
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        # 3) Final fusion + nonlinearity
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
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

        device = imfs_ref.device

        # ----------------------------
        # 1) Conv embedding over tail
        # ----------------------------
        # (B, K, L) -> (B*K, 1, L)
        x = imfs_ref.reshape(B * K, 1, L)
        x = self.temporal(x)                         # (B*K, d_model, L)

        # keep only last_k timesteps
        if self.last_k < L:
            x_tail = x[:, :, -self.last_k:]         # (B*K, d_model, last_k)
        else:
            x_tail = x                               # (B*K, d_model, L)

        # average over tail timesteps
        conv_emb = x_tail.mean(dim=-1)               # (B*K, d_model)
        conv_emb = conv_emb.view(B, K, self.d_model) # (B, K, d_model)

        # ----------------------------
        # 2) Tail statistics per mode
        # ----------------------------
        if self.last_k < L:
            tail = imfs_ref[:, :, -self.last_k:]     # (B, K, last_k)
        else:
            tail = imfs_ref                          # (B, K, L)

        # mean, std over tail
        tail_mean = tail.mean(dim=-1)                # (B, K)
        tail_std  = tail.std(dim=-1)                 # (B, K)

        # slope ≈ last - first over tail (simple trend proxy)
        first_tail = tail[..., 0]                    # (B, K)
        last_tail  = tail[..., -1]                   # (B, K)
        slope = (last_tail - first_tail) / (self.last_k + 1e-8)

        # last value directly
        last_val = last_tail                         # (B, K)

        # relative energy: ||mode_k||^2 / sum_j ||mode_j||^2
        # compute on tail to match the horizon
        energy_per_mode = (tail ** 2).sum(dim=-1)    # (B, K)
        energy_sum = energy_per_mode.sum(dim=-1, keepdim=True) + 1e-8  # (B,1)
        rel_energy = energy_per_mode / energy_sum    # (B, K)

        # stack stats: (B, K, 5)
        stats = torch.stack(
            [tail_mean, tail_std, slope, last_val, rel_energy],
            dim=-1,
        )                                            # (B, K, 5)

        # feed stats through MLP to get d_model-dim
        stats_emb = self.stats_mlp(stats)            # (B, K, d_model)

        # ----------------------------
        # 3) Fuse conv embedding + stats embedding
        # ----------------------------
        mode_feats = conv_emb + stats_emb            # (B, K, d_model)
        mode_feats = self.fusion(mode_feats)         # (B, K, d_model)

        return mode_feats

class NVMDTransformerPredictor(nn.Module):
    """
    NVMD-aware predictor with richer IMF representation.

      - Input:  imfs_ref (B, K, L) from HybridSpectralNVMD
      - Step 1: IMFRepresentation -> (B, K, d_model)
      - Step 2: Add spectral priors as per-mode bias (optional but recommended)
      - Step 3: Transformer over K mode tokens
      - Step 4: Pool over modes -> scalar prediction (RRP_next)
    """
    def __init__(
        self,
        decomposer,          # HybridSpectralNVMD instance
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        last_k: int = 32,
        use_spectral_priors: bool = True,
    ):
        super().__init__()
        self.decomposer = decomposer
        self.K = decomposer.K
        self.L = decomposer.L
        self.F = decomposer.spectral.F
        self.use_spectral_priors = use_spectral_priors

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model

        # 1) IMF representation
        self.imf_repr = IMFRepresentation(
            K=self.K,
            L=self.L,
            d_model=d_model,
            last_k=last_k,
        )

        if use_spectral_priors:
            # Masks: (K, F) -> (K, d_model)
            self.freq_proj = nn.Linear(self.F, d_model)

            # Center freq μ_k: (K,1) -> (K,d_model)
            self.center_proj = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )

            # Bandwidth σ_k: (K,1) -> (K,d_model)
            self.bandwidth_proj = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )

            # Optional per-mode index embedding
            self.mode_index_embed = nn.Embedding(self.K, d_model)

        # 2) Transformer over modes
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

        # 3) Output head
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

        # 1) Rich IMF-based representation
        mode_feats = self.imf_repr(imfs_ref)    # (B, K, d_model)

        # 2) Optional spectral priors as bias
        if self.use_spectral_priors:
            with torch.no_grad():
                masks = self.decomposer.spectral.masks().to(device)      # (K, F)
                center, bandwidth = self.decomposer.spectral.mask_stats()
                center = center.to(device)
                bandwidth = bandwidth.to(device)

            freq_emb = self.freq_proj(masks)                              # (K,d_model)
            center_emb = self.center_proj(center.unsqueeze(-1))           # (K,d_model)
            bw_emb = self.bandwidth_proj(bandwidth.unsqueeze(-1))         # (K,d_model)

            mode_indices = torch.arange(self.K, device=device)
            mode_idx_emb = self.mode_index_embed(mode_indices)            # (K,d_model)

            prior_emb = freq_emb + center_emb + bw_emb + mode_idx_emb     # (K,d_model)
            prior_emb = prior_emb.unsqueeze(0).expand(B, -1, -1)          # (B,K,d_model)

            mode_feats = mode_feats + prior_emb                           # (B,K,d_model)

        # 3) Transformer over modes
        z = self.transformer(mode_feats)           # (B, K, d_model)

        # 4) Pool and predict
        pooled = z.mean(dim=1)                     # (B, d_model)
        out = self.output_head(pooled)             # (B, 1)
        return out
