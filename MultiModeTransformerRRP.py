# train_transformer_with_nvmd.py (or just update your train_transformer.py)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_nvmd import HybridSpectralNVMD  # <-- your NVMD decomposer


class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding for 1D sequences.
    Expects input shape (B, L, d_model).
    """
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (L, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class MultiModeTransformerRRP(nn.Module):
    """
    Two modes:

    1) use_nvmd = False (original behavior)
       - Input:  x_modes (B, K, L)  — directly the mode signals
       - Output: rrp_next_hat (B, 1)

    2) use_nvmd = True
       - Input:  x_raw (B, 1, L)    — raw RRP window
       - Internally:
            x_raw --HybridSpectralNVMD--> imfs_ref (B, K, L)
            imfs_ref --Transformer--> rrp_next_hat
       - Optionally returns NVMD extras.

    Forward signatures:

        y = model(x_modes)                           # if use_nvmd = False

        y, recon_ref, smooth, ortho = model(
            x_raw, return_nvmd=True
        )                                            # if use_nvmd = True
    """
    def __init__(
        self,
        K: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        use_nvmd: bool = False,
    ):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_nvmd = use_nvmd

        # If we are using NVMD inside, build a decomposer that operates on this seq_len.
        if self.use_nvmd:
            self.decomposer = HybridSpectralNVMD(K=K, signal_len=seq_len)

        # Project K-mode vector at each time → d_model
        self.input_proj = nn.Linear(K, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # (B, L, d_model)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.rrp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_nvmd: bool = False,
    ):
        """
        If self.use_nvmd == False:
            x: (B, K, L)  — modes
        If self.use_nvmd == True:
            x: (B, 1, L)  — raw RRP

        Returns:
            if not use_nvmd or not return_nvmd:
                rrp_next_hat: (B, 1)

            if use_nvmd and return_nvmd:
                rrp_next_hat: (B, 1)
                recon_ref:   (B, 1, L)
                smooth_loss: scalar (tensor)
                ortho_loss:  scalar (tensor)
        """
        if self.use_nvmd:
            # x is raw RRP: (B,1,L)
            x_raw = x
            B, C, L = x_raw.shape
            assert C == 1, f"Expected x_raw with channel=1, got {C}"
            assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

            # Decompose
            imfs_ref, recon_ref, imfs_lin, recon_lin = self.decomposer(x_raw)  # (B,K,L), (B,1,L), ...

            # Use refined IMFs as modes for the transformer
            x_modes = imfs_ref  # (B,K,L)

            # NVMD regularizers
            smooth_loss = self.decomposer.spectral.spectral_smoothness_loss()
            ortho_loss  = self.decomposer.spectral.orthogonality_loss()
        else:
            # Original behavior: x is already (B,K,L)
            x_modes = x
            B, K, L = x_modes.shape
            assert K == self.K, f"Expected K={self.K}, got {K}"
            assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"
            recon_ref = None
            smooth_loss = None
            ortho_loss = None

        # -----------------------------
        # Transformer path (unchanged)
        # -----------------------------
        # Treat time as sequence dimension
        x_seq = x_modes.permute(0, 2, 1)   # (B, L, K)

        x_seq = self.input_proj(x_seq)     # (B, L, d_model)
        x_seq = self.pos_enc(x_seq)        # (B, L, d_model)

        h = self.encoder(x_seq)            # (B, L, d_model)

        # Use last time step as representation
        h_last = h[:, -1, :]               # (B, d_model)

        rrp_next_hat = self.rrp_head(h_last)  # (B, 1)

        if self.use_nvmd and return_nvmd:
            return rrp_next_hat, recon_ref, smooth_loss, ortho_loss
        else:
            return rrp_next_hat
