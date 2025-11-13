#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nvmd_autoencoder import NVMD_Autoencoder 


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
    Transformer-like architecture that takes all K modes over L timesteps
    and predicts the next-step RRP as a single scalar.

    Input:
        x_modes: (B, K, L)
            - B: batch size
            - K: number of modes (e.g., 13)
            - L: sequence length

    Output:
        rrp_next_hat: (B, 1)
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
    ):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.d_model = d_model

        # Project K-mode vector at each timestep → d_model features
        self.input_proj = nn.Linear(K, d_model)

        # Positional encoding over time
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder (self-attention over time, mixing modes via input_proj)
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

        # Final head using last timestep representation → scalar RRP
        self.rrp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_modes: torch.Tensor) -> torch.Tensor:
        """
        x_modes: (B, K, L)

        returns:
            rrp_next_hat: (B, 1)
        """
        B, K, L = x_modes.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # (B, K, L) → (B, L, K): treat time as sequence dim
        x = x_modes.permute(0, 2, 1)  # (B, L, K)

        # Project K → d_model
        x = self.input_proj(x)        # (B, L, d_model)

        # Add positional encoding
        x = self.pos_enc(x)           # (B, L, d_model)

        # Transformer encoder
        h = self.encoder(x)           # (B, L, d_model)

        # Use last timestep representation as summary of window
        h_last = h[:, -1, :]          # (B, d_model)

        # Predict RRP_{t+1}
        rrp_next_hat = self.rrp_head(h_last)  # (B, 1)
        return rrp_next_hat



class NVMDTransformerRRPModel(nn.Module):
    """
    Full model:
        raw RRP window (B,1,L)
        → NVMD_Autoencoder decomposer → IMFs (B,K,L)
        → MultiModeTransformerRRP → next-step RRP (B,1)
    """
    def __init__(
        self,
        seq_len: int = 64,
        K: int = 13,
        base: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        freeze_decomposer: bool = True,
    ):
        super().__init__()
        self.decomposer = NVMD_Autoencoder(
            in_ch=1,
            base=base,
            K=K,
            signal_len=seq_len,
        )

        self.predictor = MultiModeTransformerRRP(
            K=K,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        if freeze_decomposer:
            for p in self.decomposer.parameters():
                p.requires_grad = False
            self.decomposer.eval()

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B,1,L) raw RRP window
        returns:
            rrp_next_hat: (B,1)
        """
        imfs = self.decomposer(x_raw)          # (B,K,L)
        rrp_next_hat = self.predictor(imfs)    # (B,1)
        return rrp_next_hat

