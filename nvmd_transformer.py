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

from nvmd_autoencoder import MultiModeNVMD

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
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


class ModeTimeTransformerRRP(nn.Module):
    def __init__(
        self,
        K: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.d_model = d_model


        self.value_proj = nn.Linear(1, d_model)
        self.mode_emb = nn.Embedding(K, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        full_len = K * seq_len + 1
        self.pos_enc = PositionalEncoding(d_model, max_len=full_len, dropout=dropout)

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


        mode_ids = torch.arange(K).view(K, 1).expand(K, seq_len).reshape(-1)  # (K*L,)
        self.register_buffer("mode_ids_flat", mode_ids, persistent=False)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x_modes: torch.Tensor) -> torch.Tensor:

        B, K, L = x_modes.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"


        x_flat = x_modes.reshape(B, K * L, 1)          # (B, K*L, 1)
        token_vals = self.value_proj(x_flat)           # (B, K*L, d_model)

        mode_emb = self.mode_emb(self.mode_ids_flat)   # (K*L, d_model)
        mode_emb = mode_emb.unsqueeze(0)               # (1, K*L, d_model)
        x = token_vals + mode_emb                      # (B, K*L, d_model)
        cls = self.cls_token.expand(B, 1, self.d_model)  # (B,1,d_model)
        x = torch.cat([cls, x], dim=1)                   # (B, 1+K*L, d_model)
        x = self.pos_enc(x)                              # (B, 1+K*L, d_model)
        h = self.encoder(x)                              # (B, 1+K*L, d_model)
        cls_out = h[:, 0, :]                             # (B, d_model)

        rrp_next_hat = self.rrp_head(cls_out)            # (B, 1)
        return rrp_next_hat



class NVMDTransformerRRPModel(nn.Module):
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
        self.decomposer = MultiModeNVMD(
            base=base,
            K=K,
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
        imfs, sum = self.decomposer(x_raw)          # (B,K,L)
        rrp_next_hat = self.predictor(imfs)    # (B,1)
        return rrp_next_hat

