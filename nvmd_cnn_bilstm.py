# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- import your existing modules/classes
from nvmd_autoencoder import NVMD_Autoencoder
from train.baseline.cnn_bilstm import MRC_BiLSTM


class NVMD_MRC_BiLSTM(nn.Module):
    def __init__(
        self,
        signal_len: int,
        K: int = 13,                # number of decomposed signals
        base: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        bidirectional: bool = True,
        freeze_decomposer: bool = False,
        d_mode_embed: int = 128
        
    ):
        super().__init__()
        self.signal_len = signal_len
        self.K = K

        self.decomposer = NVMD_Autoencoder(in_ch=1, base=base, K=K, signal_len=signal_len)

        if freeze_decomposer:
            for p in self.decomposer.parameters():
                p.requires_grad = False

        self.mode_embed = nn.Embedding(K, d_mode_embed)
 
        self.predictor = MRC_BiLSTM(
            input_dim=1 + d_mode_embed,
            seq_len=signal_len,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """
        x: (B, 1, L)
        returns:
            imfs_pred_norm: (B, K, L)  # normalized per-mode sequences from decomposer
            y_modes_norm:   (B, K)     # per-mode next-step (normalized) predictions
        """
        imfs_pred_norm = torch.sigmoid(self.decomposer(x))  # (B, K, L)

        device = x.device
        mode_ids = torch.arange(self.K, device=device)      # (K,)
        mode_emb = self.mode_embed(mode_ids)                # (K, d_mode_embed)

        # Expand to batch: (B,K,d_mode_embed)
        mode_emb = mode_emb.unsqueeze(0).expand(B, -1, -1)  # (B,K,d_mode_embed)

        imfs_flat = imfs_pred_norm.view(B * self.K, 1, L)   # (B*K,1,L)

        mode_emb_flat = mode_emb.view(B * self.K, self.d_mode_embed)  # (B*K,d)
        mode_emb_flat = mode_emb_flat.unsqueeze(-1).expand(-1, -1, L) # (B*K,d,L)

        predictor_in = torch.cat([imfs_flat, mode_emb_flat], dim=1)

        y_flat = self.predictor(predictor_in)  # (B*K,1)
        
        y_modes_norm = y_flat.view(B, self.K) # (B,K)

        return imfs_pred_norm, y_modes_norm



    @torch.no_grad()
    def decompose(self, x):
        return self.decomposer(x)


# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    B, L = 1, 32
    K = 13

    model = NVMD_MRC_BiLSTM(
        signal_len=L,
        K=K,
        base=64,
        lstm_hidden=128,
        lstm_layers=3,
        bidirectional=True,
        freeze_decomposer=False,
        
    )

    x = torch.randn(B, 1, L)
    imfs, y = model(x)
    print("Input:", x.shape)
    print("IMFs :", imfs.shape)
    print("Pred :", y.shape)

