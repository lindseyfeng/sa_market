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
    ):
        super().__init__()
        self.signal_len = signal_len
        self.K = K

        self.decomposer = NVMD_Autoencoder(in_ch=1, base=base, K=K, signal_len=signal_len)

        if freeze_decomposer:
            for p in self.decomposer.parameters():
                p.requires_grad = False

 
        self.predictors = nn.ModuleList([
            MRC_BiLSTM(
                input_dim=1,              # each IMF is 1 channel
                seq_len=signal_len,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional
            )
            for _ in range(K)
        ])

    def forward(self, x):
        """
        x: (B, 1, L)
        returns:
            imfs_pred_norm: (B, K, L)  # normalized per-mode sequences from decomposer
            y_modes_norm:   (B, K)     # per-mode next-step (normalized) predictions
        """
        imfs_pred_norm = torch.sigmoid(self.decomposer(x))  # (B, K, L)

        y_list = []
        for i in range(self.K):
            imf_i_norm = imfs_pred_norm[:, i:i+1, :]     # (B,1,L)
            y_i_norm = self.predictors[i](imf_i_norm)    # (B,1) -> next-step (normalized) for mode i
            y_list.append(y_i_norm)

        y_modes_norm = torch.cat(y_list, dim=1)          # (B,K)
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
        freeze_decomposer=False
    )

    x = torch.randn(B, 1, L)
    imfs, y = model(x)
    print("Input:", x.shape)
    print("IMFs :", imfs.shape)
    print("Pred :", y.shape)

