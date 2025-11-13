# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvmd_autoencoder import NVMD_Autoencoder
from train.baseline.cnn_bilstm import MRC_BiLSTM


class NVMD_MRC_BiLSTM(nn.Module):
    def __init__(
        self,
        signal_len: int,
        base: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        bidirectional: bool = True,
        freeze_decomposer: bool = False,
        use_sigmoid: bool = False,   
    ):
        super().__init__()
        self.signal_len = signal_len
        self.use_sigmoid = use_sigmoid

        # NOTE: per-mode decomposer -> K=1
        self.decomposer = NVMD_Autoencoder(
            in_ch=1,
            base=base,
            K=1,
            signal_len=signal_len,
        )

        if freeze_decomposer:
            for p in self.decomposer.parameters():
                p.requires_grad = False

        self.predictor = MRC_BiLSTM(
            input_dim=1,            
            seq_len=signal_len,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
        )

    def _decompose_norm(self, x: torch.Tensor) -> torch.Tensor:

        imf = self.decomposer(x)     # (B,1,L) 
        if self.use_sigmoid:
            imf = torch.sigmoid(imf)
        return imf

    def forward(self, x: torch.Tensor):

        B, C, L = x.shape
        assert C == 1, f"Expected x with 1 channel, got {C}"
        assert L == self.signal_len, f"Expected seq_len={self.signal_len}, got {L}"


        imf_pred_norm = self._decompose_norm(x)        # (B,1,L)
        y_mode_norm = self.predictor(imf_pred_norm)    # (B,1)

        return imf_pred_norm, y_mode_norm

    @torch.no_grad()
    def decompose(self, x: torch.Tensor):

        return self._decompose_norm(x)


class MultiModeNVMD_MRC_BiLSTM(nn.Module):

    def __init__(
        self,
        signal_len: int,
        K: int = 13,
        base: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        bidirectional: bool = True,
        freeze_decomposer: bool = False,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.signal_len = signal_len
        self.K = K

        self.models = nn.ModuleList([
            NVMD_MRC_BiLSTM(
                signal_len=signal_len,
                base=base,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
                freeze_decomposer=freeze_decomposer,
                use_sigmoid=use_sigmoid,
            )
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: (B,1,L)

        returns:
            imfs_pred_norm_all: (B,K,L)
            y_modes_norm:       (B,K)
        """
        B, C, L = x.shape
        assert C == 1, f"Expected x with 1 channel, got {C}"
        assert L == self.signal_len, f"Expected seq_len={self.signal_len}, got {L}"

        imfs_list = []
        y_list = []

        for m in self.models:
            imf_i_norm, y_i_norm = m(x)     # (B,1,L), (B,1)
            imfs_list.append(imf_i_norm)
            y_list.append(y_i_norm)

        y_modes_norm = torch.cat(y_list, dim=1)           # list of (B,1)   -> (B,K)

        return imfs_pred_norm_all, y_modes_norm

    @torch.no_grad()
    def decompose_all(self, x: torch.Tensor):
        """
        Returns stacked decomposed sequences from all per-mode models:
            (B,K,L)
        """
        imfs_list = []
        for m in self.models:
            imf_i = m.decompose(x)    # (B,1,L)
            imfs_list.append(imf_i)
        return torch.cat(imfs_list, dim=1)  # (B,K,L)
