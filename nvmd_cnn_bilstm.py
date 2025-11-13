# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvmd_autoencoder import NVMD_Autoencoder
from MRC_BiLSTM import MRC_BiLSTM


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
        self.rrp_head = nn.Linear(lstm_hidden * (2 if bidirectional else 1), 1)

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
        y_mode_norm, last_h = self.predictor(imf_pred_norm)    # (B,1)

        rrp_hat = self.rrp_head(last_h)  # (B,1)
        return imf_pred_norm, y_mode_norm, rrp_hat

    @torch.no_grad()
    def decompose(self, x: torch.Tensor):

        return self._decompose_norm(x)


class MultiModeNVMD_MRC_BiLSTM(nn.Module):
    """
    Heterogeneous multi-mode wrapper: takes a list of *already built*
    NVMD_MRC_BiLSTM submodels. Each submodel can have different base/hidden/layers.
    """
    def __init__(self, models: list[nn.Module], use_sigmoid: bool = True):
        super().__init__()
        assert len(models) > 0, "Need at least one submodel"
        self.models = nn.ModuleList(models)
        self.K = len(models)
        self.signal_len = models[0].signal_len   # all should share seq_len
        self.use_sigmoid = use_sigmoid

        # MLP price head: K mode scalars -> 1 price
        hidden = max(4, self.K // 2)
        self.price_head = nn.Sequential(
            nn.Linear(self.K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B,1,L)

        returns:
            imfs_pred_norm_all: (B,K,L)
            y_modes_norm:       (B,K)
            y_price:            (B,1)
        """
        B, C, L = x.shape
        assert C == 1, f"Expected x with 1 channel, got {C}"
        assert L == self.signal_len, f"Expected seq_len={self.signal_len}, got {L}"

        imfs_list = []
        y_list = []

        for m in self.models:
            imf_i_norm, y_i_norm = m(x)   # (B,1,L), (B,1)
            imfs_list.append(imf_i_norm)
            y_list.append(y_i_norm)

        imfs_pred_norm_all = torch.cat(imfs_list, dim=1)  # (B,K,L)
        y_modes_norm = torch.cat(y_list, dim=1)           # (B,K)
        y_price = self.price_head(y_modes_norm)           # (B,1)

        return imfs_pred_norm_all, y_modes_norm, y_price

    @torch.no_grad()
    def decompose_all(self, x: torch.Tensor):
        imfs_list = []
        for m in self.models:
            imf_i = m.decompose(x)        # (B,1,L)
            imfs_list.append(imf_i)
        return torch.cat(imfs_list, dim=1)  # (B,K,L)
