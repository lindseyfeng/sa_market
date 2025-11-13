import torch
import torch.nn as nn

# Adjust these imports to match your actual file structure
# e.g. from nvmd_autoencoder import MultiModeNVMD
#      from train.baseline.cnn_bilstm import MRC_BiLSTM
from nvmd_autoencoder import MultiModeNVMD
from train.baseline.cnn_bilstm import MRC_BiLSTM


class NVMDMultiModePriceModel(nn.Module):
    
    def __init__(
        self,
        signal_len: int,
        K: int = 13,
        base: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        bidirectional: bool = True,
        freeze_decomposer: bool = True,
        mlp_hidden: int | None = 64,
    ):
        super().__init__()
        self.signal_len = signal_len
        self.K = K

        self.decomposer = MultiModeNVMD(K=K, base=base)

        if freeze_decomposer:
            for p in self.decomposer.parameters():
                p.requires_grad = False

        self.predictors = nn.ModuleList([
            MRC_BiLSTM(
                input_dim=1,
                seq_len=signal_len,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
            )
            for _ in range(K)
        ])

        in_dim = K
        if mlp_hidden is None:
            # simple linear comb of mode predictions
            self.price_head = nn.Linear(in_dim, 1)
        else:
            self.price_head = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, 1),
            )

    def forward(self, x: torch.Tensor):

        B, C, L = x.shape
        assert C == 1, f"Expected x with 1 channel, got {C}"
        assert L == self.signal_len, f"Expected seq_len={self.signal_len}, got {L}"

        imfs, _ = self.decomposer(x)       

        mode_preds = []
        for k, predictor in enumerate(self.predictors):
            # slice mode k: (B,1,L)
            mode_k = imfs[:, k:k+1, :]       
            y_k = predictor(mode_k)         
            mode_preds.append(y_k)

        y_modes = torch.cat(mode_preds, dim=1)   

        y_price = self.price_head(y_modes)        

        return imfs, y_modes, y_price

    def load_decomposer_ckpt(self, ckpt_path: str, strict: bool = True):
        ck = torch.load(ckpt_path, map_location="cpu")
        state = ck.get("model_state", ck)
        missing, unexpected = self.decomposer.load_state_dict(state, strict=strict)
        return missing, unexpected
