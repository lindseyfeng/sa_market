from transformer_predictor import MultiModeTransformerPredictor

class NVMD_Transformer_Model(nn.Module):
    def __init__(self, seq_len=64, K=13, base=64, d_model=128, n_heads=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        # Your existing decomposer
        self.decomposer = NVMD_Autoencoder(in_ch=1, base=base, K=K, signal_len=seq_len)

        # New transformer predictor on top of all modes jointly
        self.predictor = MultiModeTransformerPredictor(
            K=K,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )

    def forward(self, x_raw):
        """
        x_raw: (B,1,L) raw RRP window
        returns:
            imfs_recon:   (B,K,L)
            imf_next_hat: (B,K)
            rrp_next_hat: (B,1)
        """
        imfs = self.decomposer(x_raw)             # (B,K,L)
        imfs_recon, imf_next_hat, rrp_next_hat = self.predictor(imfs)
        return imfs_recon, imf_next_hat, rrp_next_hat
