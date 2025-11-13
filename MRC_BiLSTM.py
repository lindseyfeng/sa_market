import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1, use_bn=True, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.k, self.s, self.d = k, s, d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=0, dilation=d, bias=not use_bn)
        self.bn   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act  = act

    def forward(self, x):
        pad_left = (self.k - 1) * self.d          # causal (left) pad
        x = F.pad(x, (pad_left, 0))
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class rCNN(nn.Module):
    def __init__(self, in_ch, expand_factor=50, act=nn.ReLU(inplace=True)):
        super().__init__()
        mid = expand_factor * in_ch
        # expand, map, compress
        self.conv1 = ConvBlock1D(in_ch, mid, k=4, s=1, d=1, act=act)         # (B, mid, L)
        self.conv2 = ConvBlock1D(mid,   mid, k=4, s=1, d=1, act=act)         # (B, mid, L)
        self.conv3 = ConvBlock1D(mid,   in_ch, k=4, s=1, d=1, act=nn.Identity())  # (B, in_ch, L)

        self.out_act = act   # optional, applied after residual add

    def forward(self, x):
        x0 = x                              # (B, in_ch, L)
        x  = self.conv1(x)                  # (B, mid,   L)
        x  = self.conv2(x) 
        x1 = x                 # (B, mid,   L)
        x  = self.conv3(x+x0)                  # (B, in_ch, L)
        return self.out_act(x + x1)         # residual add (no projection needed)


class rCNN(nn.Module):
    """
    Shapes:
      x0: (B, C, L)
      x1 = conv1(x0)                  -> (B, C,   L)
      x2 = conv2(x1)                  -> (B, C,   L)
      x3 = conv3(cat[x2, x0])         -> (B, C,   L)   # conv3 expects 2*C in-ch, returns C
      y  = cat[out_act(x3), x2]       -> (B, 2*C, L)   # block output
    """
    def __init__(self, in_ch, act=nn.ReLU(inplace=True)):
        super().__init__()
        C = in_ch
        self.conv1 = ConvBlock1D(C,     C,     k=4, s=1, d=1, act=act)          # (B, C,   L)
        self.conv2 = ConvBlock1D(C,     C,     k=3, s=1, d=1, act=act)          # (B, C,   L)
        self.conv3 = ConvBlock1D(2*C,   C,     k=2, s=1, d=1, act=act)# (B, C,   L)
        self.out_act = act
        self.out_channels = 2*C  # for convenience when wiring the next block

    def forward(self, x):                     # x: (B, C, L)
        x0 = x
        x1 = self.conv1(x0)                   # (B, C, L)
        x2 = self.conv2(x1)                   # (B, C, L)
        x3_in = torch.cat([x2, x0], dim=1)    # (B, 2*C, L)  <-- was x2 + x0
        x3 = self.conv3(x3_in)                # (B, C,   L)
        y  = torch.cat([x3, x2], dim=1)  # (B, 2*C, L)  <-- was out_act(x3) + x2
        return y



class biLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.01, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.linear1 = nn.Linear(out_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU(inplace=True)
        self.input_dim = input_dim

    def forward(self, x):                # x: (B, L, input_dim)
      
        _, (h_n, _) = self.lstm(x)       # h_n: (num_layers*dirs, B, hidden_dim)
        if self.bidirectional:
            # last layer's forward and backward states
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 2*hidden_dim)
        else:
            h_last = h_n[-1]                                 # (B, hidden_dim)
        x = self.act(self.linear1(h_last))  # (B, hidden_dim)
        x = self.linear2(x)                 # (B, 1)
        return x, h_last


class MRC_BiLSTM(nn.Module):
    """
    CNN (concat residuals) → **FULL FLATTEN** to (B, C_final*L) → add seq dim → BiLSTM → (B,1)

    Channel growth with your rCNN:
      C0 = input_dim
      C1 = 2*C0
      C2 = 4*C0
      C3 = 8*C0
    """
    def __init__(self, input_dim, seq_len, lstm_hidden=128, lstm_layers=3, bidirectional=True):
        super().__init__()
        C0 = input_dim
        self.cnn1 = rCNN(C0)              # out: 2*C0
        C1 = 2 * C0
        self.cnn2 = rCNN(C1)              # out: 4*C0
        C2 = 2 * C1
        self.cnn3 = rCNN(C2)              # out: 8*C0
        C3 = 2 * C2                        # final channels after 3 blocks

        self.seq_len = seq_len
        self.final_channels = C3

        # LSTM will see a single time step with feature size = C3 * L
        lstm_in = C3 * seq_len
        self.lstm = biLSTM(input_dim=lstm_in, hidden_dim=lstm_hidden,
                           num_layers=lstm_layers, dropout=0.01,
                           bidirectional=bidirectional)

    def forward(self, x):                  # x: (B, C0, L)
        x = self.cnn1(x)                   # (B, 2*C0, L)
        x = self.cnn2(x)                   # (B, 4*C0, L)
        x = self.cnn3(x)                   # (B, 8*C0, L)
   

        B = x.size(0)
        x = x.view(B, -1)                  # -------- (B, C3 * L)  <— your requested (B, C*L)
        x = x.unsqueeze(1)                 # make seq_len=1 → (B, 1, C3*L)
  
        return self.lstm(x)               # (B, 1)


# ----------------------------
# Shape check with your dims
# ----------------------------
if __name__ == "__main__":
    B, C_in, L = 2, 1, 1000          # your input: 24 × 1000
    base = 64                         # choose any from {32,64,128,256}; 64 is common
    model = MRC_BiLSTM(input_dim=C_in, seq_len = L, lstm_hidden=128, lstm_layers=3, bidirectional=True)

    x = torch.randn(B, C_in, L)
    y = model(x)
    print("Input :", x.shape, x)         # (2, 24, 1000)
    print("Output:", y.shape, y)         # (2, 1000, 1)
