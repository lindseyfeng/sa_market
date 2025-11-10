#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correct BO + your original CNN-concat + flatten → one BiLSTM → Dense
"""

import argparse, os, random, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import optuna
except:
    optuna = None

# =========================
# Fixed k for each mode
# =========================
K_BY_MODE = {
    "Mode_1": 9, "Mode_2": 8, "Mode_3": 9, "Mode_4": 11,
    "Mode_5": 10, "Mode_6": 10, "Mode_7": 10, "Mode_8": 9,
    "Mode_9": 7, "Mode_10": 7, "Mode_11": 7, "Mode_12": 6,
    "EWT_Component_1": 7, "EWT_Component_2": 10, "EWT_Component_3": 10, "EWT_Component_4": 7,
    "EWT_Component_5": 7, "EWT_Component_6": 8, "EWT_Component_7": 7, "EWT_Component_8": 6,
    "EWT_Component_9": 8, "EWT_Component_10": 11, "EWT_Component_11": 9, "EWT_Component_12": 10,
    "Residual_minus_EWT_Sum": 5,
}

def get_k(mode):
    return K_BY_MODE[mode]

def denorm(v, s_min, s_max):
    return v * (s_max - s_min) + s_min

# =========================
# Your CNN + concat model
# =========================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k, act=nn.ReLU()):
        super().__init__()
        pad = k - 1  # causal left pad
        self.pad = (pad, 0)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k)
        self.act = act
    def forward(self, x):
        x = F.pad(x, self.pad)
        return self.act(self.conv(x))

class rCNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # all kernels = 4 as you want
        self.c1 = ConvBlock1D(in_ch, out_ch, 4)
        self.c2 = ConvBlock1D(out_ch, out_ch, 4)
        
        # conv3 takes concat([x2, x]) = (out_ch + in_ch) channels
        self.c3 = ConvBlock1D(out_ch + in_ch, out_ch, 4)

    def forward(self, x):
        x1 = self.c1(x)                        # (B, out_ch, L)
        x2 = self.c2(x1)                       # (B, out_ch, L)
        
        # concat input skip, like paper
        x3 = self.c3(torch.cat([x2, x], dim=1))# (B, out_ch, L)
        
        # concat residual output with x2
        return torch.cat([x3, x2], dim=1)      # (B, 2*out_ch, L)

class biLSTM(nn.Module):
    def __init__(self, input_dim, hidden, d1, d2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden*2, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.out = nn.Linear(d2, 1)
        self.act = nn.ReLU()
    def forward(self, x):                     # x: (B, L, C)
        _, (h, _) = self.lstm(x)              # h: (2, B, H)
        h = torch.cat([h[-2], h[-1]], 1)      # (B, 2H)
        z = self.act(self.fc1(h))
        z = self.act(self.fc2(z))
        return self.out(z)                    # (B, 1)


class MRC_BiLSTM(nn.Module):
    def __init__(self, k, g1, g2, g3, lstm_hidden, d1, d2):
        super().__init__()
        self.g1 = rCNN(1, g1)
        self.g2 = rCNN(g1*2, g2)
        self.g3 = rCNN(g2*2, g3)
        final_c = g3 * 2
        self.lstm_in = final_c * k
        self.lstm = biLSTM(self.lstm_in, lstm_hidden, d1, d2)
    def forward(self, x):
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        B,C,L = x.shape
        return self.lstm(x.contiguous().view(B,1,C*L))

# =========================
# Dataset
# =========================

class OneModeDataset(Dataset):
    def __init__(self, s, k):
        self.s = s
        self.k = k
        self.s_min, self.s_max = float(s.min()), float(s.max())
        self.x = (s - self.s_min) / (self.s_max - self.s_min)
    def __len__(self): return len(self.s)-self.k
    def __getitem__(self, i):
        x = self.x[i:i+self.k]
        y = self.s[i+self.k]
        return x.unsqueeze(0), y.unsqueeze(0)
    

class EvalDataset(Dataset):
    def __init__(self, s, k, s_min, s_max):
        self.s = s
        self.k = k
        self.x = (s - s_min) / (s_max - s_min)
    def __len__(self): return len(self.s)-self.k
    def __getitem__(self, i):
        x = self.x[i:i+self.k]
        y = self.s[i+self.k]
        return x.unsqueeze(0), y.unsqueeze(0)


# =========================
# Train helper
# =========================
def train_one(train_series, val_series, k, ctor, batch, epochs, device):
    # series is train only now, val passed separately
    trds = OneModeDataset(train_series, k)
    vads = EvalDataset(val_series, k, trds.s_min, trds.s_max)

    trdl = DataLoader(trds, batch, True)
    vadl = DataLoader(vads, batch, False)
    model = ctor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        for xb, yb in trdl:
            opt.zero_grad()
            loss = F.mse_loss(denorm(model(xb.to(device)), trds.s_min, trds.s_max), yb.to(device))
            loss.backward()
            opt.step()

        model.eval() 
        v_losses = [] 
        with torch.no_grad(): 
            for xb, yb in vadl: 
                x = model(xb.to(device))
                v_losses.append(F.mse_loss(denorm(x, trds.s_min, trds.s_max), yb.to(device)).item()) 
     
        v = np.mean(v_losses)


        # ---- Logging ----
        print(f"[Epoch {ep:02d}] val_mse = {v:.6f}")

        # ---- Track best ----
        if v < best:
            best = v
            best_state = {k:v.clone() for k,v in model.state_dict().items()}

    # ---- Restore best ----
    model.load_state_dict(best_state)
    return model, best, (trds.s_min, trds.s_max)


# =========================
# Main (Optuna)
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-col", default="Mode_1")
    ap.add_argument("--csv", default="../../VMD_modes_with_residual_2018_2021_with_EWT.csv")
    ap.add_argument("--optuna-trials", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    s = torch.tensor(df[args.mode_col].to_numpy(), dtype=torch.float32)
    df_val = pd.read_csv("../../VMD_modes_with_residual_2021_2022_with_EWT.csv")
    s_val = torch.tensor(df_val[args.mode_col].to_numpy(), dtype=torch.float32)

    k = get_k(args.mode_col)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.optuna_trials > 0:
        assert optuna is not None, "pip install optuna"
        def obj(t):
            g1 = t.suggest_categorical("g1",[32,64,128])
            g2 = t.suggest_categorical("g2",[32,64,128])
            g3 = t.suggest_categorical("g3",[32,64,128])
            lstm = t.suggest_categorical("lstm",[50,100,128])
            d1 = t.suggest_categorical("d1",[40,100])
            d2 = t.suggest_categorical("d2",[40,100])
            batch = t.suggest_categorical("batch",[300,500,1000])
            def ctor():
                return MRC_BiLSTM(k,g1,g2,g3,lstm,d1,d2)
            _,best,_ = train_one(s, s_val, k, ctor, batch, epochs=10, device=device)

            return best
        study = optuna.create_study(direction="minimize")
        study.optimize(obj, args.optuna_trials)
        print(study.best_params)
        return

    else:
        def ctor(): return MRC_BiLSTM(k,64,64,64,128,100,100)
        model,best,scaler = train_one(s, s_val, k, ctor, 512, epochs=20, device=device)
        print("Best",best,"scaler",scaler)

if __name__=="__main__":
    main()
