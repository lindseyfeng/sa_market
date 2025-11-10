#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, torch, json, os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ✅ use the exact model from train_bo
from train_bo_dropout import MRC_BiLSTM, OneModeDataset, EvalDataset
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- Paper K mapping ---
K_BY_MODE = {
    # VMD modes (Mode_1 .. Mode_12)
    "Mode_1": 9,   "Mode_2": 8,   "Mode_3": 9,   "Mode_4": 11,
    "Mode_5": 10,  "Mode_6": 10,  "Mode_7": 10,  "Mode_8": 9,
    "Mode_9": 7,   "Mode_10": 7,  "Mode_11": 7,  "Mode_12": 6,

    # EWT components (EWT_Component_1 .. EWT_Component_12)
    "EWT_Component_1": 7,   "EWT_Component_2": 10,  "EWT_Component_3": 10,  "EWT_Component_4": 7,
    "EWT_Component_5": 7,   "EWT_Component_6": 8,   "EWT_Component_7": 7,   "EWT_Component_8": 6,
    "EWT_Component_9": 8,   "EWT_Component_10": 11, "EWT_Component_11": 9,  "EWT_Component_12": 10,

    # EWT residual component
    "Residual_minus_EWT_Sum": 5,
}
def get_k(mode): return K_BY_MODE[mode]
def denorm(v, s_min, s_max):
    return v * (s_max - s_min) + s_min


# === Train ===
def train(args):
    df = pd.read_csv("../../VMD_modes_with_residual_2018_2021_with_EWT.csv")
    df_eval = pd.read_csv(args.eval_csv)
    s_val = torch.tensor(df_eval[args.mode_col].to_numpy(), dtype=torch.float32)

    s = torch.tensor(df[args.mode_col].to_numpy(), dtype=torch.float32)
    k = get_k(args.mode_col)

    train_data = OneModeDataset(s, k)
    val_data   = EvalDataset(s_val, k, train_data.s_min, train_data.s_max)
    dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_data, batch_size=args.batch_size)

    model = MRC_BiLSTM(
        k=k,
        g1=args.filters_g1,
        g2=args.filters_g2,
        g3=args.filters_g3,
        lstm_hidden=args.lstm,
        d1=args.dense1,
        d2=args.dense2
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    best_state = None

    for ep in range(args.epochs):
        model.train()

        for xb, yb in dl:
            opt.zero_grad()
            loss = F.mse_loss(denorm(model(xb.to(device)), train_data.s_min, train_data.s_max), yb.to(device))
            loss.backward()
            opt.step()

        model.eval() 
        v_losses = []
        mae_losses =[] 

        with torch.no_grad(): 
            for xb, yb in vl: 
                x = model(xb.to(device))
                y_pred = denorm(x, train_data.s_min, train_data.s_max)
                mae_losses.append(F.l1_loss(y_pred, yb.to(device).item()))
                v_losses.append(F.mse_loss(y_pred, yb.to(device)).item()) 
     
        val_loss = np.mean(v_losses)
        print(f"[EP {ep+1}] val={val_loss:.6f}, MAE = {np.mean(mae_losses):.6f}")
        if val_loss < best:
            best = val_loss; best_state = model.state_dict()

    print(f"✅ Best MSE: {best:.6f}")

   # ==== SAVE ====
    os.makedirs(f"checkpoints/{args.mode_col}", exist_ok=True)
    train_min, train_max = float(train_data.s_min), float(train_data.s_max)
    ckpt = f"checkpoints/{args.mode_col}/best.pt"
    torch.save({
        "model_state_dict": best_state,
        "model_kwargs": {
            "k": get_k(args.mode_col),
            "g1": args.filters_g1,
            "g2": args.filters_g2,
            "g3": args.filters_g3,
            "lstm_hidden": args.lstm,
            "d1": args.dense1,
            "d2": args.dense2
        },
        "k": get_k(args.mode_col),
        "scaler": (train_min, train_max),  # since you removed normalization
    }, ckpt)

    print(f"💾 Saved model -> {ckpt}")

    with open(f"scaler_{args.mode_col}.txt","w") as f:
        f.write(f"{train_data.s_min},{train_data.s_max}")

    with open(f"checkpoints/{args.mode_col}/config.json","w") as f:
        json.dump(vars(args), f, indent=2)

    print("✅ Saved model + scaler + config")

# --- CLI ---
ap = argparse.ArgumentParser()
ap.add_argument("--mode-col", default="Mode_1")

# ✅ model hyperparams (arg form)
ap.add_argument("--filters-g1", type=int, default=64)
ap.add_argument("--filters-g2", type=int, default=64)
ap.add_argument("--filters-g3", type=int, default=64)
ap.add_argument("--eval-csv", default="../../VMD_modes_with_residual_2021_2022_with_EWT.csv")


ap.add_argument("--lstm", type=int, default=150)

ap.add_argument("--dense1", type=int, default=100)
ap.add_argument("--dense2", type=int, default=100)

ap.add_argument("--batch-size", type=int, default=512)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-3)

train(ap.parse_args())
