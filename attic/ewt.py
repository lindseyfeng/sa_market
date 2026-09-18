import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from ewtpy import EWT1D

# -----------------------------
# 1. Load VMD residual data
# -----------------------------
input_file = "VMD_modes_with_residual_2018_2021.csv"
df = pd.read_csv(input_file)

# Expect these columns to exist
if 'Residual' not in df.columns:
    raise ValueError("Expected column 'Residual' not found in the CSV.")
if 'SETTLEMENTDATE' not in df.columns:
    print("Warning: 'SETTLEMENTDATE' not found; continuing without timestamp column.")

residual = df['Residual'].to_numpy()
timestamps = df['SETTLEMENTDATE'].to_numpy() if 'SETTLEMENTDATE' in df.columns else None

print(f"Residual signal length: {len(residual)}")

# -----------------------------
# 2. Set EWT parameters
# -----------------------------
N = 12  # number of components

# -----------------------------
# 3. Time the EWT decomposition
# -----------------------------
start_time = time.time()
ewt, mfb, boundaries = EWT1D(
    residual, 
    N=N, 
    log=0, 
    detect="locmax",
    completion=0, 
    reg='average',
    lengthFilter=10, 
    sigmaFilter=5
)
end_time = time.time()

print(f"EWT completed in {end_time - start_time:.2f} seconds")
print(f"Decomposed into {N} components, each length {ewt.shape[0]}")

# -----------------------------
# 4. Build a DataFrame for EWT components
#    (shape expected: [T, N])
# -----------------------------
ewt_cols = [f'EWT_Component_{i+1}' for i in range(N)]
ewt_df = pd.DataFrame(ewt, columns=ewt_cols)

# Keep original columns and append new ones
out_df = df.copy()
out_df[ewt_cols] = ewt_df

# -----------------------------
# 5. Add sums: EWT_Sum and IMF_Sum
# -----------------------------
# Sum of all EWT components (should approx. reconstruct the residual)
out_df['EWT_Sum'] = out_df[ewt_cols].sum(axis=1)

# Auto-detect IMF/VMD mode columns from the original df
# Common patterns: 'IMF_1', 'IMF1', 'VMD_Mode_1', 'Mode1', etc.
def is_imf_col(col: str) -> bool:
    name = col.lower()
    # Heuristics: contains 'imf' OR ('mode' in name) OR ('vmd' in name but not 'residual')
    # Exclude obviously non-mode columns
    if 'residual' in name:
        return False
    if 'imf' in name:
        return True
    if 'mode' in name:
        return True
    if 'vmd' in name:
        return True
    return False

# Only scan original columns for IMFs/modes
original_cols = list(df.columns)
imf_cols = [c for c in original_cols if is_imf_col(c)]

if len(imf_cols) == 0:
    print("Warning: No IMF/VMD mode columns detected in the original file; 'IMF_Sum' will be NaN.")
    out_df['IMF_Sum'] = np.nan
else:
    out_df['IMF_Sum'] = out_df[imf_cols].sum(axis=1)
    print(f"Detected {len(imf_cols)} IMF/VMD mode columns for IMF_Sum: {imf_cols[:6]}{'...' if len(imf_cols)>6 else ''}")

# Optional check: difference between residual and EWT reconstruction
out_df['Residual_minus_EWT_Sum'] = out_df['Residual'] - out_df['EWT_Sum']

# -----------------------------
# 6. Save with all original + new columns
# -----------------------------
output_file = "VMD_modes_with_residual_2021_2022_with_EWT.csv"
out_df.to_csv(output_file, index=False)
print(f"Saved augmented file with EWT components and sums to '{output_file}'")

# -----------------------------
# 7. Optional: plot decomposed components
# -----------------------------
plt.figure(figsize=(12, max(6, 0.6*N + 2)))
for i in range(N):
    plt.subplot(N, 1, i+1)
    plt.plot(ewt[:, i])
    plt.title(f"EWT Component {i+1}")
    plt.tight_layout()
plt.show()
