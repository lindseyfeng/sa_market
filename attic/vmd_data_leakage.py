import pandas as pd
from vmdpy import VMD
import numpy as np
import time

# -----------------------------
# 1. Load combined CSV
# -----------------------------
# Assuming your CSV is loaded correctly as before
df = pd.read_csv("SA_prices_combined_2018_2022.csv")
df = df[(df['RRP'] >= 1) & (df['RRP'] <= 981.65)].copy()
df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)

df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

# -----------------------------
# 2. VMD parameters
# -----------------------------
K = 12
alpha = 2000
tau = 0.
DC = 0
init = 1
tol = 1e-7

# -----------------------------
# 3. Define Periods
# -----------------------------
periods = {
    "2018_2018": ("2018-01-01", "2018-12-31"),
    "2019_2019": ("2019-01-01", "2019-12-31"),
    # Add other periods as needed
}

# -----------------------------
# 4. Processing Function (Leakage Free)
# -----------------------------
def run_vmd_on_period(full_df, start_date, end_date, buffer_size=1000):
    """
    Runs VMD on a specific time slice. 
    Includes a 'buffer' of previous data to prevent edge effects at the start 
    of the period, ensuring continuity without looking into the future.
    """
    
    # 1. Identify the indices for the target period
    mask = (full_df['SETTLEMENTDATE'] >= start_date) & (full_df['SETTLEMENTDATE'] <= end_date)
    period_df = full_df[mask].copy()
    
    if period_df.empty:
        print(f"No data found for {start_date} to {end_date}")
        return None

    start_idx = period_df.index[0]
    end_idx = period_df.index[-1]
    
    # 2. Add Buffer (Lookback)
    # We take 'buffer_size' rows from BEFORE the start_date to help VMD stabilize.
    # We do NOT take rows from after end_date (prevents future leakage).
    actual_buffer_start = max(0, start_idx - buffer_size)
    
    # Slice the signal including the historical buffer
    signal_chunk = full_df.iloc[actual_buffer_start : end_idx + 1]['RRP'].values
    
    # Calculate how many points belong to the buffer
    buffer_len = start_idx - actual_buffer_start
    
    print(f"Processing {start_date} to {end_date}")
    print(f"Signal len: {len(period_df)}, Buffer len: {buffer_len}, Total input to VMD: {len(signal_chunk)}")

    # 3. Run VMD on the chunk
    # We perform a small edge pad on the chunk solely for algorithmic stability
    pad_len = 10 
    signal_pad = np.pad(signal_chunk, (0, pad_len), mode='edge')
    
    t0 = time.time()
    u, u_hat, omega = VMD(signal_pad, alpha, tau, K, DC, init, tol)
    print(f"VMD finished in {time.time() - t0:.2f}s")
    
    # Remove the small edge pad
    u = u[:, :len(signal_chunk)]
    
    # 4. Remove the Historical Buffer
    # We only want to save the data corresponding to the requested start_date -> end_date
    # So we slice off the first 'buffer_len' points
    u_valid = u[:, buffer_len:]
    
    # 5. Build DataFrame
    vmf_df = pd.DataFrame(u_valid.T, columns=[f"Mode_{i+1}" for i in range(K)])
    
    # Re-attach timestamps and original RRP from the specific period_df
    vmf_df['SETTLEMENTDATE'] = period_df['SETTLEMENTDATE'].values
    vmf_df['RRP'] = period_df['RRP'].values
    
    # Calculate Residual
    vmf_sum = vmf_df[[f"Mode_{i+1}" for i in range(K)]].sum(axis=1)
    vmf_df['Residual'] = vmf_df['RRP'] - vmf_sum
    
    # Format Date
    vmf_df['SETTLEMENTDATE'] = pd.to_datetime(vmf_df['SETTLEMENTDATE'])
    vmf_df['SETTLEMENTDATE'] = vmf_df['SETTLEMENTDATE'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return vmf_df

# -----------------------------
# 5. Loop and Save
# -----------------------------
# We use a buffer (e.g., 2000 points) so the VMD modes for Jan 1st aren't distorted
# by being at the very edge of the signal.
BUFFER_SIZE = 2000 

for period_name, (s_date, e_date) in periods.items():
    result_df = run_vmd_on_period(df, s_date, e_date, buffer_size=BUFFER_SIZE)
    
    if result_df is not None:
        csv_name = f"VMD_modes_with_residual_{period_name}.csv"
        result_df.to_csv(csv_name, index=False)
        print(f"Saved {csv_name}\n")