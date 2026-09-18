import pandas as pd
import numpy as np
from vmdpy import VMD
from joblib import Parallel, delayed
import time
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_CSV = "SA_prices_combined_2018_2022.csv"
OUTPUT_FILE = "vmdnet_data_train.npz"

LOOKBACK_WINDOW = 336   # P in the paper (e.g., 2 weeks of hours)
FORECAST_HORIZON = 96   # F in the paper (e.g., predict next 4 days)
K = 10                  # Number of modes
ALPHA = 2000            # Bandwidth constraint
TAU = 0.
DC = 0
INIT = 1
TOL = 1e-7

N_JOBS = -1             # Use all CPU cores

# ---------------------------------------------------------
# WORKER FUNCTION (Runs on 1 CPU Core)
# ---------------------------------------------------------
def process_single_window(current_idx, signal, window_size, forecast_horizon, K, alpha, tau, DC, init, tol):
    """
    1. Slices history [t-P : t]
    2. Slices target [t : t+F]
    3. Runs VMD on history
    4. Returns (Modes, Frequencies, Target)
    """
    # 1. Define Slices
    start_idx = current_idx - window_size
    end_idx = current_idx
    
    # Safety Check
    if start_idx < 0 or end_idx + forecast_horizon > len(signal):
        return None

    # Get Input Chunk (History)
    chunk = signal[start_idx : end_idx]
    
    # Get Target Chunk (Future)
    target = signal[end_idx : end_idx + forecast_horizon]

    # 2. VMD Pre-processing (Padding)
    # We use 'reflect' to minimize boundary effects at the end of the signal
    pad_len = 100 
    chunk_pad = np.pad(chunk, (0, pad_len), mode='reflect')

    try:
        # 3. Run VMD
        # u: (K, window + pad)
        # omega: (K, iterations)
        u, _, omega = VMD(chunk_pad, alpha, tau, K, DC, init, tol)
        
        # 4. Post-processing
        # Remove the padding to restore original window size
        u_valid = u[:, :window_size]  # Shape: (K, P)
        
        # Get the final converged center frequencies
        omega_final = omega[-1, :]    # Shape: (K,)
        
        return (u_valid, omega_final, target)

    except Exception as e:
        return None

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Ensure sorted and clean
    df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)
    signal = df['RRP'].values.astype(np.float32) # Assuming column is RRP

    # Define indices to run
    # We start at LOOKBACK_WINDOW and end such that we have room for the forecast
    start_index = LOOKBACK_WINDOW
    end_index = len(signal) - FORECAST_HORIZON
    
    # Create list of indices to process
    indices_to_run = range(start_index, end_index)
    
    print(f"Starting VMD Sample-wise Decomposition on {len(indices_to_run)} windows...")
    print(f"Config: K={K}, Alpha={ALPHA}, Window={LOOKBACK_WINDOW}")

    t0 = time.time()
    
    # Run Parallel Processing
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_single_window)(
            i, signal, LOOKBACK_WINDOW, FORECAST_HORIZON, K, ALPHA, TAU, DC, INIT, TOL
        ) for i in indices_to_run
    )
    
    # Filter out None results (errors)
    results = [r for r in results if r is not None]
    
    print(f"Decomposition finished in {time.time() - t0:.2f} seconds.")
    print("Stacking arrays...")

    # Unpack results
    # Each result is (modes, freqs, target)
    all_modes = np.stack([r[0] for r in results])    # Shape: (N, K, P)
    all_freqs = np.stack([r[1] for r in results])    # Shape: (N, K)
    all_targets = np.stack([r[2] for r in results])  # Shape: (N, F)

    print(f"Final Data Shapes:")
    print(f"Modes (X): {all_modes.shape} (Samples, Modes, Time)")
    print(f"Freqs (W): {all_freqs.shape} (Samples, Modes)")
    print(f"Targets (Y): {all_targets.shape} (Samples, Horizon)")

    # Save compressed
    print(f"Saving to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE, 
        modes=all_modes, 
        frequencies=all_freqs, 
        targets=all_targets
    )
    print("Done.")