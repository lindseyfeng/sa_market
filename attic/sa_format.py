import pandas as pd
import os
import glob

# Input folder
input_folder = "sa_test"

# Pattern for matching files
pattern = os.path.join(input_folder, "PRICE_AND_DEMAND_2022*.csv")

# Process each file that matches the pattern
for input_file in glob.glob(pattern):
    # Construct output file name (append _30min before .csv)
    base, ext = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(input_folder, f"{base}_30min{ext}")
    
    # -----------------------------
    # 1. Read CSV
    # -----------------------------
    df = pd.read_csv(input_file)

    # -----------------------------
    # 2. Keep only rows where minute is 0 or 30 without converting
    # -----------------------------
    df['minute'] = df['SETTLEMENTDATE'].str[-5:-3].astype(int)
    df_30min = df[df['minute'].isin([0, 30])].copy()
    df_30min.drop(columns=['minute'], inplace=True)
    df_30min.reset_index(drop=True, inplace=True)

    # -----------------------------
    # 3. Save cleaned CSV
    # -----------------------------
    df_30min.to_csv(output_file, index=False)
    print(f"Done! Saved aligned 30-minute CSV as '{output_file}'")
