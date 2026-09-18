import pandas as pd
import glob
import os

# -----------------------------
# 1. Folders containing CSVs
# -----------------------------
folders = ["sa_dataset", "sa_test"]

# Collect all CSV files from all folders
csv_files = []
for folder in folders:
    csv_files += glob.glob(os.path.join(folder, "*.csv"))

print(f"Found {len(csv_files)} CSV files across all folders.")

# Optional: sort files by year-month in filename
def extract_year_month(f):
    base = os.path.basename(f).replace(".csv", "")
    parts = base.split("_")
    for p in parts:
        if "-" in p:  # likely YYYY-MM
            return p
    return "0000-00"

csv_files = sorted(csv_files, key=lambda x: extract_year_month(x))

# -----------------------------
# 2. Read and combine CSVs
# -----------------------------
df_list = []
for f in csv_files:
    temp_df = pd.read_csv(f)
    temp_df['source_file'] = os.path.basename(f)  # track source file
    df_list.append(temp_df)

# Concatenate all CSVs and reset index
df_all = pd.concat(df_list, ignore_index=True)
print(f"Total rows after concatenation: {len(df_all)}")

# -----------------------------
# 3. Convert timestamps safely
# -----------------------------
df_all['SETTLEMENTDATE'] = pd.to_datetime(df_all['SETTLEMENTDATE'], errors='coerce')

# Drop invalid timestamps
rows_to_drop = df_all[df_all['SETTLEMENTDATE'].isna()]
if not rows_to_drop.empty:
    print("Rows dropped due to invalid/missing SETTLEMENTDATE:")
    print(rows_to_drop[['REGION','SETTLEMENTDATE','TOTALDEMAND','RRP','PERIODTYPE','source_file']])

df_all = df_all.dropna(subset=['SETTLEMENTDATE'])

# -----------------------------
# 4. Remove duplicate timestamps
# -----------------------------
df_all = df_all.drop_duplicates(subset=['SETTLEMENTDATE'])
df_all = df_all.sort_values('SETTLEMENTDATE').reset_index(drop=True)

# -----------------------------
# 5. Filter for 2018-01-01 to 2022-12-31
# -----------------------------
df_all = df_all[(df_all['SETTLEMENTDATE'] >= "2018-01-01") &
                (df_all['SETTLEMENTDATE'] <= "2022-12-31")].copy()
df_all.reset_index(drop=True, inplace=True)
print(f"Total rows after filtering 2018-2022: {len(df_all)}")

# -----------------------------
# 6. Check 30-min spacing using seconds
# -----------------------------
df_all['time_diff_sec'] = df_all['SETTLEMENTDATE'].diff().dt.total_seconds()

# Only flag differences not equal to 1800 seconds (30 min) with 1-second tolerance
tolerance = 1
problem_rows = df_all[(df_all['time_diff_sec'].notna()) & 
                      (abs(df_all['time_diff_sec'] - 1800) > tolerance)]

print(f"Total problematic rows: {len(problem_rows)}")
if len(problem_rows) > 0:
    print("Sample problematic rows:")
    print(problem_rows[['REGION','SETTLEMENTDATE','RRP','source_file','time_diff_sec']].head(20))
    problem_rows.to_csv("problematic_rows_combined.csv", index=False)
    print("Problematic rows saved to 'problematic_rows_combined.csv'")
else:
    print("✅ All rows are exactly 30 minutes apart.")

# -----------------------------
# 7. Save final combined CSV
# -----------------------------
df_all.drop(columns=['time_diff_sec'], inplace=True)
output_file = "SA_prices_combined_2018_2022.csv"
df_all.to_csv(output_file, index=False)
print(f"Combined CSV saved as '{output_file}'")
