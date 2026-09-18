import pandas as pd
import glob
import os

# -----------------------------
# 1. Folder containing all CSVs
# -----------------------------
folder_path = "sa_test"

def extract_year_month(f):
    base = os.path.basename(f).replace(".csv", "")
    parts = base.split("_")
    for p in parts:
        if "-" in p:
            return p
    return "0000-00"

csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")),
                   key=lambda x: extract_year_month(x))
print(f"Found {len(csv_files)} CSV files.")

# -----------------------------
# 2. Read and combine all CSVs, keep track of source file
# -----------------------------
df_list = []
for f in csv_files:
    temp_df = pd.read_csv(f)
    temp_df['source_file'] = os.path.basename(f)  # add column with file name
    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)
print(f"Total rows after concatenation: {len(df_all)}")

# -----------------------------
# 3. Convert timestamps
# -----------------------------
df_all['SETTLEMENTDATE'] = pd.to_datetime(df_all['SETTLEMENTDATE'], errors='coerce')

# Identify rows that will be dropped
rows_to_drop = df_all[df_all['SETTLEMENTDATE'].isna()]

print("Rows that will be dropped due to invalid/missing SETTLEMENTDATE:")
print(rows_to_drop[['REGION','SETTLEMENTDATE','TOTALDEMAND','RRP','PERIODTYPE','source_file']])

# Drop invalid timestamps
df_all = df_all.dropna(subset=['SETTLEMENTDATE'])
df_all.sort_values('SETTLEMENTDATE', inplace=True)
df_all.reset_index(drop=True, inplace=True)

# -----------------------------
# 4. Filter for Jan 1, 2018 to Dec 31, 2021
# -----------------------------
df_train = df_all[(df_all['SETTLEMENTDATE'] >= "2018-01-01") &
                  (df_all['SETTLEMENTDATE'] <= "2021-12-31")].copy()
df_train.sort_values('SETTLEMENTDATE', inplace=True)
df_train.reset_index(drop=True, inplace=True)

print(f"Total rows in training period: {len(df_train)}")

# -----------------------------
# 5. Ensure 30-minute spacing
# -----------------------------
df_train['time_diff'] = df_train['SETTLEMENTDATE'].diff()

# Identify problematic rows (time difference != 30 min)
problem_rows = df_train[df_train['time_diff'] != pd.Timedelta(minutes=30)]

print(f"Total problematic rows: {len(problem_rows)}")
if len(problem_rows) > 0:
    print("Sample problematic rows (with source file):")
    print(problem_rows[['REGION','SETTLEMENTDATE','TOTALDEMAND','RRP','PERIODTYPE','source_file','time_diff']].head(20))
    problem_rows.to_csv("problematic_rows.csv", index=False)
    print("Problematic rows saved to 'problematic_rows.csv'")
else:
    print("✅ All rows are exactly 30 minutes apart.")

# -----------------------------
# 6. Save combined CSV
# -----------------------------
df_train.drop(columns=['time_diff'], inplace=True)
df_train.to_csv("SA_prices_combined_2021_2022.csv", index=False)
print("Combined CSV saved as 'SA_prices_combined_2021_2022.csv'")
