import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller
# -----------------------------
# 1. Load and clean data
# -----------------------------
df = pd.read_csv("SA_prices_combined_2018_2022.csv")
df = df[df['RRP'] >= 1].copy()
df = df[df['RRP'] <= 1000].copy()
print(len(df))

df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], errors='coerce')
df = df.dropna(subset=['SETTLEMENTDATE'])
df.sort_values('SETTLEMENTDATE', inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract time features
df['Year'] = df['SETTLEMENTDATE'].dt.year
df['Month'] = df['SETTLEMENTDATE'].dt.month_name()
df['Weekday'] = df['SETTLEMENTDATE'].dt.day_name()
df['Hour'] = df['SETTLEMENTDATE'].dt.hour

# -----------------------------
# 2. Full time series plot
# -----------------------------
plt.figure(figsize=(15,5))
plt.plot(df['SETTLEMENTDATE'], df['RRP'], color='blue', linewidth=0.5)
plt.title("Electricity Price (RRP) Trend: Jan 2018 – Dec 2021")
plt.xlabel("Date")
plt.ylabel("RRP (AUD/MWh)")
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Yearly trend plots
# -----------------------------
years = sorted(df['Year'].unique())
plt.figure(figsize=(15,10))
for i, year in enumerate(years, 1):
    plt.subplot(len(years),1,i)
    df_year = df[df['Year'] == year]
    plt.plot(df_year['SETTLEMENTDATE'], df_year['RRP'], linewidth=0.5)
    plt.title(f"Electricity Price Trend - {year}")
    plt.ylabel("RRP")
    if i == len(years):
        plt.xlabel("Date")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Boxplot helper function
# -----------------------------
def boxplot_with_mean(df, x_col, y_col, order=None, title="", figsize=(12,6), rotation=0):
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_col, y=y_col, data=df, order=order, showfliers=True, color='skyblue')
    
    # Compute mean for each group
    means = df.groupby(x_col)[y_col].mean()
    if order:
        means = means.reindex(order)
    
    # Overlay mean on figure
    for i, mean_val in enumerate(means):
        plt.text(i, mean_val + 5, f"{mean_val:.1f}", color='red', ha='center', fontweight='bold')
    
    plt.title(title)
    plt.ylabel("RRP (AUD/MWh)")
    plt.xlabel(x_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. Yearly boxplot with mean
# -----------------------------
boxplot_with_mean(df, 'Year', 'RRP', order=years, title="Electricity Price Distribution per Year (Mean in Red)")

# -----------------------------
# 6. Month boxplot with mean
# -----------------------------
months_order = ['January','February','March','April','May','June','July','August',
                'September','October','November','December']
boxplot_with_mean(df, 'Month', 'RRP', order=months_order, title="Electricity Price Distribution per Month (Mean in Red)", rotation=45)

# -----------------------------
# 7. Weekday boxplot with mean
# -----------------------------
weekdays_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
boxplot_with_mean(df, 'Weekday', 'RRP', order=weekdays_order, title="Electricity Price Distribution per Weekday (Mean in Red)")

# -----------------------------
# 8. Hour boxplot with mean
# -----------------------------
hours_order = list(range(24))
boxplot_with_mean(df, 'Hour', 'RRP', order=hours_order, title="Electricity Price Distribution per Hour of Day (Mean in Red)")



# import pandas as pd
# import numpy as np
# from scipy.stats import skew, kurtosis, jarque_bera
# from statsmodels.tsa.stattools import adfuller

# # Read your dataset
# df = pd.read_csv("SA_prices_combined_2018_2021.csv")  # replace with your CSV
# df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], errors='coerce')
# df = df.dropna(subset=['SETTLEMENTDATE'])
# df = df[df['RRP'] >= 1].copy()
# df = df[df['RRP'] <= 981.65].copy()

# print(len(df))

# # Focus on the electricity price column
price_series = df['RRP']

# -----------------------------
# 1. Basic descriptive stats
# -----------------------------
mean_val = price_series.mean()
max_val = price_series.max()
min_val = price_series.min()
std_val = price_series.std()

# -----------------------------
# 2. Distribution shape
# -----------------------------
skew_val = skew(price_series, bias=False)
kurt_val = kurtosis(price_series, fisher=True, bias=False)  # Fisher=True gives excess kurtosis

# -----------------------------
# 3. Jarque-Bera test
# -----------------------------
jb_stat, jb_pval = jarque_bera(price_series)

# -----------------------------
# 4. Augmented Dickey-Fuller test
# -----------------------------
adf_result = adfuller(price_series)
adf_stat = adf_result[0]
adf_pval = adf_result[1]

# -----------------------------
# 5. Print summary
# -----------------------------
print("Electricity Price (RRP) Statistical Summary")
print("-------------------------------------------------")
print(f"Mean: {mean_val:.2f}")
print(f"Max: {max_val:.2f}")
print(f"Min: {min_val:.2f}")
print(f"Std: {std_val:.2f}")
print(f"Skew: {skew_val:.2f}")
print(f"Kurtosis: {kurt_val:.2f}")
print(f"Jarque-Bera Stat: {jb_stat:.2f}, p-value: {jb_pval:.4f}")
print(f"ADF Stat: {adf_stat:.2f}, p-value: {adf_pval:.4f}")
