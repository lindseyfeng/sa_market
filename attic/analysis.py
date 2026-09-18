import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def analyze_yearly_statistics(df, output_prefix="yearly_analysis"):
    """
    分析数据的年度统计特征
    """
    print("=" * 80)
    print("年度数据分析报告")
    print("=" * 80)
    
    # 确保日期列是datetime类型
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    df['Year'] = df['SETTLEMENTDATE'].dt.year
    df['Month'] = df['SETTLEMENTDATE'].dt.month
    df['YearMonth'] = df['SETTLEMENTDATE'].dt.to_period('M')
    
    # 1. 基本年度统计
    print("\n1. 年度RRP基本统计:")
    print("-" * 40)
    
    yearly_stats = df.groupby('Year')['RRP'].agg([
        'count', 'mean', 'std', 'min', 'max', 
        lambda x: x.quantile(0.25),  # Q1
        lambda x: x.quantile(0.50),  # Median
        lambda x: x.quantile(0.75),  # Q3
    ]).round(2)
    
    yearly_stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Max', 'Q1', 'Median', 'Q3']
    print(yearly_stats)
    
    # 2. 月度统计
    print("\n2. 月度RRP统计 (前5个月):")
    print("-" * 40)
    monthly_stats = df.groupby(['Year', 'Month'])['RRP'].agg(['mean', 'std']).round(2)
    print(monthly_stats.head(10))
    
    # 3. 极端值分析
    print("\n3. 极端值分析:")
    print("-" * 40)
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]['RRP']
        extreme_high = year_data.nlargest(5).values
        extreme_low = year_data.nsmallest(5).values
        print(f"{year}年 - 最高5值: {extreme_high}, 最低5值: {extreme_low}")
    
    # 4. VMD模态分析
    print("\n4. VMD模态年度统计:")
    print("-" * 40)
    mode_cols = [f"Mode_{i+1}" for i in range(12)] + ['Residual']
    
    yearly_mode_stats = df.groupby('Year')[mode_cols].mean().round(4)
    print("各模态年度均值:")
    print(yearly_mode_stats)
    
    # 5. 模态贡献度分析
    print("\n5. 模态对总信号的贡献度 (%):")
    print("-" * 40)
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        total_energy = (year_data['RRP'] ** 2).sum()
        mode_contributions = {}
        
        for mode in mode_cols:
            mode_energy = (year_data[mode] ** 2).sum()
            contribution_pct = (mode_energy / total_energy) * 100
            mode_contributions[mode] = contribution_pct
        
        print(f"{year}年模态贡献度:")
        for mode, contrib in sorted(mode_contributions.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {mode}: {contrib:.2f}%")
    
    # 6. 数据完整性检查
    print("\n6. 数据完整性检查:")
    print("-" * 40)
    completeness = df.groupby('Year').size()
    print("每年数据点数:")
    print(completeness)
    
    # 可视化部分
    create_visualizations(df, output_prefix)
    
    return yearly_stats

def create_visualizations(df, output_prefix):
    """
    创建数据可视化图表
    """
    print(f"\n生成可视化图表到 {output_prefix}_*.png 文件...")
    
    # 1. 年度价格分布箱线图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Year', y='RRP')
    plt.title('年度RRP分布箱线图')
    plt.xticks(rotation=45)
    
    # 2. 月度平均价格热力图
    plt.subplot(2, 2, 2)
    monthly_avg = df.groupby(['Year', 'Month'])['RRP'].mean().unstack('Year')
    sns.heatmap(monthly_avg, cmap='YlOrRd', annot=False, cbar_kws={'label': '平均RRP'})
    plt.title('月度平均RRP热力图')
    plt.xlabel('年份')
    plt.ylabel('月份')
    
    # 3. 主要模态的年度变化
    plt.subplot(2, 2, 3)
    mode_cols = [f"Mode_{i+1}" for i in range(4)]  # 只看前4个主要模态
    yearly_mode_avg = df.groupby('Year')[mode_cols].mean()
    yearly_mode_avg.plot(ax=plt.gca(), marker='o')
    plt.title('主要模态年度均值变化')
    plt.xlabel('年份')
    plt.ylabel('模态幅值')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. 价格时间序列
    plt.subplot(2, 2, 4)
    df_sample = df.iloc[::100]  # 抽样显示避免过于密集
    plt.plot(df_sample['SETTLEMENTDATE'], df_sample['RRP'], alpha=0.7, linewidth=1)
    plt.title('RRP时间序列 (抽样显示)')
    plt.xlabel('日期')
    plt.ylabel('RRP')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 年度分布对比图
    plt.figure(figsize=(15, 10))
    years = sorted(df['Year'].unique())
    
    for i, year in enumerate(years, 1):
        plt.subplot(2, 3, i)
        year_data = df[df['Year'] == year]['RRP']
        plt.hist(year_data, bins=50, alpha=0.7, density=True)
        plt.title(f'{year}年RRP分布')
        plt.xlabel('RRP')
        plt.ylabel('密度')
        
        # 添加统计信息
        stats_text = f'均值: {year_data.mean():.1f}\n标准差: {year_data.std():.1f}'
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 模态相关性热力图
    plt.figure(figsize=(12, 10))
    mode_cols = [f"Mode_{i+1}" for i in range(12)] + ['Residual', 'RRP']
    correlation_matrix = df[mode_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, cbar_kws={'shrink': .8})
    plt.title('模态与RRP的相关性矩阵')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_train_val_distributions():
    """
    比较训练集和验证集的分布差异
    """
    print("\n" + "="*80)
    print("训练集-验证集分布对比分析")
    print("="*80)
    
    # 加载你的训练和验证数据
    train_df = pd.read_csv("VMD_modes_with_residual_2018_2021.csv")
    val_df = pd.read_csv("VMD_modes_with_residual_2021_2022.csv")
    
    train_df['SETTLEMENTDATE'] = pd.to_datetime(train_df['SETTLEMENTDATE'])
    val_df['SETTLEMENTDATE'] = pd.to_datetime(val_df['SETTLEMENTDATE'])
    
    print("\n训练集统计:")
    print(f"时间范围: {train_df['SETTLEMENTDATE'].min()} 到 {train_df['SETTLEMENTDATE'].max()}")
    print(f"RRP - 均值: {train_df['RRP'].mean():.2f}, 标准差: {train_df['RRP'].std():.2f}")
    print(f"数据点数: {len(train_df)}")
    
    print("\n验证集统计:")
    print(f"时间范围: {val_df['SETTLEMENTDATE'].min()} 到 {val_df['SETTLEMENTDATE'].max()}")
    print(f"RRP - 均值: {val_df['RRP'].mean():.2f}, 标准差: {val_df['RRP'].std():.2f}")
    print(f"数据点数: {len(val_df)}")
    
    # 分布差异量化
    mean_diff = abs(train_df['RRP'].mean() - val_df['RRP'].mean())
    std_diff = abs(train_df['RRP'].std() - val_df['RRP'].std())
    mean_ratio = val_df['RRP'].mean() / train_df['RRP'].mean()
    
    print(f"\n分布差异:")
    print(f"均值差异: {mean_diff:.2f} ({mean_ratio:.2f}x)")
    print(f"标准差差异: {std_diff:.2f}")
    
    if mean_ratio > 1.5 or mean_ratio < 0.67:
        print("⚠️  警告: 训练集和验证集均值差异超过50%，可能导致模型泛化问题!")
    
    # 可视化对比
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(train_df['RRP'], bins=50, alpha=0.7, label='Train', density=True)
    plt.hist(val_df['RRP'], bins=50, alpha=0.7, label='Val', density=True)
    plt.xlabel('RRP')
    plt.ylabel('密度')
    plt.legend()
    plt.title('训练集 vs 验证集分布')
    
    plt.subplot(2, 2, 2)
    # 时间序列对比
    train_sample = train_df.iloc[::100]
    val_sample = val_df.iloc[::100]
    plt.plot(train_sample['SETTLEMENTDATE'], train_sample['RRP'], alpha=0.7, label='Train')
    plt.plot(val_sample['SETTLEMENTDATE'], val_sample['RRP'], alpha=0.7, label='Val')
    plt.xlabel('日期')
    plt.ylabel('RRP')
    plt.legend()
    plt.title('时间序列对比')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('train_val_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主执行函数
def main():
    # 加载你的数据
    print("加载数据...")
    df = pd.read_csv("VMD_modes_with_residual_2018_2021.csv")
    df = df[(df['RRP'] >= 1) & (df['RRP'] <= 981.65)].copy()
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    
    # 分析年度统计
    yearly_stats = analyze_yearly_statistics(df, "yearly_analysis")
    
    # 比较训练验证集分布
    compare_train_val_distributions()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()