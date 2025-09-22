import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

result_df = pd.read_csv('result.csv')
print("Result col names:", result_df.columns.tolist())
print("Result top rows:")
print(result_df.head())
 
result_df = result_df.rename(columns={
    result_df.columns[0]: 'Sample',
    result_df.columns[1]: 'Cellsize_Entropy',
    result_df.columns[3]: 'NC_Entropy'   
})

# load TNM data,frist row is sample name
with open('Demo.tnm.txt', 'r') as f:
    lines = f.readlines()

#  
tnm_samples = lines[0].strip().split()
#  
t_stages = lines[1].strip().split() if len(lines) > 1 else []
n_stages = lines[2].strip().split() if len(lines) > 2 else []
m_stages = lines[3].strip().split() if len(lines) > 3 else []

# 
tnm_data = []
for i, sample in enumerate(tnm_samples):
    if i < len(t_stages) and i < len(n_stages) and i < len(m_stages):
        tnm_data.append({
            'Sample': sample,
            'T_stage': t_stages[i],
            'N_stage': n_stages[i],
            'M_stage': m_stages[i]
        })

tnm_df = pd.DataFrame(tnm_data)

#  merge data
merged_df = result_df.merge(tnm_df, on='Sample', how='left')
missing_tnm = merged_df[merged_df['T_stage'].isna()]
if not missing_tnm.empty:
    print(f"Warning no tnm data: {list(missing_tnm['Sample'])}")
    merged_df = merged_df.dropna(subset=['T_stage', 'N_stage', 'M_stage'])

print(f"merged data: {len(merged_df)}")

#  
fig, axes = plt.subplots(4, 3, figsize=(20, 24))

# first row: Cellsize entropy vialon plot
sns.violinplot(x='T_stage', y='Cellsize_Entropy', data=merged_df, ax=axes[0, 0])
axes[0, 0].set_title('Cellsize entropy - T ', fontsize=14)
axes[0, 0].set_xlabel('T Stage', fontsize=12)
axes[0, 0].set_ylabel('Cellsize Entropy', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

sns.violinplot(x='N_stage', y='Cellsize_Entropy', data=merged_df, ax=axes[0, 1])
axes[0, 1].set_title('Cellsize entropy  - N ', fontsize=14)
axes[0, 1].set_xlabel('N Stage', fontsize=12)
axes[0, 1].set_ylabel('Cellsize Entropy', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

sns.violinplot(x='M_stage', y='Cellsize_Entropy', data=merged_df, ax=axes[0, 2])
axes[0, 2].set_title('Cellsize entropy - M ', fontsize=14)
axes[0, 2].set_xlabel('M Stage', fontsize=12)
axes[0, 2].set_ylabel('Cellsize Entropy', fontsize=12)
axes[0, 2].tick_params(axis='x', rotation=45)

#  
sns.boxplot(x='T_stage', y='Cellsize_Entropy', data=merged_df, ax=axes[1, 0])
axes[1, 0].set_title('Cellsize entropy T ', fontsize=14)
axes[1, 0].set_xlabel('T Stage', fontsize=12)
axes[1, 0].set_ylabel('Cellsize Entropy', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

sns.boxplot(x='N_stage', y='Cellsize_Entropy', data=merged_df, ax=axes[1, 1])
axes[1, 1].set_title('Cellsize entropy N ', fontsize=14)
axes[1, 1].set_xlabel('N Stage', fontsize=12)
axes[1, 1].set_ylabel('Cellsize Entropy', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)

sns.boxplot(x='M_stage', y='Cellsize_Entropy', data=merged_df, ax=axes[1, 2])
axes[1, 2].set_title('Cellsize entropy M', fontsize=14)
axes[1, 2].set_xlabel('M Stage', fontsize=12)
axes[1, 2].set_ylabel('Cellsize Entropy', fontsize=12)
axes[1, 2].tick_params(axis='x', rotation=45)

#  
sns.violinplot(x='T_stage', y='NC_Entropy', data=merged_df, ax=axes[2, 0])
axes[2, 0].set_title('N/C entropy T ', fontsize=14)
axes[2, 0].set_xlabel('T Stage', fontsize=12)
axes[2, 0].set_ylabel('N/C Entropy', fontsize=12)
axes[2, 0].tick_params(axis='x', rotation=45)

sns.violinplot(x='N_stage', y='NC_Entropy', data=merged_df, ax=axes[2, 1])
axes[2, 1].set_title('N/C entropy - N ', fontsize=14)
axes[2, 1].set_xlabel('N Stage', fontsize=12)
axes[2, 1].set_ylabel('N/C Entropy', fontsize=12)
axes[2, 1].tick_params(axis='x', rotation=45)

sns.violinplot(x='M_stage', y='NC_Entropy', data=merged_df, ax=axes[2, 2])
axes[2, 2].set_title('N/C entropy- M', fontsize=14)
axes[2, 2].set_xlabel('M Stage', fontsize=12)
axes[2, 2].set_ylabel('N/C Entropy', fontsize=12)
axes[2, 2].tick_params(axis='x', rotation=45)

#  
sns.boxplot(x='T_stage', y='NC_Entropy', data=merged_df, ax=axes[3, 0])
axes[3, 0].set_title('N/C entropy - T ', fontsize=14)
axes[3, 0].set_xlabel('T Stage', fontsize=12)
axes[3, 0].set_ylabel('N/C Entropy', fontsize=12)
axes[3, 0].tick_params(axis='x', rotation=45)

sns.boxplot(x='N_stage', y='NC_Entropy', data=merged_df, ax=axes[3, 1])
axes[3, 1].set_title('N/C entropy - N ', fontsize=14)
axes[3, 1].set_xlabel('N Stage', fontsize=12)
axes[3, 1].set_ylabel('N/C Entropy', fontsize=12)
axes[3, 1].tick_params(axis='x', rotation=45)

sns.boxplot(x='M_stage', y='NC_Entropy', data=merged_df, ax=axes[3, 2])
axes[3, 2].set_title('N/C entropy - M ', fontsize=14)
axes[3, 2].set_xlabel('M Stage', fontsize=12)
axes[3, 2].set_ylabel('N/C Entropy', fontsize=12)
axes[3, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('entropy_tnm_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

  
# Cellsize entropy
print("\nCellsize entropy:")
if len(merged_df['T_stage'].unique()) > 1:
    t_groups = [group['Cellsize_Entropy'].values for name, group in merged_df.groupby('T_stage')]
    if len(t_groups) >= 2:
        t_stat, t_p = stats.kruskal(*t_groups)
        print(f"T  Kruskal-Wallis TEST p: {t_p:.4f}")

if len(merged_df['N_stage'].unique()) > 1:
    n_groups = [group['Cellsize_Entropy'].values for name, group in merged_df.groupby('N_stage')]
    if len(n_groups) >= 2:
        n_stat, n_p = stats.kruskal(*n_groups)
        print(f"N  Kruskal-Wallis TEST p : {n_p:.4f}")

if len(merged_df['M_stage'].unique()) > 1:
    m_groups = [group['Cellsize_Entropy'].values for name, group in merged_df.groupby('M_stage')]
    if len(m_groups) >= 2:
        m_stat, m_p = stats.kruskal(*m_groups)
        print(f"M Kruskal-Wallis TEST p : {m_p:.4f}")
 
if len(merged_df['T_stage'].unique()) > 1:
    t_groups = [group['NC_Entropy'].values for name, group in merged_df.groupby('T_stage')]
    if len(t_groups) >= 2:
        t_stat, t_p = stats.kruskal(*t_groups)
        print(f"T  Kruskal-Wallis  TEST p : {t_p:.4f}")

if len(merged_df['N_stage'].unique()) > 1:
    n_groups = [group['NC_Entropy'].values for name, group in merged_df.groupby('N_stage')]
    if len(n_groups) >= 2:
        n_stat, n_p = stats.kruskal(*n_groups)
        print(f"N Kruskal-Wallis TEST p : {n_p:.4f}")

if len(merged_df['M_stage'].unique()) > 1:
    m_groups = [group['NC_Entropy'].values for name, group in merged_df.groupby('M_stage')]
    if len(m_groups) >= 2:
        m_stat, m_p = stats.kruskal(*m_groups)
        print(f"M  Kruskal-Wallis TEST p : {m_p:.4f}")
  
cellsize_t_stats = merged_df.groupby('T_stage')['Cellsize_Entropy'].agg(['count', 'median', 'median', 'std'])
print(cellsize_t_stats)
cellsize_t_stats = merged_df.groupby('N_stage')['Cellsize_Entropy'].agg(['count', 'median', 'median', 'std'])
print(cellsize_t_stats)
cellsize_t_stats = merged_df.groupby('M_stage')['Cellsize_Entropy'].agg(['count', 'median', 'median', 'std'])
print(cellsize_t_stats)

 
nc_t_stats = merged_df.groupby('T_stage')['NC_Entropy'].agg(['count', 'median', 'median', 'std'])
print(nc_t_stats)
nc_t_stats = merged_df.groupby('N_stage')['NC_Entropy'].agg(['count', 'median', 'median', 'std'])
print(nc_t_stats)
nc_t_stats = merged_df.groupby('M_stage')['NC_Entropy'].agg(['count', 'median', 'median', 'std'])
print(nc_t_stats)
 
merged_df.to_csv('entropy_tnm_merged_results.csv', index=False)
print("\nRESULTS saved: entropy_tnm_merged_results.csv 和 entropy_tnm_analysis.png")
