import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# load survival data
survival_data = pd.read_csv('demotime.txt', sep='\t')
survival_data.columns = ['samplename', 'time', 'status']

# load index
metrics_data = pd.read_csv('demo.result.csv', header=None, names=['samplename', 'metric1', 'metric2'])

#mergedata
merged_data = pd.merge(survival_data, metrics_data, on='samplename')

# creat KM curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#  
def plot_km_curve(ax, data, metric_col, title):
    # groups
    median_value = data[metric_col].median()
    data['group'] = np.where(data[metric_col] >= median_value, 'High', 'Low')

    # initialization KaplanMeierFitter
    kmf = KaplanMeierFitter()

    # 
    groups = data['group'].unique()
    colors = ['red', 'blue']

    for i, group in enumerate(groups):
        group_data = data[data['group'] == group]
        kmf.fit(group_data['time'], group_data['status'], label=group)
        kmf.plot(ax=ax, ci_show=False, color=colors[i])

    #   log-rank test
    high_risk = data[data['group'] == 'High']
    low_risk = data[data['group'] == 'Low']

    results = logrank_test(high_risk['time'], low_risk['time'],
                          high_risk['status'], low_risk['status'])

    #  
    ax.set_title(f'{title}\nLog-rank p-value: {results.p_value:.4f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='upper right')

    return results.p_value

# 
p_value1 = plot_km_curve(axes[0], merged_data, 'metric1', 'Metric 1')

#  
p_value2 = plot_km_curve(axes[1], merged_data, 'metric2', 'Metric 2')

#  
plt.tight_layout()

# save PDF
plt.savefig('km_curves.pdf', format='pdf', bbox_inches='tight')
plt.show()

# print results
print(f"Metric 1 log-rank p-value: {p_value1:.4f}")
print(f"Metric 2 log-rank p-value: {p_value2:.4f}")
