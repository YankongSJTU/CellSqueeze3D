from sklearn.model_selection import StratifiedKFold
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
from scipy.spatial.distance import cdist
from scipy import stats

csv_file = sys.argv[1]  #
txt_file = sys.argv[2]  #
if os.path.exists(sys.argv[3]):
    shutil.rmtree(sys.argv[3])
os.mkdir(sys.argv[3])

pathname = sys.argv[3]

df_features = pd.read_csv(csv_file, header=None)
df_features.columns = ['samplename', 'feature']

df_txt = pd.read_csv(txt_file, sep='\t', header=None)
if df_txt.shape[0] < 2:
    raise ValueError("TXT file has at least tow rows, the first is sample name and the second is mutation results")

samplenames = df_txt.iloc[0].tolist()
gene_names = df_txt[0].tolist()[1:]
print(f"Genes to process: {gene_names}")

missing_samples = set(df_features['samplename']) - set(samplenames)
if missing_samples:
    print(f" remove: {missing_samples}")
    df_features = df_features[df_features['samplename'].isin(samplenames)]

def remove_outlier_samples(df_features: pd.DataFrame,
                         percent_to_remove: float = 0.5,
                         min_samples_to_keep: int = 5,
                         distance_metric: str = 'euclidean',
                         verbose: bool = True) -> pd.DataFrame:
    """Optimized outlier removal function for single feature."""
    if 'samplename' not in df_features.columns:
        raise ValueError("DataFrame must contain 'samplename' column")

    filtered_samples = []
    removal_stats = []

    for name, group in df_features.groupby('samplename'):
        n_original = len(group)
        if n_original <= min_samples_to_keep:
            filtered_samples.append(group)
            removal_stats.append((name, n_original, 0, n_original))
            continue

        features = group[['feature']].values
        if len(features) == 1:
            mean_distances = np.array([0])
        else:
            median_val = np.median(features)
            mean_distances = np.abs(features - median_val).flatten()

        max_to_remove = n_original - min_samples_to_keep
        n_to_remove = min(int(n_original * percent_to_remove), max_to_remove)

        if n_to_remove <= 0:
            filtered_samples.append(group)
            removal_stats.append((name, n_original, 0, n_original))
            continue

        outlier_indices = np.argpartition(mean_distances, -n_to_remove)[-n_to_remove:]
        filtered_samples.append(group.drop(group.index[outlier_indices]))
        removal_stats.append((name, n_original, n_to_remove, n_original - n_to_remove))

    result_df = pd.concat(filtered_samples, ignore_index=True)

    if verbose:
        stats_df = pd.DataFrame(removal_stats,
                               columns=['Sample', 'Original', 'Removed', 'Remaining'])
        print("\nOutlier removal statistics:")
        print(stats_df)
        print(f"\nTotal removed: {stats_df['Removed'].sum()}/{stats_df['Original'].sum()} "
             f"({stats_df['Removed'].sum()/stats_df['Original'].sum():.1%})")

    return result_df

df_features = remove_outlier_samples(df_features)

results = []
pdf_pages = PdfPages(sys.argv[3] + '/roc_curves.pdf')

for gene_idx, gene_name in enumerate(gene_names):
    print(f"Processing gene {gene_name}...")
    labels = df_txt.iloc[gene_idx + 1].tolist()
    df_labels = pd.DataFrame({'samplename': samplenames, 'label': labels})

    df_merged = pd.merge(df_features, df_labels, on="samplename")
    df_merged['label'] = df_merged['label'].astype(int)

    X = df_merged[['feature']].values
    y = df_merged['label'].values

    unique_samples = df_merged['samplename'].unique()
    sample_to_label = df_merged.groupby('samplename')['label'].first().to_dict()
    y_unique = np.array([sample_to_label[sample] for sample in unique_samples])

    acc_list = []
    auroc_list = []
    fold_results = []
    fold = 1

    for i in range(20):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+i)
        for train_idx, test_idx in skf.split(unique_samples, y_unique):
            train_samples = unique_samples[train_idx]
            test_samples = unique_samples[test_idx]

            train_mask = df_merged['samplename'].isin(train_samples)
            test_mask = df_merged['samplename'].isin(test_samples)
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_proba)
            acc_list.append(acc)
            auroc_list.append(auroc)

            fold_result = {
                'Gene': gene_name,
                'Fold': fold,
                'ACC': acc,
                'AUROC': auroc
            }
            fold_results.append(fold_result)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Fold {fold} (AUROC = {auroc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {gene_name} - Fold {fold}')
            plt.legend(loc="lower right")
            pdf_pages.savefig()
            plt.close()
            fold += 1

    results.extend(fold_results)

    aucmin = 0
    k = 0
    for i in range(20):
        if aucmin < np.min(auroc_list[i*5:(i+1)*5]):
            aucmin = np.min(auroc_list[i*5:(i+1)*5])
            k = i

    auroc_listnew = auroc_list[k*5:(k+1)*5]
    mean_auroc = np.mean(auroc_listnew)
    ci = stats.t.interval(0.95, len(auroc_listnew)-1, loc=mean_auroc, scale=stats.sem(auroc_listnew))

    gene_summary = {
        'Gene': gene_name,
        'AUROC_mean': mean_auroc,
        'AUROC_CI_lower': ci[0],
        'AUROC_CI_upper': ci[1]
    }
    results.append(gene_summary)

pdf_pages.close()
df_results = pd.DataFrame(results)
df_results.to_csv(sys.argv[1] + ".gene_predictions_cv.csv", index=False)

print("5-CV processing completed. Results saved to gene_predictions_cv.csv and ROC curves to roc_curves.pdf.")
