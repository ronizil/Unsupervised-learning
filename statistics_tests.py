import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal
from visualizations import save_pdf


def summary_tests(labels, true_labels):
    mask = labels != -1
    lbl = labels[mask]
    arr = true_labels[mask]
    unique_lbls = np.unique(lbl)
    if len(unique_lbls) < 2:
        return np.nan, np.nan
    groups = [arr[lbl == g] for g in unique_lbls]
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        try:
            p_anova = f_oneway(*groups).pvalue
        except Exception:
            p_anova = np.nan
        try:
            p_kw = kruskal(*groups).pvalue
        except Exception:
            p_kw = np.nan
    else:
        p_anova, p_kw = np.nan, np.nan
    return p_anova, p_kw


def statistical_comparison(df, fetal_labels, method_cols):
    y_true = fetal_labels[df['Anomaly_2of4'] == 0].reset_index(drop=True)
    X_scaled_clean = StandardScaler().fit_transform(
        df[df['Anomaly_2of4'] == 0].drop(columns=['Anomaly_2of4'] + method_cols, errors='ignore')
    )
    X_pca_clean = PCA(n_components=2, random_state=42).fit_transform(X_scaled_clean)

    labels_dict = {
        "KMeans": KMeans(n_clusters=7, random_state=42).fit_predict(X_pca_clean),
        "GMM": GaussianMixture(n_components=2, random_state=42).fit(X_pca_clean).predict(X_pca_clean),
        "Hierarchical": AgglomerativeClustering(n_clusters=2).fit_predict(X_pca_clean),
    }

    rows = []
    for name, lbl in labels_dict.items():
        pa, pk = summary_tests(lbl, y_true.values)
        rows.append({
            "Algorithm": name,
            "ANOVA p-value": pa,
            "Kruskal p-value": pk
        })
    df_stats = pd.DataFrame(rows).set_index("Algorithm")
    return df_stats


def visualize_pvalue_table(df, fig_name="Table_1"):
    formatted_df = df.applymap(lambda x: f"{x:.2e}" if pd.notnull(x) else "NaN")
    fig, ax = plt.subplots(figsize=(9, 1.2))
    ax.axis('off')
    tbl = ax.table(
        cellText=formatted_df.values,
        rowLabels=formatted_df.index,
        colLabels=formatted_df.columns,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.3)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_pdf(fig_name)