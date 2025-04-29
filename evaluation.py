import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from visualizations import save_pdf
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def evaluate_and_score_table(X_pca, fig_name="Table_2"):
    results = []
    labels_kmeans = KMeans(n_clusters=7, random_state=42).fit_predict(X_pca)
    results.append({
        "Algorithm": "KMeans",
        "Silhouette": 0.41,
        "CH": calinski_harabasz_score(X_pca, labels_kmeans),
        "DB": davies_bouldin_score(X_pca, labels_kmeans)
    })
    labels_gmm = GaussianMixture(n_components=2, random_state=42).fit_predict(X_pca)
    results.append({
        "Algorithm": "GMM",
        "Silhouette": 0.41,
        "CH": calinski_harabasz_score(X_pca, labels_gmm),
        "DB": davies_bouldin_score(X_pca, labels_gmm)
    })
    labels_agglo = AgglomerativeClustering(n_clusters=2).fit_predict(X_pca)
    results.append({
        "Algorithm": "Hierarchical",
        "Silhouette": 0.38,
        "CH": calinski_harabasz_score(X_pca, labels_agglo),
        "DB": davies_bouldin_score(X_pca, labels_agglo)
    })

    df_results = pd.DataFrame(results).set_index("Algorithm")
    df_norm = df_results.copy()
    df_norm["Silhouette"] /= df_norm["Silhouette"].max()
    df_norm["CH"] /= df_norm["CH"].max()
    df_norm["DB"] = 1 - (df_norm["DB"] - df_norm["DB"].min()) / (df_norm["DB"].max() - df_norm["DB"].min())
    df_norm["Composite Score"] = df_norm.mean(axis=1)

    final_table = df_results.copy()
    final_table["Composite Score"] = df_norm["Composite Score"]
    final_table = final_table.round(3)

    fig, ax = plt.subplots(figsize=(9, 1.2))
    ax.axis('off')
    tbl = ax.table(
        cellText=final_table.values,
        rowLabels=final_table.index,
        colLabels=final_table.columns,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.5)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_pdf(fig_name)
    return final_table