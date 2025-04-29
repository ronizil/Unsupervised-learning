import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_samples
import warnings

# Suppress seaborn palette warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === PDF saving ===
OUTDIR = Path("figures_pdf")
OUTDIR.mkdir(exist_ok=True)

def save_pdf(name: str, dpi: int = 300, close: bool = True) -> None:
    path = OUTDIR / f"{name}.pdf"
    plt.savefig(path, format="pdf", dpi=dpi,
                bbox_inches='tight', pad_inches=0)
    print(f"✓ PDF saved to → {path}")
    if close:
        plt.close()

# === Barplot of PCA loadings ===
def plot_loading_bars(loadings_df, component: str, signed=True, fig_name=""):
    plt.figure(figsize=(10, 6))
    sorted_vals = loadings_df[component].sort_values(ascending=True)
    if not signed:
        sorted_vals = loadings_df[component].abs().sort_values(ascending=True)
    palette = "coolwarm" if signed else "viridis"
    sns.barplot(x=sorted_vals.values, y=sorted_vals.index, palette=palette)
    if signed:
        plt.axvline(0, color='black', linewidth=1)
        plt.title(f"PCA Loadings on {component} (Signed)")
        plt.xlabel("Loading Value")
    else:
        plt.xlabel("Absolute loading")
    plt.ylabel("Feature")
    plt.tight_layout()
    save_pdf(fig_name)

# === Silhouette Plot ===
def plot_silhouette_bars(X_embedded, labels, fig_name="fig3A"):
    fig, ax = plt.subplots(figsize=(8, 6))
    silhouette_vals = silhouette_samples(X_embedded, labels)
    y_lower = 10
    n_clusters = len(set(labels))
    colors = sns.color_palette("Set2", n_clusters)
    for i in range(n_clusters):
        cluster_sil_vals = silhouette_vals[labels == i]
        cluster_sil_vals.sort()
        size = cluster_sil_vals.shape[0]
        y_upper = y_lower + size
        color = colors[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals,
                         facecolor=color, edgecolor=color)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10
    ax.set_xlabel("Silhouette score")
    ax.set_ylabel("Cluster index")
    ax.set_xlim([-0.2, 1])
    ax.set_yticks([])
    plt.tight_layout()
    save_pdf(fig_name)
    plt.close()

# === PCA scatter plot with cluster labels ===
def plot_pca_scatter(X_pca, labels, fig_name="fig3B", n_clusters=7):
    palette = sns.color_palette("tab10", n_clusters)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=palette, s=50)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    save_pdf(fig_name)

# === PCA loading magnitude combined plot (|loading| for both PCs) ===
def plot_combined_loading_magnitude(loadings_df, fig_name="fig2A"):
    magnitude = loadings_df.abs().max(axis=1).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=magnitude.values,
        y=magnitude.index,
        hue=magnitude.index,
        dodge=False,
        palette="viridis",
        legend=False
    )
    plt.xlabel("|loading| both PCs")
    plt.tight_layout()
    save_pdf(fig_name)

# === Individual signed PCA loading plots for PC1 and PC2 ===
def plot_individual_pc_loadings(loadings_df, fig_prefix="fig2"):
    for i, pc in enumerate(['PC1', 'PC2']):
        plt.figure(figsize=(10, 6))
        sorted_vals = loadings_df[pc].sort_values(ascending=True)
        sns.barplot(x=sorted_vals.values, y=sorted_vals.index, palette="coolwarm")
        plt.axvline(0, color='black', linewidth=1)
        plt.xlabel(f"Loading Value ({pc})")
        plt.ylabel("Feature")
        plt.tight_layout()
        save_pdf(f"{fig_prefix}{chr(66 + i)}")

# === Absolute PCA loading plots for PC1 and PC2 ===
def plot_absolute_pc_loadings(loadings_df, fig_prefix="fig3"):
    for i, pc in enumerate(['PC1', 'PC2']):
        plt.figure(figsize=(10, 6))
        sorted_vals = loadings_df[pc].abs().sort_values(ascending=True)
        sns.barplot(x=sorted_vals.values, y=sorted_vals.index, palette="viridis")
        plt.xlabel("Absolute loading")
        plt.ylabel("Feature")
        plt.tight_layout()
        save_pdf(f"{fig_prefix}{chr(69 + i)}")

# === Signed PCA loading plots for PC1 and PC2 (fig3C and fig3D) ===
def plot_signed_pc_loadings(loadings_df_clean):
    for i, pc in enumerate(['PC1', 'PC2']):
        plt.figure(figsize=(10, 6))
        sorted_vals = loadings_df_clean[pc].sort_values(ascending=True)
        sns.barplot(
            x=sorted_vals.values,
            y=sorted_vals.index,
            palette="coolwarm"
        )
        plt.axvline(0, color='black', linewidth=1)
        plt.title(f"PCA Loadings on {pc} (Signed)")
        plt.xlabel("Loading Value")
        plt.ylabel("Feature")
        plt.tight_layout()
        save_pdf(f"fig3{'C' if i == 0 else 'D'}")

# === Top-4 features PCA scatter plot ===
def plot_top4_scatter(X_pca_n, labels, fig_name="fig4A"):
    palette = sns.color_palette("tab10", 7)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca_n[:, 0], y=X_pca_n[:, 1], hue=labels, palette=palette, s=50)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    save_pdf(fig_name)

# === Print top-4 cluster label distribution ===
def print_cluster_health_distribution(df_clusters):
    dist_all = (
        df_clusters
        .groupby("cluster")["health_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    ).round(2).applymap(lambda x: f"{x:.2f}%")
    print("\n=== KMeans (7 clusters) on Top 4 Features — All Points ===")
    print(dist_all)
