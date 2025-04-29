from load_data import load_and_preprocess_data
from dimensionality import compute_pca, get_pca_loadings
from clustering import (
    elbow_plot,
    kmeans_loss_heatmap,
    kmeans_silhouette_heatmap,
    gmm_silhouette_matrix,
    dbscan_coerced_heatmap,
    agglomerative_heatmap
)
from anomaly_detection import detect_anomalies, get_clean_data
from evaluation import evaluate_and_score_table
from statistics_tests import statistical_comparison, visualize_pvalue_table
from merge_pdfs import merge_figures_to_grid
from visualizations import (
    plot_combined_loading_magnitude,
    plot_individual_pc_loadings,
    plot_absolute_pc_loadings,
    plot_signed_pc_loadings,
    plot_silhouette_bars,
    plot_pca_scatter,
    plot_top4_scatter,
    print_cluster_health_distribution
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    # Load and preprocess
    df, fetal_labels, X_scaled = load_and_preprocess_data()

    # Dimensionality reduction
    X_pca, pca_model = compute_pca(X_scaled, n_components=2)

    # Clustering evaluation
    elbow_plot(X_pca)
    kmeans_loss_heatmap(X_scaled)
    kmeans_silhouette_heatmap(X_scaled)
    gmm_silhouette_matrix(X_scaled)
    dbscan_coerced_heatmap(X_scaled)
    agglomerative_heatmap(X_scaled)

    # Anomaly detection
    df_with_anomalies, method_cols = detect_anomalies(X_scaled, df)
    df_clean = get_clean_data(df_with_anomalies, method_cols)

    # Cleaned PCA
    features_for_pca = df_clean.drop(columns=method_cols + ['fetal_health', 'Anomaly_2of4'], errors='ignore').columns
    X_scaled_clean = X_scaled[df_with_anomalies['Anomaly_2of4'] == 0]
    X_pca_clean, pca_clean_model = compute_pca(X_scaled_clean, n_components=2)

    # Visualizations: PCA loading barplots
    loadings_df_clean = get_pca_loadings(pca_clean_model, features_for_pca)
    plot_combined_loading_magnitude(loadings_df_clean)
    plot_individual_pc_loadings(loadings_df_clean)

    # Silhouette by cluster
    kmeans_labels = KMeans(n_clusters=7, random_state=42).fit_predict(X_pca)
    plot_silhouette_bars(X_pca, kmeans_labels)
    plot_pca_scatter(X_pca, kmeans_labels)
    plot_absolute_pc_loadings(loadings_df_clean)
    plot_signed_pc_loadings(loadings_df_clean)

    # Top-4 loading scatter plot
    top_4_features = loadings_df_clean.abs().max(axis=1).sort_values(ascending=False).head(6).index.tolist()
    X_top4 = df[top_4_features]
    scaler = MinMaxScaler()
    X_top4_scaled = scaler.fit_transform(X_top4)
    X_pca_n = PCA(n_components=2, random_state=42).fit_transform(X_top4_scaled)
    kmeans = KMeans(n_clusters=7, random_state=42).fit(X_pca_n)
    labels = kmeans.labels_
    plot_top4_scatter(X_pca_n, labels)

    df_clusters = pd.DataFrame({
        "cluster": labels,
        "fetal_health": fetal_labels
    })
    df_clusters["health_label"] = df_clusters["fetal_health"].map({1: 'Normal', 2: 'Suspect', 3: 'Pathological'})
    print_cluster_health_distribution(df_clusters)

    # Statistical comparison
    df_stats = statistical_comparison(df_with_anomalies, fetal_labels, method_cols)
    visualize_pvalue_table(df_stats, "Table_1")

    # Composite score table
    evaluate_and_score_table(X_pca, fig_name="Table_2")

    # Merge final figure grid
    merge_figures_to_grid()

if __name__ == "__main__":
    main()
