import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from config import CLUSTER_RANGE, DIMENSION_RANGE, EPS_VALUES
from visualizations import save_pdf


def elbow_plot(X_pca):
    loss_list = []
    initial_loss = None
    elbow_k = None

    for k in CLUSTER_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        loss = kmeans.inertia_
        loss_list.append(loss)
        if initial_loss is None:
            initial_loss = loss
        if loss <= 0.5 * initial_loss and elbow_k is None:
            elbow_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(CLUSTER_RANGE, loss_list, label='PCA dim=2')
    if elbow_k:
        plt.axvline(elbow_k, color='red', linestyle='--', label=f'Elbow k={elbow_k}')
    plt.xlabel("Number of Clusters")
    plt.ylabel("KMeans Loss")
    plt.legend()
    plt.grid(True)
    save_pdf("fig1B")


def kmeans_loss_heatmap(X_scaled):
    loss_results = {}
    for d in DIMENSION_RANGE:
        X_pca = PCA(n_components=d).fit_transform(X_scaled)
        losses = []
        for k in CLUSTER_RANGE:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(X_pca)
            losses.append(model.inertia_)
        loss_results[f"{d}D"] = losses

    loss_df = pd.DataFrame(loss_results, index=CLUSTER_RANGE).T
    plt.figure(figsize=(10, 6))
    sns.heatmap(loss_df, fmt=".0f", cmap="Blues", annot=False, cbar_kws={'label': 'Loss'})
    plt.xlabel("Number of Clusters")
    plt.ylabel("PCA Dimensions")
    save_pdf("fig1A")


def kmeans_silhouette_heatmap(X_scaled):
    silhouette_results = {}
    for d in DIMENSION_RANGE:
        X_pca = PCA(n_components=d).fit_transform(X_scaled)
        sils = []
        for k in CLUSTER_RANGE:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(X_pca)
            sils.append(silhouette_score(X_pca, labels))
        silhouette_results[f"{d}D"] = sils

    sil_df = pd.DataFrame(silhouette_results, index=CLUSTER_RANGE).T
    plt.figure(figsize=(10, 6))
    sns.heatmap(sil_df, fmt=".2f", cmap="Blues", annot=False, cbar_kws={'label': 'Silhouette Score'})
    plt.xlabel("Number of Clusters")
    plt.ylabel("PCA Dimensions")
    save_pdf("fig1C")


def gmm_silhouette_matrix(X_scaled):
    matrix = np.zeros((len(DIMENSION_RANGE), len(CLUSTER_RANGE)))
    for i, n_dims in enumerate(DIMENSION_RANGE):
        X_reduced = PCA(n_components=n_dims).fit_transform(X_scaled)
        for j, n_clusters in enumerate(CLUSTER_RANGE):
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gmm.fit_predict(X_reduced)
            try:
                score = silhouette_score(X_reduced, labels)
            except:
                score = np.nan
            matrix[i, j] = score

    df = pd.DataFrame(matrix,
                      index=[f"{d}D" for d in DIMENSION_RANGE],
                      columns=[str(k) for k in CLUSTER_RANGE])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=False, cmap="Purples", fmt=".2f",
                cbar_kws={'label': 'Silhouette Score'})
    plt.xlabel("Number of Clusters")
    plt.ylabel("PCA Dimensions")
    plt.tight_layout()
    save_pdf("fig1D")


def dbscan_coerced_heatmap(X_scaled):
    attempts = []
    for n_dims in DIMENSION_RANGE:
        X_reduced = PCA(n_components=n_dims).fit_transform(X_scaled)
        for eps in EPS_VALUES:
            db = DBSCAN(eps=eps, min_samples=5)
            labels = db.fit_predict(X_reduced)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                try:
                    score = silhouette_score(X_reduced, labels)
                except:
                    score = np.nan
            else:
                score = np.nan
            attempts.append((n_dims, eps, n_clusters, score))

    score_matrix = pd.DataFrame(
        np.zeros((len(DIMENSION_RANGE), len(CLUSTER_RANGE))),
        index=[f"{d}D" for d in DIMENSION_RANGE],
        columns=[str(k) for k in CLUSTER_RANGE]
    )

    for n_dims in DIMENSION_RANGE:
        relevant_rows = [row for row in attempts if row[0] == n_dims]
        for k in CLUSTER_RANGE:
            best_diff = float('inf')
            best_score = -1.0
            found_any = False
            for (_, eps, n_clusters, score) in relevant_rows:
                if np.isnan(score): continue
                diff = abs(n_clusters - k)
                if diff < best_diff or (diff == best_diff and score > best_score):
                    best_diff = diff
                    best_score = score
                    found_any = True
            score_matrix.loc[f"{n_dims}D", str(k)] = best_score if found_any else 0.0

    plt.figure(figsize=(10, 6))
    sns.heatmap(score_matrix, fmt=".2f", cmap="Greens", annot=False,
                cbar_kws={'label': 'Silhouette Score'})
    plt.xlabel("Number of Clusters")
    plt.ylabel("PCA Dimensions")
    plt.tight_layout()
    save_pdf("fig1E")


def agglomerative_heatmap(X_scaled):
    matrix = np.zeros((len(DIMENSION_RANGE), len(CLUSTER_RANGE)))
    for i, d in enumerate(DIMENSION_RANGE):
        X_reduced = PCA(n_components=d).fit_transform(X_scaled)
        for j, k in enumerate(CLUSTER_RANGE):
            agglo = AgglomerativeClustering(n_clusters=k)
            labels = agglo.fit_predict(X_reduced)
            matrix[i, j] = silhouette_score(X_reduced, labels)

    df = pd.DataFrame(matrix,
                      index=[f"{d}D" for d in DIMENSION_RANGE],
                      columns=[str(k) for k in CLUSTER_RANGE])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=False, cmap='Reds', fmt=".2f",
                cbar_kws={'label': 'Silhouette Score'})
    plt.xlabel("Number of Clusters")
    plt.ylabel("PCA Dimensions")
    plt.tight_layout()
    save_pdf("fig1F")
