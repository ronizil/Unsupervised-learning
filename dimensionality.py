import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import CLUSTER_RANGE, DIMENSION_RANGE


def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def cumulative_explained_variance(X, max_dims=20):
    pca = PCA(n_components=min(max_dims, X.shape[1]))
    pca.fit(X)
    return np.cumsum(pca.explained_variance_ratio_)


def get_pca_loadings(pca_model, feature_names):
    return pd.DataFrame(
        data=pca_model.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
        index=feature_names
    )


def reduce_and_score(X_scaled):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    loss_results = {}
    sil_results = {}

    for d in DIMENSION_RANGE:
        X_pca_d = PCA(n_components=d).fit_transform(X_scaled)
        losses = []
        sils = []
        for k in CLUSTER_RANGE:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(X_pca_d)
            losses.append(model.inertia_)
            sils.append(silhouette_score(X_pca_d, labels))
        loss_results[f"{d}D"] = losses
        sil_results[f"{d}D"] = sils

    return pd.DataFrame(loss_results, index=CLUSTER_RANGE).T, \
           pd.DataFrame(sil_results, index=CLUSTER_RANGE).T


