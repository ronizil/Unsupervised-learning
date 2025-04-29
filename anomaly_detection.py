import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from config import ANOMALY_STD_THRESHOLD, SVM_PERCENTILE, IFOREST_CONTAMINATION


def detect_anomalies(X_scaled, df):
    """
    Apply 4 anomaly detection methods:
        - KMeans Distance
        - GMM Log-Likelihood
        - One-Class SVM
        - Isolation Forest
    and return df with 5 new columns:
        'Anomaly_KMeans', 'Anomaly_GMM', 'Anomaly_SVM', 'Anomaly_IF', 'Anomaly_2of4'
    """
    # KMeans-based anomaly
    kmeans = KMeans(n_clusters=7, n_init=10, random_state=42).fit(X_scaled)
    distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    threshold_kmeans = np.mean(distances) + ANOMALY_STD_THRESHOLD * np.std(distances)
    df['Anomaly_KMeans'] = (distances > threshold_kmeans).astype(int)

    # GMM-based anomaly
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X_scaled)
    log_probs = gmm.score_samples(X_scaled)
    threshold_gmm = np.mean(log_probs) - ANOMALY_STD_THRESHOLD * np.std(log_probs)
    df['Anomaly_GMM'] = (log_probs < threshold_gmm).astype(int)

    # One-Class SVM anomaly
    svm = OneClassSVM(kernel="rbf", gamma='scale').fit(X_scaled)
    svm_scores = svm.decision_function(X_scaled)
    threshold_svm = np.percentile(svm_scores, SVM_PERCENTILE)
    df['Anomaly_SVM'] = (svm_scores < threshold_svm).astype(int)

    # Isolation Forest anomaly
    iso = IsolationForest(contamination=IFOREST_CONTAMINATION, random_state=42)
    df['Anomaly_IF'] = (iso.fit_predict(X_scaled) == -1).astype(int)

    # 2-of-4 rule
    method_cols = ['Anomaly_KMeans', 'Anomaly_GMM', 'Anomaly_SVM', 'Anomaly_IF']
    df['Anomaly_2of4'] = (df[method_cols].sum(axis=1) >= 2).astype(int)

    return df, method_cols


def get_clean_data(df, method_cols):
    """
    Filters the dataset to exclude rows marked as anomalies in 2 of 4 methods.
    Returns a clean DataFrame.
    """
    df_clean = df[df['Anomaly_2of4'] == 0].reset_index(drop=True)
    return df_clean
