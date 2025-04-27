import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import string
from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas
from scipy.stats import shapiro
from scipy.stats import f_oneway, kruskal
from scipy.stats import pearsonr
from scipy.stats import norm
from scipy.stats import f_oneway, ttest_rel
import numpy as np
from pandas.plotting import table
import math
from sklearn.preprocessing import LabelEncoder
import itertools
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from sklearn.metrics import mutual_info_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from scipy.stats import f_oneway, kruskal
from sklearn.neighbors import LocalOutlierFactor
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from pathlib import Path
import umap
import umap.umap_ as umap

# pdf
OUTDIR = Path("figures_pdf")
OUTDIR.mkdir(exist_ok=True)
def save_pdf(name: str, dpi: int = 300, close: bool = True) -> None:
    path = OUTDIR / f"{name}.pdf"
    plt.savefig(path, format="pdf", dpi=dpi,
                bbox_inches='tight', pad_inches=0)
    print(f"✓ PDF saved to → {path}")
    if close:
        plt.close()



file_path = r"C:\Users\RoniZil\Downloads\fetal_health.csv"
df = pd.read_csv(file_path)
# ראינו כי העמודה האחרונה בדאטא היא תיוג לכן נסיר אותה ונעבוד בלדעיה כדי שהפרויקט יהיה לא מפוקח
fetal_labels = df['fetal_health'].copy()
df = df.drop(columns=[df.columns[-1]])


# נוודא שאין צורך להשתמש בone hot אז נבדוק שאין ערכים חסרים וכל הדאטא מספרי
all_numeric = df.map(lambda x: isinstance(x, (int, float))).all().all()
has_text_columns = df.select_dtypes(include=['object']).empty
has_missing_values = not df.isnull().values.any()
print(all_numeric and has_text_columns and has_missing_values)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
results = []



# נשתמש בהורדת מימדים בעזרת אלגוריתם PCA, נבדוק לכמה מימדים להוריד בעזרת בדיקת Cumulative Explained Variance ונשתמש בשיטת Elbow עם צבירת שונות של 90-95%
# Elbow
cluster_range = range(2, 11)
loss_list = []
sil_list = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    loss_list.append(kmeans.inertia_)
    sil_list.append(silhouette_score(X_pca, labels))
initial_loss = loss_list[0]
elbow_k = None
for k_idx, k in enumerate(cluster_range):
    if loss_list[k_idx] <= 0.50 * initial_loss:
        elbow_k = k
        break
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, loss_list, label='PCA dim=2')
if elbow_k:
    plt.axvline(elbow_k, color='red', linestyle='--', label=f'Elbow k={elbow_k}')
plt.xlabel("Number of Clusters")
plt.ylabel("KMeans Loss")
plt.legend()
plt.grid(True)
save_pdf("fig1B")

# Heatmap ל-loss k-means
dimensions = range(2, min(df.shape[1], 11))
loss_results = {}
silhouette_results = {}
for d in dimensions:
    X_pca_2 = PCA(n_components=d).fit_transform(X_scaled)
    losses = []
    sils = []
    for k in cluster_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_pca_2)
        losses.append(model.inertia_)
        sils.append(silhouette_score(X_pca_2, labels))
    loss_results[f"{d}D"] = losses
    silhouette_results[f"{d}D"] = sils
loss_matrix = pd.DataFrame(loss_results, index=cluster_range).T
plt.figure(figsize=(10, 6))
sns.heatmap(loss_matrix, fmt=".0f", cmap="Blues", annot=False, cbar_kws={'label': 'Loss'})
plt.xlabel("Number of Clusters")
plt.ylabel("PCA Dimensions")
save_pdf("fig1A")


# Heatmap ל-silhouette של k-means
sil_matrix = pd.DataFrame(silhouette_results, index=cluster_range).T
plt.figure(figsize=(10, 6))
sns.heatmap(sil_matrix, fmt=".2f", cmap="Blues", annot=False, cbar_kws={'label': 'Silhouette Score'})
plt.xlabel("Number of Clusters")
plt.ylabel("PCA Dimensions")
save_pdf("fig1C")

# GMM Silhouette Score Heatmap
dimensions_range = range(2, min(df.shape[1], 11))
clusters_range = range(2, 11)
score_matrix = np.zeros((len(dimensions_range), len(clusters_range)))
for i, n_dims in enumerate(dimensions_range):
    pca = PCA(n_components=n_dims)
    X_reduced = pca.fit_transform(X_scaled)
    for j, n_clusters in enumerate(clusters_range):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(X_reduced)
        try:
            score = silhouette_score(X_reduced, labels)
        except:
            score = np.nan
        score_matrix[i, j] = score
gmm_silhouette_matrix = pd.DataFrame(score_matrix,
                                     index=[f"{d}D" for d in dimensions_range],
                                     columns=[f"{k}" for k in clusters_range])
plt.figure(figsize=(10, 6))
sns.heatmap(gmm_silhouette_matrix, annot=False, cmap="Purples", fmt=".2f",
            cbar_kws={'label': 'Silhouette Score'})
plt.xlabel("Number of Clusters")
plt.ylabel("PCA Dimensions")
plt.tight_layout()
save_pdf("fig1D")


# DBSCAN Clustering
eps_values = np.arange(0.05, 2.05, 0.05)
dimensions_range = range(2, min(df.shape[1], 11))
cluster_range_dbscan = range(2, 11)
attempts = []
for n_dims in dimensions_range:
    X_reduced = PCA(n_components=n_dims).fit_transform(X_scaled)
    for eps in eps_values:
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
dbscan_score_matrix_coerced = pd.DataFrame(
    np.zeros((len(dimensions_range), len(cluster_range_dbscan))),
    index=[f"{d}D" for d in dimensions_range],
    columns=[str(k) for k in cluster_range_dbscan]
)
for n_dims in dimensions_range:
    relevant_rows = [row for row in attempts if row[0] == n_dims]
    for k in cluster_range_dbscan:
        best_diff = float('inf')
        best_score = -1.0
        found_any = False
        for (_, eps, n_clusters, score) in relevant_rows:
            if np.isnan(score):
                continue
            diff = abs(n_clusters - k)
            if diff < best_diff:
                best_diff = diff
                best_score = score
                found_any = True
            elif diff == best_diff and score > best_score:
                best_score = score
                found_any = True
        if not found_any:
            best_score = 0.0
        dbscan_score_matrix_coerced.loc[f"{n_dims}D", str(k)] = best_score
plt.figure(figsize=(10, 6))
sns.heatmap(
    dbscan_score_matrix_coerced,
    fmt=".2f",
    cmap="Greens", annot=False,
    linewidths=0,
    cbar_kws={'label': 'Silhouette Score'}
)
plt.xlabel("Number of Clusters")
plt.ylabel("PCA Dimensions")
plt.tight_layout()
save_pdf("fig1E")

# Agglomerative Clustering
agglo_scores = np.zeros((len(range(2, 11)), len(cluster_range)))
for i, n_dims in enumerate(range(2, 11)):
    X_reduced = PCA(n_components=n_dims).fit_transform(X_scaled)
    for j, k in enumerate(cluster_range):
        agglo = AgglomerativeClustering(n_clusters=k)
        labels = agglo.fit_predict(X_reduced)
        agglo_scores[i, j] = silhouette_score(X_reduced, labels)

agglo_heatmap = pd.DataFrame(agglo_scores,
                             index=[f"{d}D" for d in range(2, 11)],
                             columns=[str(k) for k in cluster_range])
plt.figure(figsize=(10, 6))
sns.heatmap(agglo_heatmap, annot=False, cmap='Reds', fmt=".2f",
            cbar_kws={'label': 'Silhouette Score'})
plt.xlabel("Number of Clusters")
plt.ylabel("PCA Dimensions")
plt.tight_layout()
save_pdf("fig1F")


# ————— Grid-merging code ————— #
PDF_FILES = [
    "figures_pdf/fig1A.pdf",
    "figures_pdf/fig1B.pdf",
    "figures_pdf/fig1C.pdf",
    "figures_pdf/fig1D.pdf",
    "figures_pdf/fig1E.pdf",
    "figures_pdf/fig1F.pdf",
]
OUTPUT_FILE = "figures_pdf/merged_fig1.pdf"
GRID_COLS, GRID_ROWS = 2, 3
PAGE_WIDTH, PAGE_HEIGHT = 450, 550
FONT_SIZE = 14
LABEL_OFFSET = 4
CELL_W = PAGE_WIDTH  / GRID_COLS
CELL_H = PAGE_HEIGHT / GRID_ROWS

def scale_to_fit(page, w, h):
    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
    s = min(w/pw, h/ph)
    page.scale_by(s)
    return page

def merge_fig1_to_grid():
    writer = PdfWriter()
    letters = string.ascii_uppercase
    panels = []
    for idx, path in enumerate(PDF_FILES):
        reader = PdfReader(path)
        pg = reader.pages[0]
        panels.append(( scale_to_fit(pg, CELL_W, CELL_H),
                        letters[idx] ))
    for i in range(0, len(panels), GRID_COLS*GRID_ROWS):
        blank = PageObject.create_blank_page(width=PAGE_WIDTH,
                                             height=PAGE_HEIGHT)
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
        c.setFont("Helvetica-Bold", FONT_SIZE)
        for j in range(GRID_COLS*GRID_ROWS):
            if i+j>=len(panels): break
            row, col = divmod(j, GRID_COLS)
            x = col*CELL_W
            y = PAGE_HEIGHT - (row+1)*CELL_H
            pg, letter = panels[i+j]
            blank.merge_translated_page(pg, x, y)
            c.drawString(x+LABEL_OFFSET,
                         y+CELL_H-FONT_SIZE-LABEL_OFFSET,
                         letter)
        c.save(); buf.seek(0)
        lbl = PdfReader(buf).pages[0]
        blank.merge_page(lbl)
        writer.add_page(blank)
    with open(OUTPUT_FILE,'wb') as f:
        writer.write(f)
    print("✅ merged grid saved to", OUTPUT_FILE)
merge_fig1_to_grid()




# ===  ללא הפחתת מימדים ===
kmeans_scores_raw = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    kmeans_scores_raw.append(score)
kmeans_df_raw = pd.DataFrame([kmeans_scores_raw], index=["No Reduction"], columns=[str(k) for k in cluster_range])
gmm_scores_raw = []
for k in cluster_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    gmm_scores_raw.append(score)
gmm_df_raw = pd.DataFrame([gmm_scores_raw], index=["No Reduction"], columns=[str(k) for k in cluster_range])
agglo_scores_raw = []
for k in cluster_range:
    agglo = AgglomerativeClustering(n_clusters=k)
    labels = agglo.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    agglo_scores_raw.append(score)
agglo_df_raw = pd.DataFrame([agglo_scores_raw], index=["No Reduction"], columns=[str(k) for k in cluster_range])
eps_values = np.arange(0.05, 2.05, 0.05)
dbscan_score_dict = {str(k): [] for k in cluster_range}
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if 2 <= n_clusters <= 10:
        try:
            score = silhouette_score(X_scaled, labels)
            dbscan_score_dict[str(n_clusters)].append(score)
        except:
            continue
dbscan_avg_scores = []
for k in cluster_range:
    scores = dbscan_score_dict[str(k)]
    dbscan_avg_scores.append(np.mean(scores) if scores else 0.0)
dbscan_df_raw = pd.DataFrame([dbscan_avg_scores], index=["No Reduction"], columns=[str(k) for k in cluster_range])
kmeans_max_with_pca = 0.41
gmm_max_with_pca = 0.41
dbscan_max_with_pca = 0.34
agglo_max_with_pca = 0.38
algorithms = ["KMeans", "Agglomerative", "GMM", "DBSCAN"]
with_pca = [kmeans_max_with_pca, agglo_max_with_pca, gmm_max_with_pca, dbscan_max_with_pca]
without_pca = [
    max(kmeans_scores_raw),
    max(agglo_scores_raw),
    max(gmm_scores_raw),
    max(dbscan_avg_scores)]
print("\nהשוואת ביצועים: האם הפחתת מימדים שיפרה את הביצועים?\n")
for algo, pca_val, raw_val in zip(algorithms, with_pca, without_pca):
    better = "Yes" if pca_val > raw_val else "No"
    print(f"{algo}: With PCA = {pca_val:.3f}, Without PCA = {raw_val:.3f} → PCA Better with algorithm {algo}? {better}")

# אנומליות
# --- KMeans ---
kmeans = KMeans(n_clusters=7, n_init=10, random_state=42).fit(X_scaled)
distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
threshold_kmeans = np.mean(distances) + 3 * np.std(distances)
df['Anomaly_KMeans'] = (distances > threshold_kmeans).astype(int)
# --- GMM ---
gmm = GaussianMixture(n_components=2, random_state=42).fit(X_scaled)
log_probs = gmm.score_samples(X_scaled)
threshold_gmm = np.mean(log_probs) - 3 * np.std(log_probs)
df['Anomaly_GMM'] = (log_probs < threshold_gmm).astype(int)
# --- One-Class SVM ---
svm = OneClassSVM(kernel="rbf", gamma='scale').fit(X_scaled)
svm_scores = svm.decision_function(X_scaled)
threshold_svm = np.percentile(svm_scores, 1)
df['Anomaly_SVM'] = (svm_scores < threshold_svm).astype(int)
# --- Isolation Forest ---
iso = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly_IF'] = (iso.fit_predict(X_scaled) == -1).astype(int)
# --- 2 of 4 cutoff ---
method_cols = ['Anomaly_KMeans', 'Anomaly_GMM', 'Anomaly_SVM', 'Anomaly_IF']
df['Anomaly_2of4'] = (df[method_cols].sum(axis=1) >= 2).astype(int)
percent_anomaly_2of4 = 100 * df['Anomaly_2of4'].sum() / len(df)
print(f"Anomaly_2of4 מופיע ב-{percent_anomaly_2of4:.2f}% מהדאטה")
df_clean = df[df['Anomaly_2of4'] == 0].reset_index(drop=True)
warnings.filterwarnings("ignore")
features_for_pca = df_clean.drop(columns=method_cols + ['fetal_health', 'Anomaly_2of4'], errors='ignore').columns
X_scaled_clean = scaler.fit_transform(df_clean[features_for_pca])
pca_clean_model = PCA(n_components=2, random_state=42)
X_pca_clean = pca_clean_model.fit_transform(X_scaled_clean)

# 2dim |loadings|
loadings_df_clean = pd.DataFrame(
    data=pca_clean_model.components_.T,
    columns=['PC1', 'PC2'],
    index=features_for_pca
)
loading_magnitudes_clean = loadings_df_clean.abs().max(axis=1).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=loading_magnitudes_clean.values,
    y=loading_magnitudes_clean.index,
    hue=loading_magnitudes_clean.index,
    dodge=False,
    palette="viridis",
    legend=False
)
plt.xlabel("|loading| both PCs")
plt.tight_layout()
save_pdf("fig2A")


# גרפים נפרדים לפי רכיב PCA
for i, pc in enumerate(['PC1', 'PC2']):
    plt.figure(figsize=(10, 6))
    sorted_vals = loadings_df_clean[pc].sort_values(ascending=True)
    sns.barplot(
        x=sorted_vals.values,
        y=sorted_vals.index,
        palette="coolwarm"
    )
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel(f"Loading Value ({pc})")
    plt.ylabel("Feature")
    plt.tight_layout()
    fig_name = f"fig2{chr(66 + i)}"
    save_pdf(fig_name)

# סילואט לפי קלאסטרים של KMEANS בעל הציון הממוצע המקסימלי
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(X_pca)
silhouette_vals = silhouette_samples(X_pca, labels)
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10
set2_colors = sns.color_palette("Set2", 7)
for i in range(7):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster
    color = set2_colors[i]
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color)
    ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
    y_lower = y_upper + 10
ax.set_xlabel("Silhouette score")
ax.set_ylabel("Cluster index")
ax.set_xlim([-0.2, 1])
ax.set_yticks([])
plt.tight_layout()
save_pdf("fig3A")
plt.close()




# בחירת אלגוריתם אשכול אידיאלי
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
save_pdf("Table_2")





# קרושקל ואנובה
y_true = fetal_labels[df['Anomaly_2of4'] == 0].reset_index(drop=True)
X_scaled_clean = StandardScaler().fit_transform(
    df_clean.drop(columns=['Anomaly_2of4'] + method_cols, errors='ignore')
)
X_pca_clean = PCA(n_components=2, random_state=42).fit_transform(X_scaled_clean)
labels_dict = {
    "KMeans": KMeans(n_clusters=7, random_state=42).fit_predict(X_pca_clean),
    "GMM": GaussianMixture(n_components=2, random_state=42).fit(X_pca_clean).predict(X_pca_clean),
    "Hierarchical": AgglomerativeClustering(n_clusters=2).fit_predict(X_pca_clean),
}
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
rows = []
for name, lbl in labels_dict.items():
    pa, pk = summary_tests(lbl, y_true.values)
    rows.append({
        "Algorithm": name,
        "ANOVA p-value": pa,
        "Kruskal p-value": pk
    })
df_stats = pd.DataFrame(rows).set_index("Algorithm")
formatted_df_stats = df_stats.applymap(lambda x: f"{x:.2e}")
fig, ax = plt.subplots(figsize=(9, 1.2))
ax.axis('off')
tbl = ax.table(
    cellText=formatted_df_stats.values,
    rowLabels=formatted_df_stats.index,
    colLabels=formatted_df_stats.columns,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.0, 1.3)
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
save_pdf("Table_1")




# PCA 2D 7 clusters KMEANS
df = pd.read_csv(file_path)
fetal_labels = df['fetal_health'].copy()
X = df.drop(columns=['fetal_health']).iloc[:, 0:21]
feature_names = X.columns.tolist()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca_2d = pca.fit_transform(X_scaled)
km = KMeans(n_clusters=7, random_state=42)
cluster_labels = km.fit_predict(X_pca_2d)
plt.figure(figsize=(8,6))
palette = sns.color_palette("tab10", 7)
sns.scatterplot(x=X_pca_2d[:,0], y=X_pca_2d[:,1],
                hue=cluster_labels, palette=palette, s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.tight_layout()
save_pdf("fig3B")

# signed loadings
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


# Absolut loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=['PC1','PC2']
)

for i, pc in enumerate(['PC1', 'PC2']):
    plt.figure(figsize=(10,6))
    vals = loadings[pc].abs().sort_values(ascending=True)
    sns.barplot(x=vals.values, y=vals.index, palette="viridis")
    plt.xlabel('Absolute loading')
    plt.ylabel('Feature')
    plt.tight_layout()
    save_pdf(f"fig3{chr(69+i)}")


#PCA top 4 influential features
pca_full = PCA(n_components=2, random_state=42)
pca_full.fit(X_scaled_clean)  # this is the model
loadings = pd.DataFrame(
    pca_full.components_.T,
    index=features_for_pca,
    columns=['PC1', 'PC2']
)
top_4_features = loadings.abs().max(axis=1).sort_values(ascending=False).head(6).index.tolist()
X_top4 = df[top_4_features]
X_top4_scaled = scaler.fit_transform(X_top4)
pca = PCA(n_components=2, random_state=42)
X_pca_n = pca.fit_transform(X_top4_scaled)
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(X_pca_n)
plt.figure(figsize=(8, 6))
palette = sns.color_palette("tab10", 7)
sns.scatterplot(x=X_pca_n[:, 0], y=X_pca_n[:, 1], hue=labels, palette=palette, s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.tight_layout()
save_pdf("fig4A")

df_clusters = pd.DataFrame({
    "cluster": labels,
    "fetal_health": fetal_labels
})
label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
df_clusters["health_label"] = df_clusters["fetal_health"].map(label_map)
dist_all = (
    df_clusters
    .groupby("cluster")["health_label"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    * 100
).round(2).applymap(lambda x: f"{x:.2f}%")
print("\n=== KMeans (7 clusters) on Top 4 Features — All Points ===")
print(dist_all)
centroids = kmeans.cluster_centers_
distances = np.linalg.norm(X_pca_n - centroids[labels], axis=1)
df_clusters["distance_to_center"] = distances




# UMAP 2D
X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
kmeans = KMeans(n_clusters=7, random_state=42).fit(X_umap)
labels = kmeans.labels_
sil = silhouette_score(X_umap, labels)

plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=labels, palette='tab10',
    s=20, alpha=0.7, legend=False
)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(False)
plt.tight_layout()
save_pdf("fig4B")



labels_kmeans = KMeans(n_clusters=7, random_state=42).fit_predict(X_pca)
if 'histogram_tendency' not in df.columns:
    raise ValueError("")
df['Cluster'] = labels_kmeans
tendency_per_cluster = (
    df.groupby('Cluster')['histogram_tendency']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .round(3) * 100
)
print("histogram_tendency percentage")
print(tendency_per_cluster)