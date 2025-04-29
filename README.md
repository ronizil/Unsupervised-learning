# Unsupervised Learning on Fetal Health Data

This project applies unsupervised machine learning techniques to analyze fetal health data. We use clustering and anomaly detection methods to explore the structure of the dataset and identify potentially pathological cases without supervision.

---

## 📂 Project Structure

```
fetal_health_project/
├── anomaly_detection.py
├── clustering.py
├── config.py
├── dimensionality.py
├── evaluation.py
├── load_data.py
├── main.py
├── merge_pdfs.py
├── statistics_tests.py
├── visualizations.py
├── figures_pdf/                
│   ├── fig1A.pdf to fig1F.pdf
│   ├── merged_fig1.pdf
│   ├── Table_1.pdf
│   └── Table_2.pdf
└── README.md
```

---

## 📊 Methods Used

### Dimensionality Reduction
- PCA (Principal Component Analysis)

### Clustering Algorithms
- KMeans
- Gaussian Mixture Model (GMM)
- DBSCAN
- Agglomerative (Hierarchical) Clustering

### Anomaly Detection Methods
- KMeans Distance
- GMM Likelihood
- One-Class SVM
- Isolation Forest

### Statistical Analysis
- One-Way ANOVA
- Kruskal-Wallis H-Test

---

## 🛠️ Setup

### 1. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 2. Install required packages:
```
pip install -r requirements.txt
```
If no `requirements.txt`, install manually:
```
pip install numpy pandas matplotlib seaborn scikit-learn pypdf reportlab
```

### 3. Place your dataset
Ensure the dataset `fetal_health.csv` is located at:
```
C:/Users/RoniZil/Downloads/fetal_health.csv
```
Or change `DATA_PATH` in `config.py` accordingly.

---

## ▶️ How to Run

Execute the full analysis pipeline:
```
python main.py
```
This will:
- Run PCA and all clustering algorithms
- Detect anomalies
- Save all clustering and evaluation figures
- Merge figures into a single PDF grid
- Save statistical test results as PDF tables

---

## 📁 Outputs

All figures and tables are saved under the `figures_pdf/` folder, including:
- `fig1A–F.pdf`: Clustering evaluation heatmaps
- `merged_fig1.pdf`: Merged visual overview
- `Table_1.pdf`: ANOVA + Kruskal results
- `Table_2.pdf`: Composite clustering scores

---

## 👩‍🔬 Author Notes

This project was developed for academic purposes as part of an unsupervised learning course. It demonstrates clustering evaluation, statistical validation, and anomaly interpretation in real-world biomedical data.

---

## 📬 Contact
For questions or feedback, please contact: [ronishzil@gmail.com]
