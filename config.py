from pathlib import Path
import numpy as np

# === File paths ===
DATA_PATH = r"C:\Users\RoniZil\Downloads\fetal_health.csv"
FIGURE_DIR = Path("figures_pdf")
FIGURE_DIR.mkdir(exist_ok=True)

# === Plot grid layout for merged PDFs ===
MERGED_FIGURE_PATH = FIGURE_DIR / "merged_fig1.pdf"
GRID_COLS = 2
GRID_ROWS = 3
PAGE_WIDTH = 450
PAGE_HEIGHT = 550
FONT_SIZE = 14
LABEL_OFFSET = 4
CELL_W = PAGE_WIDTH / GRID_COLS
CELL_H = PAGE_HEIGHT / GRID_ROWS

# === Cluster parameters ===
CLUSTER_RANGE = range(2, 11)
DIMENSION_RANGE = range(2, 11)
EPS_VALUES = [round(e, 2) for e in np.arange(0.05, 2.05, 0.05)]

# === Anomaly Detection Cutoffs ===
ANOMALY_STD_THRESHOLD = 3
SVM_PERCENTILE = 1
IFOREST_CONTAMINATION = 0.01
