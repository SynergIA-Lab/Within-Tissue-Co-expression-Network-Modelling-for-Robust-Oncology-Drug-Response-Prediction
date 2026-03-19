"""
Paso 2B v3.0: Edge Disruption within-tissue + Supervised PCA.

Corrección principal vs v2.0:
  - Red de referencia construida DENTRO de cada tejido (no pan-cancer)
  - Disruption = desviación de cada línea respecto a la media de SU tejido
  - Supervised PCA: seleccionar PCs por correlación con endpoint (no por varianza)
  - Tissue accuracy objetivo: < 0.15

Outputs:
  networks_v3/within_tissue/<tissue>_corr_ref.npy
  features_v3/edge_disruption_within_tissue.csv  (660 × n_PCs)
  figures_v3/02b_tissue_accuracy_edge.png
  figures_v3/02b_edge_variance_within_tissue.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
import os, warnings
warnings.filterwarnings("ignore")

PROC  = "/Users/mriosc/Documents/paper2/data/processed"
NET   = "/Users/mriosc/Documents/paper2/networks_v3/within_tissue"
FEAT  = "/Users/mriosc/Documents/paper2/features_v3"
FIG   = "/Users/mriosc/Documents/paper2/figures_v3"
LOG   = "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"
os.makedirs(NET, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log("\n=== PASO 2B v3.0: EdgeDisrupt within-tissue ===")

X    = pd.read_csv(f"{PROC}/X_expr_matched.csv", index_col=0)
meta = pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)
tissue_ser = meta.loc[X.index, "tissue"].fillna("Unknown")

log(f"X: {X.shape}")

THRESHOLD  = 0.7
N_GENES_LARGE = 500   # tissues N>=30
N_GENES_SMALL = 200   # tissues 15<=N<30
MIN_N_TISSUE  = 15
N_PCS_MAX     = 50    # max PCs per tissue disruption

tissue_counts = tissue_ser.value_counts()
tissues_use   = tissue_counts[tissue_counts >= MIN_N_TISSUE].index.tolist()
log(f"Tissues with N>={MIN_N_TISSUE}: {len(tissues_use)}")

# ── Build within-tissue disruption vectors ────────────────────────────────────
disruption_rows = {}   # line → disruption vector (variable length per tissue)
tissue_ref_info = {}   # tissue → (top_genes, corr_ref, mask)

for tissue in tissues_use:
    t_idx = tissue_ser[tissue_ser == tissue].index
    N_t   = len(t_idx)
    n_genes = N_GENES_LARGE if N_t >= 30 else N_GENES_SMALL
    log(f"\n  {tissue} (N={N_t}, genes={n_genes}):")

    X_t = X.loc[t_idx].values.astype(np.float32)

    # Top genes by variance WITHIN this tissue
    gene_vars_t = X_t.var(axis=0)
    top_idx_t   = np.argsort(gene_vars_t)[::-1][:n_genes]
    X_top_t     = X_t[:, top_idx_t]

    # Reference correlation for this tissue
    corr_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
    for i in range(N_t):
        row = X_top_t[i]
        std = row.std()
        if std < 1e-8: continue
        z = (row - row.mean()) / std
        corr_sum += np.outer(z, z)
    corr_ref_t = (corr_sum / N_t).astype(np.float32)

    # Save reference
    np.save(f"{NET}/{tissue.replace(' ','_')}_corr_ref.npy", corr_ref_t)

    mask_t = np.triu(np.ones((n_genes, n_genes), dtype=bool), k=1)
    corr_ref_triu = corr_ref_t[mask_t]
    n_edges = corr_ref_triu.shape[0]
    log(f"    n_edges: {n_edges:,}")

    # Per-sample disruption within tissue
    for i, line in enumerate(t_idx):
        row = X_top_t[i]
        std = row.std()
        if std < 1e-8:
            disruption_rows[line] = np.zeros(n_edges, dtype=np.float32)
            continue
        z = (row - row.mean()) / std
        disr = np.abs(np.outer(z, z)[mask_t] - corr_ref_triu)
        disruption_rows[line] = disr.astype(np.float32)

    tissue_ref_info[tissue] = (top_idx_t, corr_ref_t, mask_t)

# Lines not in any processed tissue → zero vector (will be imputed or flagged)
lines_no_tissue = [l for l in X.index if l not in disruption_rows]
log(f"\nLines without within-tissue disruption: {len(lines_no_tissue)}")

# For lines without tissue: use global reference (pan-cancer, as fallback)
if lines_no_tissue:
    # Use the largest tissue's reference as fallback
    largest_tissue = tissue_counts[tissue_counts >= MIN_N_TISSUE].idxmax()
    top_idx_fb, corr_ref_fb, mask_fb = tissue_ref_info[largest_tissue]
    n_edges_fb = corr_ref_fb[mask_fb].shape[0]
    corr_ref_fb_triu = corr_ref_fb[mask_fb]
    X_fb = X.loc[lines_no_tissue].values.astype(np.float32)[:, top_idx_fb]
    for i, line in enumerate(lines_no_tissue):
        row = X_fb[i]
        std = row.std()
        if std < 1e-8:
            disruption_rows[line] = np.zeros(n_edges_fb, dtype=np.float32)
            continue
        z = (row - row.mean()) / std
        disr = np.abs(np.outer(z, z)[mask_fb] - corr_ref_fb_triu)
        disruption_rows[line] = disr.astype(np.float32)
    log(f"  Fallback: used {largest_tissue} reference for {len(lines_no_tissue)} lines")

# ── PCA per tissue group, then concatenate ────────────────────────────────────
# Each tissue has different n_edges → can't stack directly.
# Strategy: PCA within each tissue group → take top K PCs → concatenate.
log("\n--- PCA per tissue group ---")

K_PCS_PER_TISSUE = 5   # PCs per tissue to concatenate
pca_blocks = {}

for tissue in tissues_use:
    t_idx = tissue_ser[tissue_ser == tissue].index
    mat   = np.vstack([disruption_rows[l] for l in t_idx])
    n_comp = min(K_PCS_PER_TISSUE, mat.shape[0] - 1, mat.shape[1])
    pca_t  = PCA(n_components=n_comp, random_state=42)
    coords = pca_t.fit_transform(mat)
    var_exp = pca_t.explained_variance_ratio_.sum()
    log(f"  {tissue}: {n_comp} PCs, var={var_exp*100:.1f}%")
    for i, line in enumerate(t_idx):
        if line not in pca_blocks:
            pca_blocks[line] = {}
        pca_blocks[line][tissue] = coords[i]

# Fallback lines: use zero for all tissue PCs
for line in lines_no_tissue:
    pca_blocks[line] = {t: np.zeros(K_PCS_PER_TISSUE) for t in tissues_use}

# Assemble full feature matrix: 660 × (n_tissues × K_PCS_PER_TISSUE)
col_names = [f"{t.replace(' ','_')}_PC{k+1}"
             for t in tissues_use for k in range(K_PCS_PER_TISSUE)]
feat_mat = np.zeros((len(X), len(col_names)), dtype=np.float32)

for row_i, line in enumerate(X.index):
    col_i = 0
    for tissue in tissues_use:
        vec = pca_blocks.get(line, {}).get(tissue, np.zeros(K_PCS_PER_TISSUE))
        n = min(len(vec), K_PCS_PER_TISSUE)
        feat_mat[row_i, col_i:col_i+n] = vec[:n]
        col_i += K_PCS_PER_TISSUE

ed_df = pd.DataFrame(feat_mat, index=X.index, columns=col_names)
log(f"\nEdgeDisrupt within-tissue matrix: {ed_df.shape}")

# ── Tissue accuracy check ─────────────────────────────────────────────────────
log("\n--- Tissue accuracy check ---")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

le = LabelEncoder()
tissue_enc = le.fit_transform(tissue_ser.values)

# 1. Definimos las particiones EXACTAMENTE igual que en el script zscore
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Metemos el escalado dentro del Pipeline EXACTAMENTE igual que en zscore
clf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("m",   LogisticRegression(max_iter=500, C=0.1, random_state=42))
])

# Evaluamos (usamos ed_df.values crudo, el Pipeline se encarga de escalar)
acc = cross_val_score(clf, ed_df.values, tissue_enc, cv=skf, scoring="accuracy").mean()

log(f"Tissue accuracy (within-tissue EdgeDisrupt): {acc:.3f}  (target: <0.15)")

# Save
ed_df.to_csv(f"{FEAT}/edge_disruption_within_tissue.csv")
log(f"Saved: features_v3/edge_disruption_within_tissue.csv  {ed_df.shape}")

# ── Figure: tissue accuracy comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Pan-cancer\nreference", "Within-tissue\nreference"],
       [0.562, acc], color=["#D65F5F", "#6ACC65"], alpha=0.85, width=0.4)
ax.axhline(0.15, color="black", linestyle="--", linewidth=1,
           label="Acceptance threshold (0.15)")
ax.set_ylabel("Tissue prediction accuracy")
ax.set_title("EdgeDisrupt: reduction of tissue confounding")
ax.set_ylim(0, 0.65)

# Añadir el texto con el valor exacto encima de cada barra
ax.text(0, 0.562 + 0.01, "0.562", ha='center', va='bottom', fontsize=10)
ax.text(1, acc + 0.01, f"{acc:.3f}", ha='center', va='bottom', fontsize=10)

ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/02b_tissue_accuracy_edge.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved figure: figures_v3/02b_tissue_accuracy_edge.png")