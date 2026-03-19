"""
Paso 2C z-score: Métricas topológicas con z-score intra-tejido.

Motivación:
  En v3.0, Topology_v3 tiene tissue_accuracy=0.889 (confundido por tejido).
  El degree de cada gen varía enormemente entre tejidos por composición tisular,
  no por biología relevante al fármaco. La solución es estandarizar (z-score)
  cada feature de degree/hub_score DENTRO de cada tejido, de modo que lo que
  queda es la desviación relativa de cada línea respecto a su propio tejido.

Estrategia:
  1. Cargar topo_features_v3.csv ya calculado (reutilizar cálculo costoso).
  2. Aplicar z-score por tejido a TODAS las columnas de degree_* y hub_score_*
     (y también a las métricas globales como degree_mean, density, etc.).
  3. Para tejidos con N<3 líneas, usar z-score global como fallback.
  4. Guardar en zscore/features/topo_features_zscore.csv

Outputs:
  zscore/features/topo_features_zscore.csv  (660 × 72, z-scored within tissue)
  zscore/figures/02c_topo_distributions_zscore.png
  zscore/figures/02c_fak_yap_degree_zscore.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

PROC = "/Users/mriosc/Documents/paper2/data/processed"
FEAT_V3 = "/Users/mriosc/Documents/paper2/features_v3"
FEAT = "/Users/mriosc/Documents/paper2/zscore/features"
FIG  = "/Users/mriosc/Documents/paper2/zscore/figures"
LOG  = "/Users/mriosc/Documents/paper2/zscore/zscore_log.txt"
os.makedirs(FEAT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log("\n=== PASO 2C z-score: Topología per-gene con z-score intra-tejido ===")

# Load pre-computed topology features from v3.0 (reuse expensive computation)
topo_raw = pd.read_csv(f"{FEAT_V3}/topo_features_v3.csv", index_col=0)
meta = pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)
tissue_ser = meta.loc[topo_raw.index, "tissue"].fillna("Unknown")

log(f"Loaded topo_features_v3.csv: {topo_raw.shape}")
log(f"Tissues: {tissue_ser.value_counts().to_dict()}")

GENES_OF_INTEREST = [
    "PTK2", "YAP1", "TEAD1", "TEAD4",
    "EGFR", "ALK", "KRAS", "MET", "ROS1",
    "CDH1", "RHOD", "PRSS8", "RAB25", "CST6",
    "EVA1A", "ITM2A", "SLC38A5", "PLA2G16", "NTN4",
    "LGALS3", "LIF", "DUSP23",
    "VIM", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2",
    "CTGF", "CYR61", "ANKRD1", "AMOTL2",
]

# ── Z-score within tissue ─────────────────────────────────────────────────────
log("\nApplying within-tissue z-score to all features ...")

# All columns to z-score (degree_*, hub_score_*, and global metrics)
cols_to_zscore = topo_raw.columns.tolist()

topo_z = topo_raw.copy()
tissue_labels = tissue_ser.reindex(topo_raw.index)

MIN_N_ZSCORE = 3  # minimum tissue size to compute z-score within tissue

zscore_stats = []
for tissue in tissue_labels.unique():
    mask = tissue_labels == tissue
    n_t  = mask.sum()
    if n_t >= MIN_N_ZSCORE:
        sub = topo_raw.loc[mask, cols_to_zscore]
        mu  = sub.mean()
        sd  = sub.std().replace(0, np.nan)  # avoid division by zero
        topo_z.loc[mask, cols_to_zscore] = (sub - mu) / sd
        zscore_stats.append({"tissue": tissue, "n": n_t, "method": "within-tissue"})
        log(f"  {tissue}: N={n_t}, z-scored within tissue")
    else:
        # Fallback: global z-score for very small tissues
        sub = topo_raw.loc[mask, cols_to_zscore]
        mu  = topo_raw[cols_to_zscore].mean()
        sd  = topo_raw[cols_to_zscore].std().replace(0, np.nan)
        topo_z.loc[mask, cols_to_zscore] = (sub - mu) / sd
        zscore_stats.append({"tissue": tissue, "n": n_t, "method": "global-fallback"})
        log(f"  {tissue}: N={n_t} < {MIN_N_ZSCORE}, using global z-score (fallback)")

# Fill any remaining NaN (from constant features within a tissue)
topo_z = topo_z.fillna(0.0)

log(f"\nZ-scored topology feature matrix: {topo_z.shape}")
log(f"  NaN after fill: {topo_z.isna().sum().sum()}")

# Report per-gene features
per_gene_cols = [c for c in topo_z.columns if any(g in c for g in GENES_OF_INTEREST)]
global_cols   = [c for c in topo_z.columns if c not in per_gene_cols]
log(f"  Global metrics: {len(global_cols)}")
log(f"  Per-gene features: {len(per_gene_cols)}")

topo_z.to_csv(f"{FEAT}/topo_features_zscore.csv")
log(f"Saved: zscore/features/topo_features_zscore.csv")

# Also save raw (pre-zscore) for reference
topo_raw.to_csv(f"{FEAT}/topo_features_raw.csv")
log(f"Saved: zscore/features/topo_features_raw.csv")

# ── PTK2/YAP1 degree (z-scored): sensitive vs resistant ──────────────────────
log("\n--- PTK2/YAP1 degree z-scored (within-tissue) ---")
y_all = pd.read_csv(f"{PROC}/y_matched.csv", index_col=0)
DRUGS = {
    "Osimertinib":               "AUC",
    "Crizotinib":                "LN_IC50",
    "KRAS (G12C) Inhibitor-12":  "LN_IC50",
}
from scipy.stats import mannwhitneyu

fak_yap_rows = []
for drug in DRUGS:
    y_drug = y_all[drug].dropna()
    common = topo_z.index.intersection(y_drug.index)
    y_vec  = y_drug.loc[common].values
    q25, q75 = np.percentile(y_vec, 25), np.percentile(y_vec, 75)
    sens_mask = y_vec <= q25
    res_mask  = y_vec >= q75

    for gene in ["PTK2", "YAP1"]:
        col = f"degree_{gene}"
        if col not in topo_z.columns:
            log(f"  {gene}: not in features"); continue
        vals = topo_z.loc[common, col].values
        deg_s = vals[sens_mask]; deg_r = vals[res_mask]
        stat, pval = mannwhitneyu(deg_s, deg_r, alternative="two-sided")
        log(f"  {drug} — {gene}: sens={deg_s.mean():.3f}±{deg_s.std():.3f}, "
            f"res={deg_r.mean():.3f}±{deg_r.std():.3f}, p={pval:.4f}")
        fak_yap_rows.append({"drug": drug, "gene": gene,
                              "deg_sens_mean": round(deg_s.mean(), 4),
                              "deg_res_mean":  round(deg_r.mean(), 4),
                              "mw_pval": round(pval, 4)})

pd.DataFrame(fak_yap_rows).to_csv(f"{FEAT}/fak_yap_degree_zscore.csv", index=False)

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
plot_cols = ["degree_mean", "degree_std", "clustering_mean", "density",
             "entropy", "edge_disruption_mean", "delta_density", "hub_score_mean"]
for ax, col in zip(axes, plot_cols):
    if col in topo_z.columns:
        ax.hist(topo_z[col].dropna(), bins=40, color="teal",
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{col}\n(z-scored within tissue)", fontsize=8)
        ax.set_xlabel("Z-score"); ax.set_ylabel("N")
plt.suptitle("Topology z-score — feature distributions (within-tissue standardized)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/02c_topo_distributions_zscore.png", dpi=150, bbox_inches="tight")
plt.close()

# PTK2/YAP1 degree z-scored distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, gene in zip(axes, ["PTK2", "YAP1"]):
    col = f"degree_{gene}"
    if col in topo_z.columns:
        ax.hist(topo_z[col].dropna(), bins=40, color="#B47CC7",
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{gene} degree (z-score within tissue)\n|r|>0.7", fontsize=9)
        ax.set_xlabel("Z-score"); ax.set_ylabel("N cell lines")
        ax.axvline(0, color="red", linestyle="--", label="Mean=0 (by construction)")
        ax.legend(fontsize=8)
plt.suptitle("PTK2(FAK) and YAP1 degree — z-scored within tissue", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/02c_fak_yap_degree_zscore.png", dpi=150, bbox_inches="tight")
plt.close()

# Comparison: raw vs z-scored distributions for degree_YAP1
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (data, label, color) in zip(axes, [
    (topo_raw["degree_YAP1"], "v3.0 raw degree_YAP1", "steelblue"),
    (topo_z["degree_YAP1"],   "z-score degree_YAP1",  "#B47CC7"),
]):
    ax.hist(data.dropna(), bins=40, color=color, edgecolor="white", linewidth=0.3)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Value"); ax.set_ylabel("N")
plt.suptitle("YAP1 degree: raw vs within-tissue z-score", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/02c_raw_vs_zscore_YAP1.png", dpi=150, bbox_inches="tight")
plt.close()

log("Figures saved.")
log("=== PASO 2C z-score COMPLETO ===")
