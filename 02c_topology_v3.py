"""
Paso 2C v3.0: Métricas topológicas con features per-gene.

Cambios vs v2.0:
  - Añade degree individual de genes de interés (PTK2, YAP1, EGFR, ALK, KRAS,
    CDH1, LGALS3, EVA1A, LIF, ITM2A + top SHAP genes de v2.0)
  - Usa top 2000 genes (incluye PTK2 rank 4394 y YAP1 rank 1957)
  - Construye grafo within-tissue (referencia del propio tejido)
  - 14 métricas globales + per-gene degrees → ~80 features totales

Outputs:
  features_v3/topo_features_v3.csv  (660 × ~80)
  figures_v3/02c_topo_distributions_v3.png
  figures_v3/02c_fak_yap_degree_distribution.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
import os, warnings
warnings.filterwarnings("ignore")

PROC = "/Users/mriosc/Documents/paper2/data/processed"
NET  = "/Users/mriosc/Documents/paper2/networks_v3/within_tissue"
FEAT = "/Users/mriosc/Documents/paper2/features_v3"
FIG  = "/Users/mriosc/Documents/paper2/figures_v3"
LOG  = "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log("\n=== PASO 2C v3.0: Topología per-gene ===")

X    = pd.read_csv(f"{PROC}/X_expr_matched.csv", index_col=0)
meta = pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)
tissue_ser = meta.loc[X.index, "tissue"].fillna("Unknown")

# Top 2000 genes by variance — includes PTK2 (rank 4394 → need 5000)
# Use 5000 to ensure all genes of interest are included
gene_vars = X.values.var(axis=0)
top_idx   = np.argsort(gene_vars)[::-1][:5000]
top_genes = X.columns[top_idx].tolist()
top_symbols = [g.split(" (")[0] for g in top_genes]
X_top = X.values[:, top_idx].astype(np.float32)
n_samples, n_genes = X_top.shape
log(f"Genes for network: {n_genes} (top by variance)")

THRESHOLD = 0.7

# Genes of interest (biological + top SHAP from v2.0)
GENES_OF_INTEREST = [
    # Biological hypothesis
    "PTK2", "YAP1", "TEAD1", "TEAD4",
    # Drug targets
    "EGFR", "ALK", "KRAS", "MET", "ROS1",
    # Top SHAP genes from v2.0 baseline
    "CDH1", "RHOD", "PRSS8", "RAB25", "CST6",
    "EVA1A", "ITM2A", "SLC38A5", "PLA2G16", "NTN4",
    "LGALS3", "LIF", "DUSP23",
    # EMT markers
    "VIM", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2",
    # YAP/TAZ pathway
    "CTGF", "CYR61", "ANKRD1", "AMOTL2",
]

# Find indices of genes of interest in top_symbols
gene_idx_map = {}
for gene in GENES_OF_INTEREST:
    hits = [i for i, s in enumerate(top_symbols) if s == gene]
    if hits:
        gene_idx_map[gene] = hits[0]

log(f"Genes of interest found in top {n_genes}: {len(gene_idx_map)}/{len(GENES_OF_INTEREST)}")
for g, idx in gene_idx_map.items():
    log(f"  {g}: rank {top_idx[idx]+1}, var={gene_vars[top_idx[idx]]:.4f}")

# ── Build within-tissue reference correlations ────────────────────────────────
tissue_counts = tissue_ser.value_counts()
tissues_use   = tissue_counts[tissue_counts >= 15].index.tolist()

tissue_refs = {}  # tissue → corr_ref (n_genes × n_genes)
for tissue in tissues_use:
    t_idx = tissue_ser[tissue_ser == tissue].index
    X_t   = X_top[X.index.isin(t_idx)]
    corr_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
    for i in range(len(X_t)):
        row = X_t[i]; std = row.std()
        if std < 1e-8: continue
        z = (row - row.mean()) / std
        corr_sum += np.outer(z, z)
    tissue_refs[tissue] = (corr_sum / len(X_t)).astype(np.float32)

# Global reference (fallback)
corr_sum_global = np.zeros((n_genes, n_genes), dtype=np.float64)
for i in range(n_samples):
    row = X_top[i]; std = row.std()
    if std < 1e-8: continue
    z = (row - row.mean()) / std
    corr_sum_global += np.outer(z, z)
corr_ref_global = (corr_sum_global / n_samples).astype(np.float32)

log(f"Built {len(tissue_refs)} within-tissue reference correlations")

# ── Per-sample topological metrics ────────────────────────────────────────────
log(f"\nComputing per-sample metrics (n={n_samples}) ...")

records = []
for i in range(n_samples):
    line    = X.index[i]
    tissue  = tissue_ser.loc[line]
    corr_ref = tissue_refs.get(tissue, corr_ref_global)

    row = X_top[i]; std = row.std()
    if std < 1e-8:
        records.append({"sample": line}); continue

    z      = (row - row.mean()) / std
    corr_i = np.outer(z, z).astype(np.float32)
    np.fill_diagonal(corr_i, 0)
    adj_i  = (np.abs(corr_i) > THRESHOLD).astype(np.float32)
    deg_i  = adj_i.sum(axis=1)

    # Reference degree for this tissue
    adj_ref = (np.abs(corr_ref) > THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj_ref, 0)
    deg_ref = adj_ref.sum(axis=1)

    # Global metrics
    n_edges_i  = deg_i.sum() / 2
    density_i  = n_edges_i / (n_genes * (n_genes - 1) / 2)
    entropy_i  = scipy_entropy(deg_i + 1)
    density_ref = deg_ref.sum() / 2 / (n_genes * (n_genes - 1) / 2)
    entropy_ref = scipy_entropy(deg_ref + 1)

    # Clustering (vectorized)
    A2 = adj_i @ adj_i
    triangles = (A2 * adj_i).sum(axis=1) / 2
    denom = deg_i * (deg_i - 1) / 2
    clust_i = np.where(denom > 0, triangles / denom, 0.0)

    # Edge disruption mean (vs tissue reference)
    mask = np.triu(np.ones((n_genes, n_genes), dtype=bool), k=1)
    edge_disr = np.abs(corr_i[mask] - corr_ref[mask]).mean()

    rec = {
        "sample": line,
        # Global metrics
        "degree_mean":      deg_i.mean(),
        "degree_std":       deg_i.std(),
        "degree_max":       deg_i.max(),
        "clustering_mean":  clust_i.mean(),
        "clustering_std":   clust_i.std(),
        "density":          density_i,
        "n_edges":          n_edges_i,
        "entropy":          entropy_i,
        "edge_disruption_mean": edge_disr,
        # Delta vs tissue reference
        "delta_degree_mean":    deg_i.mean()   - deg_ref.mean(),
        "delta_clustering_mean": clust_i.mean() - 0.0,
        "delta_density":        density_i      - density_ref,
        "delta_entropy":        entropy_i      - entropy_ref,
        # Hub score (degree ratio vs reference)
        "hub_score_mean":   (deg_i / (deg_ref + 1)).mean(),
    }

    # Per-gene degree and hub score
    for gene, g_idx in gene_idx_map.items():
        deg_gene     = deg_i[g_idx]
        deg_gene_ref = deg_ref[g_idx]
        rec[f"degree_{gene}"]    = deg_gene
        rec[f"hub_score_{gene}"] = deg_gene / (deg_gene_ref + 1)

    records.append(rec)

    if (i + 1) % 100 == 0:
        log(f"  {i+1}/{n_samples} processed ...")

# ── Assemble and save ─────────────────────────────────────────────────────────
topo_df = pd.DataFrame(records).set_index("sample").reindex(X.index)
topo_df = topo_df.fillna(topo_df.median())
log(f"\nTopology feature matrix: {topo_df.shape}")

# Report per-gene features
per_gene_cols = [c for c in topo_df.columns if any(g in c for g in GENES_OF_INTEREST)]
global_cols   = [c for c in topo_df.columns if c not in per_gene_cols]
log(f"  Global metrics: {len(global_cols)}")
log(f"  Per-gene features: {len(per_gene_cols)}")

topo_df.to_csv(f"{FEAT}/topo_features_v3.csv")
log(f"Saved: features_v3/topo_features_v3.csv")

# ── PTK2/YAP1 degree: sensitive vs resistant ──────────────────────────────────
log("\n--- PTK2/YAP1 degree (within-tissue reference) ---")
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
    common = X.index.intersection(y_drug.index)
    y_vec  = y_drug.loc[common].values
    q25, q75 = np.percentile(y_vec, 25), np.percentile(y_vec, 75)
    sens_mask = y_vec <= q25
    res_mask  = y_vec >= q75

    for gene in ["PTK2", "YAP1"]:
        col = f"degree_{gene}"
        if col not in topo_df.columns:
            log(f"  {gene}: not in features"); continue
        vals = topo_df.loc[common, col].values
        deg_s = vals[sens_mask]; deg_r = vals[res_mask]
        stat, pval = mannwhitneyu(deg_s, deg_r, alternative="two-sided")
        log(f"  {drug} — {gene}: sens={deg_s.mean():.1f}±{deg_s.std():.1f}, "
            f"res={deg_r.mean():.1f}±{deg_r.std():.1f}, p={pval:.4f}")
        fak_yap_rows.append({"drug": drug, "gene": gene,
                              "deg_sens_mean": round(deg_s.mean(), 2),
                              "deg_res_mean":  round(deg_r.mean(), 2),
                              "mw_pval": round(pval, 4)})

pd.DataFrame(fak_yap_rows).to_csv(f"{FEAT}/fak_yap_degree_v3.csv", index=False)

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
plot_cols = ["degree_mean", "degree_std", "clustering_mean", "density",
             "entropy", "edge_disruption_mean", "delta_density",
             "hub_score_mean"]
for ax, col in zip(axes, plot_cols):
    if col in topo_df.columns:
        ax.hist(topo_df[col].dropna(), bins=40, color="steelblue",
                edgecolor="white", linewidth=0.3)
        ax.set_title(col, fontsize=8); ax.set_xlabel("Value"); ax.set_ylabel("N")
plt.suptitle("Topology v3.0 — feature distributions (within-tissue reference)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/02c_topo_distributions_v3.png", dpi=150, bbox_inches="tight")
plt.close()

# PTK2/YAP1 degree distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, gene in zip(axes, ["PTK2", "YAP1"]):
    col = f"degree_{gene}"
    if col in topo_df.columns:
        ax.hist(topo_df[col].dropna(), bins=40, color="#B47CC7",
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{gene} degree distribution\n(within-tissue, |r|>{THRESHOLD})", fontsize=9)
        ax.set_xlabel("Degree"); ax.set_ylabel("N cell lines")
        ax.axvline(topo_df[col].median(), color="red", linestyle="--",
                   label=f"Median={topo_df[col].median():.0f}")
        ax.legend(fontsize=8)
plt.suptitle("PTK2(FAK) and YAP1 degree — v3.0 within-tissue reference", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/02c_fak_yap_degree_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log("Figures saved.")
log("=== PASO 2C v3.0 COMPLETO ===")
