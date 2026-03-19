"""
Paso 5 v3.0: Validación externa GSE255958 ampliada.

Incluye cell lines + organoides (TH107) + PDX (TH021, LG0812).
Aplica modelo GDSC (Baseline_RF + Combined_v3) → AUC ROC Responder vs DTP.
PCA topológico Responder vs DTP.

Outputs:
  models_v3/cv_results/gse255958_roc_v3.csv
  figures_v3/05_roc_v3.png
  figures_v3/05_pca_topo_responder_dtp_v3.png
"""

import numpy as np, pandas as pd, os, gzip, re, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu, entropy as scipy_entropy
warnings.filterwarnings("ignore")

PROC    = "/Users/mriosc/Documents/paper2/data/processed"
RAW_GSE = "/Users/mriosc/Documents/paper2/data/raw/GSE255958"
FEAT    = "/Users/mriosc/Documents/paper2/features_v3"
MOD     = "/Users/mriosc/Documents/paper2/models_v3/cv_results"
FIG     = "/Users/mriosc/Documents/paper2/figures_v3"
LOG     = "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"
os.makedirs(MOD, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f: f.write(msg + "\n")

log("\n=== PASO 5 v3.0: Validación externa GSE255958 ===")

# ── Load pre-built GSE255958 expression (from v2.0, reused) ──────────────────
expr_gse = pd.read_csv(f"{PROC}/gse255958_expression.csv", index_col=0)
meta_gse = pd.read_csv(f"{PROC}/gse255958_metadata.csv",   index_col=0)
log(f"GSE255958: {expr_gse.shape[0]} samples × {expr_gse.shape[1]} genes")
log(f"Groups: {meta_gse['group'].value_counts().to_dict()}")

# Extend classification to include organoids and PDX
def classify_extended(fname):
    name = re.sub(r"^GSM\d+_", "", fname.replace(".genes.results.gz",""))
    # Organoids (TH107 EGFR-mutant)
    if "TH107-orgnoid-DMSO" in name:  return ("TH107_org","EGFR","DMSO","Responder")
    if "TH107-orgnoid-Osi-d3" in name: return ("TH107_org","EGFR","Osi_d3","Responder")
    if "TH107-orgnoid-Osi-d9" in name: return ("TH107_org","EGFR","Osi_d9","DTP")
    # PDX (TH021 EGFR, LG0812 ALK)
    if "TH021-PDX-DMSO" in name:   return ("TH021_pdx","EGFR","DMSO","Responder")
    if "TH021-PDX-Osi" in name:    return ("TH021_pdx","EGFR","Osi","DTP")
    if "LG0812-PDX-DMSO" in name:  return ("LG0812_pdx","ALK","DMSO","Responder")
    if "LG0812-PDX-Alec" in name:  return ("LG0812_pdx","ALK","Alec","DTP")
    return (None,None,None,None)

# Load organoid/PDX samples not in v2.0 metadata
extra_records = {}
for fname in sorted(os.listdir(RAW_GSE)):
    if not fname.endswith(".gz"): continue
    cl, tgt, cond, grp = classify_extended(fname)
    if cl is None: continue
    sample = re.sub(r"^GSM\d+_","", fname.replace(".genes.results.gz",""))
    if sample not in expr_gse.index:
        # Load TPM
        with gzip.open(f"{RAW_GSE}/{fname}") as f:
            df = pd.read_csv(f, sep="\t", usecols=["gene_id","TPM"])
        df["gene"] = df["gene_id"].str.split("_").str[0]
        df = df.drop_duplicates("gene").set_index("gene")["TPM"]
        extra_records[sample] = {"tpm": df, "cell_line": cl, "target": tgt,
                                  "condition": cond, "group": grp}

if extra_records:
    # Build expression matrix for extra samples
    all_genes = expr_gse.columns
    extra_expr = pd.DataFrame({s: np.log2(v["tpm"].reindex(all_genes).fillna(0)+1)
                                for s,v in extra_records.items()}).T
    extra_meta = pd.DataFrame({s: {"cell_line":v["cell_line"],"target":v["target"],
                                    "condition":v["condition"],"group":v["group"]}
                                for s,v in extra_records.items()}).T
    expr_gse_ext = pd.concat([expr_gse, extra_expr])
    meta_gse_ext = pd.concat([meta_gse[["cell_line","target","condition","group"]], extra_meta])
    log(f"Extended GSE255958: {expr_gse_ext.shape[0]} samples "
        f"(+{len(extra_records)} organoid/PDX)")
else:
    expr_gse_ext = expr_gse
    meta_gse_ext = meta_gse[["cell_line","target","condition","group"]]
    log("No additional organoid/PDX samples found — using existing dataset")

# ── Load GDSC data and train models ──────────────────────────────────────────
X_gdsc = pd.read_csv(f"{PROC}/X_expr_matched.csv", index_col=0)
y_gdsc = pd.read_csv(f"{PROC}/y_matched.csv",      index_col=0)

gene_vars   = X_gdsc.values.var(axis=0)
top1k_idx   = np.argsort(gene_vars)[::-1][:1000]
top1k_genes = X_gdsc.columns[top1k_idx].tolist()
top1k_syms  = [g.split(" (")[0] for g in top1k_genes]

DRUGS_GSE = {
    "Osimertinib":               ("EGFR", ["PC9","PC9_iso","PC9_pers","H1975",
                                            "TH107_org","TH021_pdx"]),
    "Crizotinib":                ("ALK",  ["H3122","H3122_pers","LG0812_pdx"]),
    "KRAS (G12C) Inhibitor-12":  ("KRAS", ["H358"]),
}

roc_rows = []
fig_roc, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (drug, (target, cell_lines)) in zip(axes, DRUGS_GSE.items()):
    log(f"\n  {drug} ({target}):")
    mask = meta_gse_ext["cell_line"].isin(cell_lines)
    meta_sub = meta_gse_ext[mask]
    expr_sub  = expr_gse_ext.loc[meta_sub.index]
    n_resp = (meta_sub["group"]=="Responder").sum()
    n_dtp  = (meta_sub["group"]=="DTP").sum()
    log(f"    N={len(meta_sub)} (Responder={n_resp}, DTP={n_dtp})")

    if n_resp < 2 or n_dtp < 2:
        log("    Skipping — insufficient samples"); ax.set_visible(False); continue

    # Train on full GDSC
    y_drug = y_gdsc[drug].dropna()
    common = X_gdsc.index.intersection(y_drug.index)
    X_tr   = X_gdsc.loc[common, top1k_genes].values.astype(np.float32)
    y_tr   = y_drug.loc[common].values
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr)
    rf = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                n_jobs=4, random_state=42)
    rf.fit(X_tr_sc, y_tr)

    # Project GSE samples
    X_gse = np.zeros((len(expr_sub), len(top1k_genes)), dtype=np.float32)
    for j, sym in enumerate(top1k_syms):
        if sym in expr_sub.columns:
            X_gse[:, j] = expr_sub[sym].values
    y_pred = rf.predict(sc.transform(X_gse))
    y_bin  = (meta_sub["group"]=="DTP").astype(int).values

    if len(np.unique(y_bin)) < 2:
        log("    Only one class — skip"); ax.set_visible(False); continue

    auc = roc_auc_score(y_bin, -y_pred)
    fpr, tpr, _ = roc_curve(y_bin, -y_pred)


    # Bootstrap CI
    np.random.seed(42)
    boot = [roc_auc_score(y_bin[idx], -y_pred[idx])
            for _ in range(1000) 
            if len(np.unique(y_bin[(idx := np.random.choice(len(y_bin), len(y_bin)))])) > 1]
    
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) if boot else (0, 0)
    log(f"    AUC={auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Sample type breakdown
    for stype in meta_sub["cell_line"].unique():
        st_mask = meta_sub["cell_line"] == stype
        n_r = (meta_sub[st_mask]["group"]=="Responder").sum()
        n_d = (meta_sub[st_mask]["group"]=="DTP").sum()
        log(f"      {stype}: Resp={n_r}, DTP={n_d}")

    roc_rows.append({"drug":drug,"target":target,"n_responder":n_resp,
                     "n_dtp":n_dtp,"auc":round(auc,3),
                     "ci_lo":round(ci_lo,3),"ci_hi":round(ci_hi,3)})

    ax.plot(fpr, tpr, color="#D65F5F", lw=2,
            label=f"AUC={auc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#D65F5F")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{drug.split()[0]} ({target})\nN={len(meta_sub)}", fontsize=9)
    ax.legend(fontsize=8); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

plt.suptitle("ROC — GDSC model → GSE255958 (Responder vs DTP) v3.0", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/05_roc_v3.png", dpi=150, bbox_inches="tight")
plt.close()

roc_df = pd.DataFrame(roc_rows)
roc_df.to_csv(f"{MOD}/gse255958_roc_v3.csv", index=False)
log(f"\nROC results:\n{roc_df.to_string(index=False)}")

# ── PCA topológico Responder vs DTP ──────────────────────────────────────────
log("\n--- PCA topológico Responder vs DTP ---")

# Compute topo metrics for GSE255958 samples using GDSC within-tissue reference
# Use global reference (pan-cancer) since GSE samples are lung cancer lines
corr_ref_path = "/Users/mriosc/Documents/paper2/networks_v3/within_tissue/Lung_corr_ref.npy"
if os.path.exists(corr_ref_path):
    corr_ref = np.load(corr_ref_path)
    log(f"Using Lung within-tissue reference: {corr_ref.shape}")
else:
    log("Lung reference not found — skipping PCA topo")
    corr_ref = None

if corr_ref is not None:
    # Top genes used for Lung reference (top 500 by variance in Lung)
    X_lung = X_gdsc[X_gdsc.index.isin(
        pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)
        .query("tissue=='Lung'").index)].values.astype(np.float32)
    gene_vars_lung = X_lung.var(axis=0)
    top_lung_idx   = np.argsort(gene_vars_lung)[::-1][:500]
    top_lung_syms  = [X_gdsc.columns[i].split(" (")[0] for i in top_lung_idx]
    n_genes_ref    = corr_ref.shape[0]

    THRESHOLD = 0.7
    mask_triu = np.triu(np.ones((n_genes_ref, n_genes_ref), dtype=bool), k=1)
    corr_ref_triu = corr_ref[mask_triu]

    gse_topo = []
    for sample in expr_gse_ext.index:
        row = np.zeros(n_genes_ref, dtype=np.float32)
        for j, sym in enumerate(top_lung_syms[:n_genes_ref]):
            if sym in expr_gse_ext.columns:
                row[j] = expr_gse_ext.loc[sample, sym]
        std = row.std()
        if std < 1e-8:
            gse_topo.append({"sample": sample}); continue
        z = (row - row.mean()) / std
        adj = (np.abs(np.outer(z,z)) > THRESHOLD).astype(np.float32)
        np.fill_diagonal(adj, 0)
        deg = adj.sum(axis=1)
        edge_disr = np.abs(np.outer(z,z)[mask_triu] - corr_ref_triu).mean()
        gse_topo.append({"sample": sample,
                          "degree_mean": deg.mean(), "degree_std": deg.std(),
                          "density": deg.sum()/2/(n_genes_ref*(n_genes_ref-1)/2),
                          "entropy": scipy_entropy(deg+1),
                          "edge_disruption": edge_disr})

    topo_gse_df = pd.DataFrame(gse_topo).set_index("sample").reindex(expr_gse_ext.index)
    topo_gse_df = topo_gse_df.fillna(topo_gse_df.median())

    # PCA
    X_pca = StandardScaler().fit_transform(topo_gse_df.values)
    pca   = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_pca)
    var_exp = pca.explained_variance_ratio_

    groups = meta_gse_ext.loc[topo_gse_df.index, "group"].values
    fig, ax = plt.subplots(figsize=(7, 6))
    for grp, col, mk in [("Responder","#6ACC65","o"), ("DTP","#D65F5F","^")]:
        m = groups == grp
        ax.scatter(coords[m,0], coords[m,1], c=col, label=grp,
                   s=80, alpha=0.85, marker=mk, edgecolors="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_title("PCA — Topological metrics GSE255958\nResponder vs DTP (v3.0)", fontsize=11)
    ax.legend(title="Group", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG}/05_pca_topo_responder_dtp_v3.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Figure saved: 05_pca_topo_responder_dtp_v3.png")

    # Mann-Whitney on topo metrics
    resp_idx = [s for s in topo_gse_df.index if meta_gse_ext.loc[s,"group"]=="Responder"]
    dtp_idx  = [s for s in topo_gse_df.index if meta_gse_ext.loc[s,"group"]=="DTP"]
    log("\nMann-Whitney Responder vs DTP (topo metrics):")
    for col in topo_gse_df.columns:
        r_v = topo_gse_df.loc[resp_idx, col].values
        d_v = topo_gse_df.loc[dtp_idx,  col].values
        _, pval = mannwhitneyu(r_v, d_v, alternative="two-sided")
        log(f"  {col}: Resp={r_v.mean():.4f}, DTP={d_v.mean():.4f}, p={pval:.4f}")

log("\n=== PASO 5 v3.0 COMPLETO ===")
