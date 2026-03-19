"""
Paso 6 v3.0: Análisis de robustez.

6A: Tissue accuracy QC para todos los feature sets v3.
6B: Permutation test (100 permutaciones) sobre el mejor modelo.
6C: Threshold sensitivity r=0.6, 0.7, 0.8 en Topology_v3.
6D: Figura resumen comparativa v2.0 vs v3.0.

Outputs:
  models_v3/cv_results/tissue_accuracy_qc_v3.csv
  models_v3/cv_results/permutation_test_v3.csv
  models_v3/cv_results/threshold_sensitivity_v3.csv
  figures_v3/06_tissue_accuracy_qc_v3.png
  figures_v3/06_permutation_test_v3.png
  figures_v3/06_v2_vs_v3_comparison.png
"""

import numpy as np, pandas as pd, warnings, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score, cross_val_score
from sklearn.metrics import make_scorer
from scipy import stats
from scipy.stats import entropy as scipy_entropy
warnings.filterwarnings("ignore")

PROC = "/Users/mriosc/Documents/paper2/data/processed"
FEAT = "/Users/mriosc/Documents/paper2/features_v3"
MOD  = "/Users/mriosc/Documents/paper2/models_v3/cv_results"
FIG  = "/Users/mriosc/Documents/paper2/figures_v3"
LOG  = "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f: f.write(msg + "\n")

def spearman_r(y_true, y_pred):
    r, _ = stats.spearmanr(y_true, y_pred)
    return r if not np.isnan(r) else 0.0

spearman_scorer = make_scorer(spearman_r)

def stratify_q(y): return pd.qcut(y, q=4, labels=False, duplicates="drop")

log("\n=== PASO 6 v3.0: Análisis de Robustez ===")

X_expr  = pd.read_csv(f"{PROC}/X_expr_matched.csv",      index_col=0)
X_resid = pd.read_csv(f"{PROC}/X_expr_residualized.csv", index_col=0)
y_all   = pd.read_csv(f"{PROC}/y_matched.csv",           index_col=0)
meta    = pd.read_csv(f"{PROC}/cell_line_metadata.csv",  index_col=0)
ma_v3   = pd.read_csv(f"{FEAT}/module_activity_v3.csv",  index_col=0)
ed_v3   = pd.read_csv(f"{FEAT}/edge_disruption_within_tissue.csv", index_col=0)
topo_v3 = pd.read_csv(f"{FEAT}/topo_features_v3.csv",   index_col=0)

gene_vars  = X_expr.values.var(axis=0)
top1k_idx  = np.argsort(gene_vars)[::-1][:1000]
X_base     = X_expr.iloc[:, top1k_idx]
gene_vars_r = X_resid.values.var(axis=0)
top1k_r_idx = np.argsort(gene_vars_r)[::-1][:1000]
X_base_r   = X_resid.iloc[:, top1k_r_idx]
combined   = pd.concat([ma_v3.fillna(0), ed_v3.fillna(0),
                         topo_v3.fillna(topo_v3.median())], axis=1)

topo_z     = pd.read_csv("/Users/mriosc/Documents/paper2/zscore/features/topo_features_zscore.csv", index_col=0)
combined_z = pd.concat([ma_v3.fillna(0), ed_v3.fillna(0), topo_z.fillna(0)], axis=1)

tissue_ser = meta.loc[X_expr.index, "tissue"].fillna("Unknown")
le = LabelEncoder()
tissue_enc = le.fit_transform(tissue_ser.values)

DRUGS = {"Osimertinib":"AUC","Crizotinib":"LN_IC50","KRAS (G12C) Inhibitor-12":"LN_IC50"}

# ── 6A: Tissue accuracy QC ────────────────────────────────────────────────────
log("\n--- 6A: Tissue accuracy QC (all feature sets) ---")
log("Target: < 0.15 (similar to baseline)")

feat_sets = {
    "Baseline_RF":       X_base,
    "Baseline_Resid_RF": X_base_r,
    "Modules_v3":        ma_v3.fillna(0),
    "EdgeDisrupt_v3":    ed_v3,
    "Topology_v3":       topo_v3,
    "Combined_v3":       combined,
}

qc_rows = []
for name, feat_df in feat_sets.items():
    X_f = SimpleImputer(strategy="median").fit_transform(feat_df.values.astype(float))
    X_f = StandardScaler().fit_transform(X_f)
    acc = cross_val_score(RidgeClassifier(), X_f, tissue_enc,
                          cv=5, scoring="accuracy").mean()
    status = "✓ OK" if acc < 0.15 else ("⚠ MARGINAL" if acc < 0.30 else "✗ CONFOUNDED")
    log(f"  {name}: {acc:.3f}  {status}")
    qc_rows.append({"feature_set": name, "tissue_accuracy": round(acc, 3),
                    "status": status.strip()})

qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(f"{MOD}/tissue_accuracy_qc_v3.csv", index=False)

# Figure 6A
fig, ax = plt.subplots(figsize=(9, 4))
colors_qc = ["#6ACC65" if r < 0.15 else ("#F0A500" if r < 0.30 else "#D65F5F")
             for r in qc_df["tissue_accuracy"]]
bars = ax.bar(qc_df["feature_set"], qc_df["tissue_accuracy"],
              color=colors_qc, alpha=0.85)
ax.axhline(0.15, color="black", linestyle="--", lw=1.5, label="Threshold 0.15")
ax.axhline(0.562, color="gray", linestyle=":", lw=1.5, label="v2.0 EdgeDisrupt (0.562)")
ax.set_ylabel("Tissue prediction accuracy")
ax.set_title("Tissue accuracy QC — v3.0 feature sets", fontsize=11)
ax.set_xticklabels(qc_df["feature_set"], rotation=30, ha="right", fontsize=9)
for bar, val in zip(bars, qc_df["tissue_accuracy"]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
            ha="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.set_ylim(0, 0.65)
plt.tight_layout()
plt.savefig(f"{FIG}/06_tissue_accuracy_qc_v3.png", dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: 06_tissue_accuracy_qc_v3.png")

# ── 6B: Permutation test ──────────────────────────────────────────────────────
log("\n--- 6B: Permutation test (100 permutations) ---")
log("Tests best model per drug against null distribution")

perm_rows = []
cv_perm = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

# Best models from Paso 3
best_models = {
    "Osimertinib":               ("Combined_zscore", combined_z, "RF"),
    "Crizotinib":                ("Combined_zscore", combined_z, "RF"),
    "KRAS (G12C) Inhibitor-12":  ("Baseline_RF",     X_base,    "RF"),
}

for drug, (feat_name, X_feat, mtype) in best_models.items():
    y_drug = y_all[drug].dropna()
    common = X_feat.index.intersection(y_drug.index)
    X_sub  = SimpleImputer(strategy="median").fit_transform(
                X_feat.loc[common].values.astype(np.float32))
    y_vec  = y_drug.loc[common].values
    strat  = stratify_q(y_vec)

    pipe = Pipeline([("sc", StandardScaler()),
                     ("rf", RandomForestRegressor(n_estimators=100,
                      max_features="sqrt", n_jobs=4, random_state=42))])

    log(f"\n  {drug} — {feat_name} (N={len(common)}):")
    from sklearn.model_selection import KFold
    cv_perm_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    score, perm_scores, pval = permutation_test_score(
        pipe, X_sub, y_vec, scoring=spearman_scorer,
        cv=cv_perm_reg, n_permutations=100, n_jobs=2, random_state=42
    )
    log(f"    Observed Spearman: {score:.3f}")
    log(f"    Permutation mean:  {perm_scores.mean():.3f} ± {perm_scores.std():.3f}")
    log(f"    p-value: {pval:.4f}  {'✓ significant' if pval < 0.05 else '✗ not significant'}")
    perm_rows.append({"drug": drug, "feature_set": feat_name,
                      "observed_spearman": round(score, 3),
                      "perm_mean": round(perm_scores.mean(), 3),
                      "perm_std":  round(perm_scores.std(), 3),
                      "pvalue": round(pval, 4),
                      "significant": pval < 0.05})

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(f"{MOD}/permutation_test_v3.csv", index=False)
log(f"\n{perm_df.to_string(index=False)}")

# Figure 6B
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (drug, (feat_name, X_feat, _)) in zip(axes, best_models.items()):
    row = perm_df[perm_df["drug"] == drug].iloc[0]
    # Recompute perm scores for plot (use stored values)
    ax.axvline(row["observed_spearman"], color="#D65F5F", lw=2.5,
               label=f"Observed={row['observed_spearman']:.3f}")
    ax.axvline(row["perm_mean"], color="gray", lw=1.5, linestyle="--",
               label=f"Perm mean={row['perm_mean']:.3f}")
    ax.set_title(f"{drug.split()[0]}\n{feat_name}\np={row['pvalue']:.4f}", fontsize=9)
    ax.set_xlabel("Spearman ρ"); ax.set_ylabel("Count")
    ax.legend(fontsize=7)
plt.suptitle("Permutation test — observed vs null distribution", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/06_permutation_test_v3.png", dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: 06_permutation_test_v3.png")

# ── 6C: Threshold sensitivity ─────────────────────────────────────────────────
log("\n--- 6C: Threshold sensitivity r=0.6, 0.7, 0.8 (Topology_v3) ---")

X_net = X_expr.iloc[:, np.argsort(X_expr.values.var(axis=0))[::-1][:5000]].values.astype(np.float32)
meta_t = meta.loc[X_expr.index, "tissue"].fillna("Unknown")
tissue_counts = meta_t.value_counts()
tissues_use   = tissue_counts[tissue_counts >= 15].index.tolist()

thr_rows = []
cv_thr = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

for thr in [0.6, 0.7, 0.8]:
    log(f"\n  Threshold r={thr}:")
    # Build tissue refs for this threshold
    tissue_refs_thr = {}
    for tissue in tissues_use:
        t_idx = meta_t[meta_t == tissue].index
        X_t   = X_net[X_expr.index.isin(t_idx)]
        cs = np.zeros((5000, 5000), dtype=np.float64)
        for i in range(len(X_t)):
            row = X_t[i]; std = row.std()
            if std < 1e-8: continue
            z = (row - row.mean()) / std
            cs += np.outer(z, z)
        tissue_refs_thr[tissue] = (cs / len(X_t)).astype(np.float32)

    # Per-sample metrics
    records = []
    for i in range(len(X_net)):
        line   = X_expr.index[i]
        tissue = meta_t.loc[line]
        ref    = tissue_refs_thr.get(tissue, list(tissue_refs_thr.values())[0])
        row    = X_net[i]; std = row.std()
        if std < 1e-8:
            records.append({"idx": i}); continue
        z = (row - row.mean()) / std
        adj = (np.abs(np.outer(z,z)) > thr).astype(np.float32)
        np.fill_diagonal(adj, 0)
        deg = adj.sum(axis=1)
        ref_adj = (np.abs(ref) > thr).astype(np.float32)
        np.fill_diagonal(ref_adj, 0)
        ref_deg = ref_adj.sum(axis=1)
        records.append({"idx": i,
                         "degree_mean": deg.mean(), "degree_std": deg.std(),
                         "density": deg.sum()/2/(5000*4999/2),
                         "entropy": scipy_entropy(deg+1),
                         "delta_density": deg.sum()/2/(5000*4999/2) - ref_deg.sum()/2/(5000*4999/2)})

    topo_thr = pd.DataFrame(records).set_index("idx")
    topo_thr = topo_thr.drop(columns=["idx"], errors="ignore").fillna(topo_thr.median())
    topo_thr.index = X_expr.index

    for drug in DRUGS:
        y_drug = y_all[drug].dropna()
        common = X_expr.index.intersection(y_drug.index)
        X_sub  = topo_thr.loc[common].values.astype(np.float32)
        y_vec  = y_drug.loc[common].values
        strat  = stratify_q(y_vec)
        fold_sp = []
        for tr, te in cv_thr.split(X_sub, strat):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("en", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=3,
                                                  max_iter=3000, n_jobs=2, random_state=42))])
            pipe.fit(X_sub[tr], y_vec[tr])
            fold_sp.append(spearman_r(y_vec[te], pipe.predict(X_sub[te])))
        sp_m = np.mean(fold_sp); sp_s = np.std(fold_sp)
        log(f"    {drug}: ρ={sp_m:.3f}±{sp_s:.3f}")
        thr_rows.append({"threshold": thr, "drug": drug,
                          "spearman_mean": round(sp_m,3), "spearman_std": round(sp_s,3)})

thr_df = pd.DataFrame(thr_rows)
thr_df.to_csv(f"{MOD}/threshold_sensitivity_v3.csv", index=False)
log(f"\n{thr_df.pivot_table(index='drug',columns='threshold',values='spearman_mean').round(3).to_string()}")

# ── 6D: v2.0 vs v3.0 comparison figure ───────────────────────────────────────
log("\n--- 6D: v2.0 vs v3.0 comparison ---")

# v2.0 results (hardcoded from previous run)
v2_data = {
    "Osimertinib":               {"Baseline_RF":0.333,"Modules":0.338,"EdgeDisrupt":0.148,"Topology":0.046},
    "Crizotinib":                {"Baseline_RF":0.569,"Modules":0.485,"EdgeDisrupt":0.537,"Topology":0.367},
    "KRAS (G12C) Inhibitor-12":  {"Baseline_RF":0.480,"Modules":0.373,"EdgeDisrupt":0.449,"Topology":0.298},
}

scores_v3 = pd.read_csv(f"{MOD}/cv_scores_v3.csv")
v3_summary = scores_v3.groupby(["drug","feature_set"])["spearman"].mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
net_map = {"Modules":"Modules_v3","EdgeDisrupt":"EdgeDisrupt_v3","Topology":"Topology_v3"}
colors_v = {"v2.0": "#AAAAAA", "v3.0": "#4878CF"}

for ax, drug in zip(axes, DRUGS.keys()):
    x = np.arange(4)
    labels = ["Baseline_RF","Modules","EdgeDisrupt","Topology"]
    v2_vals = [v2_data[drug].get(l, 0) for l in labels]
    v3_vals = [v3_summary.get((drug, net_map.get(l, l)), 0) for l in labels]
    w = 0.35
    ax.bar(x - w/2, v2_vals, w, label="v2.0", color="#AAAAAA", alpha=0.85)
    ax.bar(x + w/2, v3_vals, w, label="v3.0", color="#4878CF", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(drug.split()[0], fontsize=10)
    ax.set_ylabel("Spearman ρ")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.5, linestyle="--")

plt.suptitle("Spearman ρ: v2.0 vs v3.0 (corrected pipeline)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG}/06_v2_vs_v3_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: 06_v2_vs_v3_comparison.png")

log("\n=== PASO 6 v3.0 COMPLETO ===")
