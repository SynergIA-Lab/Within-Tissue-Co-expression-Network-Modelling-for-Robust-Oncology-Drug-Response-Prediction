"""
Paso 3 z-score: Modelado con features topológicas z-scored intra-tejido.

Evalúa solo los feature sets que cambian respecto a v3.0:
  Topology_zscore:  72 features z-scored within tissue → Elastic Net
  Combined_zscore:  Modules_v3 + EdgeDisrupt_v3 + Topology_zscore → RF

Reutiliza los scores de Baseline_RF, Baseline_Resid_RF, Modules_v3,
EdgeDisrupt_v3 de v3.0 para la comparación Wilcoxon.

Outputs:
  zscore/models/cv_results/cv_scores_zscore.csv
  zscore/models/cv_results/cv_summary_zscore.csv
  zscore/models/cv_results/wilcoxon_zscore.csv
  zscore/models/cv_results/tissue_accuracy_zscore.csv
  zscore/figures/03_heatmap_spearman_zscore.png
  zscore/figures/03_barplot_spearman_zscore.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import warnings, os
warnings.filterwarnings("ignore")

PROC    = "/Users/mriosc/Documents/paper2/data/processed"
FEAT_V3 = "/Users/mriosc/Documents/paper2/features_v3"
FEAT_Z  = "/Users/mriosc/Documents/paper2/zscore/features"
MOD_V3  = "/Users/mriosc/Documents/paper2/models_v3/cv_results"
MOD     = "/Users/mriosc/Documents/paper2/zscore/models/cv_results"
FIG     = "/Users/mriosc/Documents/paper2/zscore/figures"
LOG     = "/Users/mriosc/Documents/paper2/zscore/zscore_log.txt"
os.makedirs(MOD, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def spearman_r(y_true, y_pred):
    r, _ = stats.spearmanr(y_true, y_pred)
    return r if not np.isnan(r) else 0.0

def stratify_by_quartile(y_vec):
    return pd.qcut(y_vec, q=4, labels=False, duplicates="drop")

log("\n=== PASO 3 z-score: Modelado con Topology_zscore ===")

# ── Load data ─────────────────────────────────────────────────────────────────
X_expr  = pd.read_csv(f"{PROC}/X_expr_matched.csv",     index_col=0)
X_resid = pd.read_csv(f"{PROC}/X_expr_residualized.csv",index_col=0)
y_all   = pd.read_csv(f"{PROC}/y_matched.csv",          index_col=0)
ma_v3   = pd.read_csv(f"{FEAT_V3}/module_activity_v3.csv", index_col=0)
ed_v3   = pd.read_csv(f"{FEAT_V3}/edge_disruption_within_tissue.csv", index_col=0)
topo_z  = pd.read_csv(f"{FEAT_Z}/topo_features_zscore.csv", index_col=0)
meta    = pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)

log(f"Topology_zscore: {topo_z.shape}")

# Combined_zscore: Modules + EdgeDisrupt + Topology_zscore
combined_z = pd.concat([
    ma_v3.fillna(0),
    ed_v3.fillna(0),
    topo_z.fillna(0)
], axis=1)
log(f"Combined_zscore: {combined_z.shape}")

# Baseline features (same as v3.0)
gene_vars  = X_expr.values.var(axis=0)
top1k_idx  = np.argsort(gene_vars)[::-1][:1000]
X_baseline = X_expr.iloc[:, top1k_idx]

gene_vars_r = X_resid.values.var(axis=0)
top1k_r_idx = np.argsort(gene_vars_r)[::-1][:1000]
X_baseline_r = X_resid.iloc[:, top1k_r_idx]

DRUGS = {
    "Osimertinib":               "AUC",
    "Crizotinib":                "LN_IC50",
    "KRAS (G12C) Inhibitor-12":  "LN_IC50",
}

# Only evaluate the new z-scored feature sets (baselines reused from v3.0)
FEATURE_SETS_NEW = {
    "Topology_zscore": (topo_z,      "EN"),
    "Combined_zscore": (combined_z,  "RF"),
}

def make_model(model_type):
    if model_type == "EN":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,1.0],
                                 cv=3, max_iter=5000, n_jobs=2, random_state=42))
        ])
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                       n_jobs=4, random_state=42))
    ])

# ── CV loop: only new z-scored feature sets ───────────────────────────────────
new_scores = []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

for drug, endpoint_type in DRUGS.items():
    log(f"\n{'='*55}\n{drug}  ({endpoint_type})")
    y_drug = y_all[drug].dropna()
    common = X_expr.index.intersection(y_drug.index)
    y_vec  = y_drug.loc[common].values
    strat  = stratify_by_quartile(y_vec)
    log(f"  N={len(common)}")

    for feat_name, (X_feat, model_type) in FEATURE_SETS_NEW.items():
        X_sub = X_feat.loc[common].values.astype(np.float32)
        log(f"\n  [{feat_name} / {model_type}]  features={X_sub.shape[1]}")

        fold_sp, fold_r2, fold_rmse = [], [], []
        for fold_i, (tr, te) in enumerate(cv.split(X_sub, strat)):
            model = make_model(model_type)
            model.fit(X_sub[tr], y_vec[tr])
            y_pred = model.predict(X_sub[te])
            fold_sp.append(spearman_r(y_vec[te], y_pred))
            fold_r2.append(r2_score(y_vec[te], y_pred))
            fold_rmse.append(np.sqrt(np.mean((y_vec[te] - y_pred)**2)))
            new_scores.append({"drug": drug, "feature_set": feat_name,
                                "model": model_type, "fold": fold_i,
                                "spearman": fold_sp[-1], "r2": fold_r2[-1],
                                "rmse": fold_rmse[-1]})

        log(f"    Spearman: {np.mean(fold_sp):.3f} ± {np.std(fold_sp):.3f}")
        log(f"    R²:       {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")

# Merge with v3.0 baseline scores for Wilcoxon
v3_scores = pd.read_csv(f"{MOD_V3}/cv_scores_v3.csv")
# Keep only baseline and other network sets from v3.0
keep_sets = ["Baseline_RF", "Baseline_Resid_RF", "Modules_v3", "EdgeDisrupt_v3"]
v3_scores_keep = v3_scores[v3_scores["feature_set"].isin(keep_sets)]
new_scores_df = pd.DataFrame(new_scores)
all_scores = pd.concat([v3_scores_keep, new_scores_df], ignore_index=True)
scores_df = all_scores

# ── Save and summarize ────────────────────────────────────────────────────────
scores_df = pd.DataFrame(all_scores)
scores_df.to_csv(f"{MOD}/cv_scores_zscore.csv", index=False)

summary = (scores_df.groupby(["drug","feature_set"])["spearman"]
           .agg(["mean","std"]).round(3))
log(f"\n{'='*55}")
log("TABLA RESUMEN — Spearman ρ (mean ± sd)")
log(summary.to_string())
summary.to_csv(f"{MOD}/cv_summary_zscore.csv")

# ── Tissue accuracy QC ────────────────────────────────────────────────────────
log("\n--- Tissue accuracy QC ---")
tissue_ser = meta.loc[X_expr.index, "tissue"].fillna("Unknown")
le = LabelEncoder()
tissue_enc = le.fit_transform(tissue_ser)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tissue_acc_rows = []
feat_sets_qc = {
    "Topology_zscore": topo_z,
    "Combined_zscore": combined_z,
}
# Also check baselines for reference
feat_sets_qc_all = {
    "Baseline_RF":       X_baseline,
    "Baseline_Resid_RF": X_baseline_r,
    "Modules_v3":        ma_v3,
    "EdgeDisrupt_v3":    ed_v3,
    **feat_sets_qc,
}

for feat_name, X_feat in feat_sets_qc_all.items():
    X_sub = X_feat.reindex(X_expr.index).fillna(0).values.astype(np.float32)
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   LogisticRegression(max_iter=500, C=0.1, random_state=42, n_jobs=2))
    ])
    accs = cross_val_score(clf, X_sub, tissue_enc, cv=skf, scoring="accuracy", n_jobs=2)
    acc_mean = accs.mean()
    if acc_mean < 0.15:
        status = "✓ OK"
    elif acc_mean < 0.35:
        status = "⚠ MARGINAL"
    else:
        status = "✗ CONFOUNDED"
    log(f"  {feat_name}: tissue_accuracy={acc_mean:.3f}  {status}")
    tissue_acc_rows.append({"feature_set": feat_name,
                             "tissue_accuracy": round(acc_mean, 3),
                             "status": status})

tissue_acc_df = pd.DataFrame(tissue_acc_rows)
tissue_acc_df.to_csv(f"{MOD}/tissue_accuracy_zscore.csv", index=False)
log(tissue_acc_df.to_string(index=False))

# ── Wilcoxon: zscore sets vs baselines ───────────────────────────────────────
log("\n--- Wilcoxon signed-rank tests (zscore sets vs baselines) ---")
network_sets = ["Modules_v3", "EdgeDisrupt_v3", "Topology_zscore", "Combined_zscore"]
wilcox_rows  = []

for baseline in ["Baseline_RF", "Baseline_Resid_RF"]:
    for drug in DRUGS:
        base_sp = scores_df[(scores_df["drug"]==drug) &
                            (scores_df["feature_set"]==baseline)]["spearman"].values
        for net in network_sets:
            net_sp = scores_df[(scores_df["drug"]==drug) &
                               (scores_df["feature_set"]==net)]["spearman"].values
            if len(base_sp) == len(net_sp) > 0:
                stat, pval = stats.wilcoxon(net_sp, base_sp,
                                            alternative="greater", zero_method="wilcox")
                delta = net_sp.mean() - base_sp.mean()
                wilcox_rows.append({"baseline": baseline, "drug": drug,
                                    "network_set": net,
                                    "delta_spearman": round(delta, 4),
                                    "p_value": round(pval, 4)})

wilcox_df = pd.DataFrame(wilcox_rows)
from scipy.stats import rankdata
pvals = wilcox_df["p_value"].values
n = len(pvals); ranks = rankdata(pvals)
fdr = np.minimum(pvals * n / ranks, 1.0)
for i in range(n-2, -1, -1): fdr[i] = min(fdr[i], fdr[i+1])
wilcox_df["p_adj_fdr"] = fdr.round(4)
wilcox_df["significant"] = wilcox_df["p_adj_fdr"] < 0.05
wilcox_df["meets_effect"] = wilcox_df["delta_spearman"] > 0.05

log(wilcox_df.to_string(index=False))
wilcox_df.to_csv(f"{MOD}/wilcoxon_zscore.csv", index=False)

# ── Comparison table: v3.0 vs zscore ─────────────────────────────────────────
log("\n--- Comparación v3.0 vs z-score ---")
v3_summary = pd.read_csv(f"{MOD_V3}/cv_summary_v3.csv", index_col=[0,1])
z_summary  = summary.copy()

# Align and compare Topology and Combined
compare_rows = []
for drug in DRUGS:
    for fs_v3, fs_z in [("Topology_v3", "Topology_zscore"),
                         ("Combined_v3",  "Combined_zscore")]:
        try:
            rho_v3 = v3_summary.loc[(drug, fs_v3), "mean"]
            rho_z  = z_summary.loc[(drug, fs_z), "mean"]
            delta  = rho_z - rho_v3
            compare_rows.append({"drug": drug, "feature_set_v3": fs_v3,
                                  "feature_set_z": fs_z,
                                  "rho_v3": round(rho_v3, 3),
                                  "rho_zscore": round(rho_z, 3),
                                  "delta": round(delta, 3)})
            log(f"  {drug} | {fs_v3} → {fs_z}: {rho_v3:.3f} → {rho_z:.3f} (Δ={delta:+.3f})")
        except KeyError:
            pass

compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(f"{MOD}/comparison_v3_vs_zscore.csv", index=False)

# ── Figures ───────────────────────────────────────────────────────────────────
feat_order = ["Baseline_RF", "Baseline_Resid_RF", "Modules_v3",
              "EdgeDisrupt_v3", "Topology_zscore", "Combined_zscore"]
colors = ["#4878CF","#2196F3","#6ACC65","#D65F5F","#B47CC7","#F0A500"]

# Heatmap
pivot = (scores_df.groupby(["drug","feature_set"])["spearman"]
         .mean().unstack("feature_set"))
pivot = pivot[[c for c in feat_order if c in pivot.columns]]
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=0.7)
ax.set_title("Spearman ρ — z-score (mean, 5-fold × 3 rep)", fontsize=12)
ax.set_xlabel("Feature set"); ax.set_ylabel("Drug")
plt.tight_layout()
plt.savefig(f"{FIG}/03_heatmap_spearman_zscore.png", dpi=150, bbox_inches="tight")
plt.close()

# Bar plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, drug in zip(axes, DRUGS.keys()):
    sub = scores_df[scores_df["drug"]==drug]
    means = sub.groupby("feature_set")["spearman"].mean().reindex(feat_order)
    stds  = sub.groupby("feature_set")["spearman"].std().reindex(feat_order)
    ax.bar(range(len(feat_order)), means.values, yerr=stds.values,
           color=colors, capsize=4, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels([f.replace("_zscore","(z)").replace("_v3","").replace("_RF","")
                        for f in feat_order],
                       rotation=35, ha="right", fontsize=8)
    ax.set_title(drug.split()[0], fontsize=10)
    ax.set_ylabel("Spearman ρ")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    for i, fs in enumerate(feat_order):
        row = wilcox_df[(wilcox_df["drug"]==drug) &
                        (wilcox_df["network_set"]==fs) &
                        (wilcox_df["baseline"]=="Baseline_RF")]
        if not row.empty and row.iloc[0]["significant"] and row.iloc[0]["meets_effect"]:
            ax.text(i, means.iloc[i] + stds.iloc[i] + 0.01, "★",
                    ha="center", fontsize=12, color="gold")

plt.suptitle("Spearman ρ z-score — ★ = significativamente mejor que Baseline_RF",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/03_barplot_spearman_zscore.png", dpi=150, bbox_inches="tight")
plt.close()

# Comparison figure: v3.0 vs zscore for Topology and Combined
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
drug_list = list(DRUGS.keys())
x = np.arange(len(drug_list))
width = 0.35

for ax, (fs_v3, fs_z, title) in zip(axes, [
    ("Topology_v3", "Topology_zscore", "Topology"),
    ("Combined_v3",  "Combined_zscore",  "Combined"),
]):
    rho_v3_vals = [v3_summary.loc[(d, fs_v3), "mean"] if (d, fs_v3) in v3_summary.index else np.nan
                   for d in drug_list]
    rho_z_vals  = [z_summary.loc[(d, fs_z), "mean"] if (d, fs_z) in z_summary.index else np.nan
                   for d in drug_list]
    std_v3 = [v3_summary.loc[(d, fs_v3), "std"] if (d, fs_v3) in v3_summary.index else 0
              for d in drug_list]
    std_z  = [z_summary.loc[(d, fs_z), "std"] if (d, fs_z) in z_summary.index else 0
              for d in drug_list]

    ax.bar(x - width/2, rho_v3_vals, width, yerr=std_v3, label="v3.0 (raw degree)",
           color="steelblue", capsize=4, alpha=0.85)
    ax.bar(x + width/2, rho_z_vals,  width, yerr=std_z,  label="z-score (within tissue)",
           color="#B47CC7", capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([d.split()[0] for d in drug_list], rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"{title}: v3.0 vs z-score", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

plt.suptitle("Efecto del z-score intra-tejido sobre Spearman ρ", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG}/03_v3_vs_zscore_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

log("\n=== PASO 3 z-score COMPLETO ===")
success = wilcox_df[wilcox_df["significant"] & wilcox_df["meets_effect"]]
log(f"Criterio de éxito (ΔSpearman>0.05, p_adj<0.05): {len(success)} casos")
if len(success):
    log(success[["baseline","drug","network_set","delta_spearman","p_adj_fdr"]].to_string(index=False))
else:
    log("Ningún enfoque de red supera significativamente al baseline.")
