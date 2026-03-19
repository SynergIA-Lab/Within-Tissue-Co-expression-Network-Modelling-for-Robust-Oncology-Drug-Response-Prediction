"""
Paso 3 v3.0: Modelado y comparación predictiva.

Feature sets:
  Baseline_RF:        Top 1000 genes (varianza) → RF
  Baseline_Resid_RF:  Top 1000 genes residualizados → RF  [NUEVO]
  Modules_v3:         510 eigengenes WGCNA v3 (kNN imputed) → RF
  EdgeDisrupt_v3:     65 PCs within-tissue disruption → RF
  Topology_v3:        72 features (14 global + 58 per-gene) → Elastic Net
  Combined_v3:        Modules_v3 + EdgeDisrupt_v3 + Topology_v3 → RF  [NUEVO]

CV: 5-fold × 3 rep, estratificado por cuartiles.
Comparación: Wilcoxon vs Baseline_RF y vs Baseline_Resid_RF, FDR BH.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import warnings, os
warnings.filterwarnings("ignore")

PROC = "/Users/mriosc/Documents/paper2/data/processed"
FEAT = "/Users/mriosc/Documents/paper2/features_v3"
MOD  = "/Users/mriosc/Documents/paper2/models_v3/cv_results"
FIG  = "/Users/mriosc/Documents/paper2/figures_v3"
LOG  = "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"
os.makedirs(MOD, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def spearman_r(y_true, y_pred):
    r, _ = stats.spearmanr(y_true, y_pred)
    return r if not np.isnan(r) else 0.0

def stratify_by_quartile(y_vec):
    return pd.qcut(y_vec, q=4, labels=False, duplicates="drop")

log("\n=== PASO 3 v3.0: Modelado ===")

# ── Load data ─────────────────────────────────────────────────────────────────
X_expr  = pd.read_csv(f"{PROC}/X_expr_matched.csv",     index_col=0)
X_resid = pd.read_csv(f"{PROC}/X_expr_residualized.csv",index_col=0)
y_all   = pd.read_csv(f"{PROC}/y_matched.csv",          index_col=0)
ma_v3   = pd.read_csv(f"{FEAT}/module_activity_v3.csv", index_col=0)
ed_v3   = pd.read_csv(f"{FEAT}/edge_disruption_within_tissue.csv", index_col=0)
topo_v3 = pd.read_csv(f"{FEAT}/topo_features_v3.csv",  index_col=0)

log(f"X_expr:  {X_expr.shape}")
log(f"X_resid: {X_resid.shape}")
log(f"Modules_v3: {ma_v3.shape}  NaN%={ma_v3.isna().mean().mean()*100:.1f}%")
log(f"EdgeDisrupt_v3: {ed_v3.shape}")
log(f"Topology_v3: {topo_v3.shape}")

# Baseline features: top 1000 genes by variance
gene_vars  = X_expr.values.var(axis=0)
top1k_idx  = np.argsort(gene_vars)[::-1][:1000]
X_baseline = X_expr.iloc[:, top1k_idx]

gene_vars_r = X_resid.values.var(axis=0)
top1k_r_idx = np.argsort(gene_vars_r)[::-1][:1000]
X_baseline_r = X_resid.iloc[:, top1k_r_idx]

# Combined: concatenate all network features
combined = pd.concat([
    ma_v3.fillna(0),
    ed_v3.fillna(0),
    topo_v3.fillna(topo_v3.median())
], axis=1)
log(f"Combined_v3: {combined.shape}")

DRUGS = {
    "Osimertinib":               "AUC",
    "Crizotinib":                "LN_IC50",
    "KRAS (G12C) Inhibitor-12":  "LN_IC50",
}

FEATURE_SETS = {
    "Baseline_RF":       (X_baseline,   "RF"),
    "Baseline_Resid_RF": (X_baseline_r, "RF"),
    "Modules_v3":        (ma_v3,        "RF"),
    "EdgeDisrupt_v3":    (ed_v3,        "RF"),
    "Topology_v3":       (topo_v3,      "EN"),
    "Combined_v3":       (combined,     "RF"),
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

# ── CV loop ───────────────────────────────────────────────────────────────────
all_scores = []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

for drug, endpoint_type in DRUGS.items():
    log(f"\n{'='*55}\n{drug}  ({endpoint_type})")
    y_drug = y_all[drug].dropna()
    common = X_expr.index.intersection(y_drug.index)
    y_vec  = y_drug.loc[common].values
    strat  = stratify_by_quartile(y_vec)
    log(f"  N={len(common)}")

    for feat_name, (X_feat, model_type) in FEATURE_SETS.items():
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
            all_scores.append({"drug": drug, "feature_set": feat_name,
                                "model": model_type, "fold": fold_i,
                                "spearman": fold_sp[-1], "r2": fold_r2[-1],
                                "rmse": fold_rmse[-1]})

        log(f"    Spearman: {np.mean(fold_sp):.3f} ± {np.std(fold_sp):.3f}")
        log(f"    R²:       {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")

# ── Save and summarize ────────────────────────────────────────────────────────
scores_df = pd.DataFrame(all_scores)
scores_df.to_csv(f"{MOD}/cv_scores_v3.csv", index=False)

summary = (scores_df.groupby(["drug","feature_set"])["spearman"]
           .agg(["mean","std"]).round(3))
log(f"\n{'='*55}")
log("TABLA RESUMEN — Spearman ρ (mean ± sd)")
log(summary.to_string())
summary.to_csv(f"{MOD}/cv_summary_v3.csv")

# ── Wilcoxon: network vs Baseline_RF and vs Baseline_Resid_RF ────────────────
log("\n--- Wilcoxon signed-rank tests ---")
network_sets = ["Modules_v3", "EdgeDisrupt_v3", "Topology_v3", "Combined_v3"]
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
# FDR correction
from scipy.stats import rankdata
pvals = wilcox_df["p_value"].values
n = len(pvals); ranks = rankdata(pvals)
fdr = np.minimum(pvals * n / ranks, 1.0)
for i in range(n-2, -1, -1): fdr[i] = min(fdr[i], fdr[i+1])
wilcox_df["p_adj_fdr"] = fdr.round(4)
wilcox_df["significant"] = wilcox_df["p_adj_fdr"] < 0.05
wilcox_df["meets_effect"] = wilcox_df["delta_spearman"] > 0.05

log(wilcox_df.to_string(index=False))
wilcox_df.to_csv(f"{MOD}/wilcoxon_v3.csv", index=False)

# ── Figures ───────────────────────────────────────────────────────────────────
feat_order = ["Baseline_RF","Baseline_Resid_RF","Modules_v3",
              "EdgeDisrupt_v3","Topology_v3","Combined_v3"]
colors = ["#4878CF","#2196F3","#6ACC65","#D65F5F","#B47CC7","#F0A500"]

# Heatmap
pivot = (scores_df.groupby(["drug","feature_set"])["spearman"]
         .mean().unstack("feature_set"))
pivot = pivot[[c for c in feat_order if c in pivot.columns]]
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=0.7)
ax.set_title("Spearman ρ — v3.0 (mean, 5-fold × 3 rep)", fontsize=12)
ax.set_xlabel("Feature set"); ax.set_ylabel("Drug")
plt.tight_layout()
plt.savefig(f"{FIG}/03_heatmap_spearman_v3.png", dpi=150, bbox_inches="tight")
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
    ax.set_xticklabels([f.replace("_v3","").replace("_RF","") for f in feat_order],
                       rotation=35, ha="right", fontsize=8)
    ax.set_title(drug.split()[0], fontsize=10)
    ax.set_ylabel("Spearman ρ")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    # Mark significant improvements
    base_mean = means.get("Baseline_RF", 0)
    for i, fs in enumerate(feat_order):
        row = wilcox_df[(wilcox_df["drug"]==drug) &
                        (wilcox_df["network_set"]==fs) &
                        (wilcox_df["baseline"]=="Baseline_RF")]
        if not row.empty and row.iloc[0]["significant"] and row.iloc[0]["meets_effect"]:
            ax.text(i, means.iloc[i] + stds.iloc[i] + 0.01, "★",
                    ha="center", fontsize=12, color="gold")

plt.suptitle("Spearman ρ v3.0 — ★ = significativamente mejor que Baseline_RF",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/03_barplot_spearman_v3.png", dpi=150, bbox_inches="tight")
plt.close()

log("\n=== PASO 3 v3.0 COMPLETO ===")
success = wilcox_df[wilcox_df["significant"] & wilcox_df["meets_effect"]]
log(f"Criterio de éxito (ΔSpearman>0.05, p_adj<0.05): {len(success)} casos")
if len(success):
    log(success[["baseline","drug","network_set","delta_spearman","p_adj_fdr"]].to_string(index=False))
else:
    log("Ningún enfoque de red supera significativamente al baseline.")
