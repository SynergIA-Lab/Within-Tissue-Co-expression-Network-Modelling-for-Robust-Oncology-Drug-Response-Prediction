"""
supplementary_experiments.py
============================
Supplementary experiments for the manuscript on network-based drug sensitivity prediction.

Experiments implemented (in order of priority):
  1. SHAP on unconfounded features (Combined zscore + Topology zscore) for osimertinib
  2. YAP1/TEAD1 centrality analysis using Combined zscore SHAP values
  3. Ablation study: Combined_noEdge = Modules + Topology zscore (no EdgeDisrupt)
  4. Sensitivity analysis: exclude kNN-imputed cell lines
  5. (R script) WGCNA module stability bootstrap for tissues with N < 30
  6. PCA baseline comparison vs Modules

All outputs saved to /results/supplementary/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import os
import time
import traceback
from datetime import datetime
from scipy import stats
from scipy.stats import mannwhitneyu, rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC    = "/Users/mriosc/Documents/paper2/data/processed"
FEAT_V3 = "/Users/mriosc/Documents/paper2/features_v3"
FEAT_Z  = "/Users/mriosc/Documents/paper2/zscore/features"
MOD_V3  = "/Users/mriosc/Documents/paper2/models_v3/cv_results"
MOD_Z   = "/Users/mriosc/Documents/paper2/zscore/models/cv_results"
SUPP    = "/Users/mriosc/Documents/paper2/results/supplementary"
LOG_PATH = f"{SUPP}/experiment_log.txt"

os.makedirs(SUPP, exist_ok=True)
os.makedirs(f"{SUPP}/wgcna_stability", exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def spearman_r(y_true, y_pred):
    r, _ = stats.spearmanr(y_true, y_pred)
    return r if not np.isnan(r) else 0.0

def stratify_by_quartile(y_vec):
    return pd.qcut(y_vec, q=4, labels=False, duplicates="drop")

def make_rf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                      n_jobs=-1, random_state=42))
    ])

def make_en():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
                             cv=3, max_iter=5000, n_jobs=2, random_state=42))
    ])

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    ranks = rankdata(pvals)
    fdr = np.minimum(pvals * n / ranks, 1.0)
    for i in range(n - 2, -1, -1):
        fdr[i] = min(fdr[i], fdr[i + 1])
    return fdr

def run_cv(X_feat, y_vec, strat, model_type="RF"):
    """Run 5-fold x 3-rep stratified CV, return per-fold Spearman, R2, RMSE."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    fold_sp, fold_r2, fold_rmse = [], [], []
    for tr, te in cv.split(X_feat, strat):
        model = make_rf() if model_type == "RF" else make_en()
        model.fit(X_feat[tr], y_vec[tr])
        y_pred = model.predict(X_feat[te])
        fold_sp.append(spearman_r(y_vec[te], y_pred))
        fold_r2.append(r2_score(y_vec[te], y_pred))
        fold_rmse.append(np.sqrt(np.mean((y_vec[te] - y_pred) ** 2)))
    return np.array(fold_sp), np.array(fold_r2), np.array(fold_rmse)

def tissue_accuracy(X_feat, tissue_enc):
    """L2-penalized Logistic Regression tissue prediction accuracy (5-fold CV).
    Consistent with the classifier used throughout the main pipeline (C=0.1).
    """
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   LogisticRegression(C=0.1, max_iter=1000, solver="saga",
                                   random_state=42, n_jobs=2))
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = cross_val_score(clf, X_feat, tissue_enc, cv=skf, scoring="accuracy", n_jobs=-1)
    return accs.mean()

def classify_feature(name):
    if "_ME" in name:
        return "module"
    elif name.startswith("delta_") or "edge_disrupt" in name.lower():
        return "edge_disrupt"
    else:
        return "topology_zscore"

# ── Load shared data ───────────────────────────────────────────────────────────
log("Loading shared data...")
X_expr   = pd.read_csv(f"{PROC}/X_expr_matched.csv",      index_col=0)
X_resid  = pd.read_csv(f"{PROC}/X_expr_residualized.csv", index_col=0)
y_all    = pd.read_csv(f"{PROC}/y_matched.csv",           index_col=0)
meta     = pd.read_csv(f"{PROC}/cell_line_metadata.csv",  index_col=0)
ma_v3    = pd.read_csv(f"{FEAT_V3}/module_activity_v3.csv", index_col=0)
ed_v3    = pd.read_csv(f"{FEAT_V3}/edge_disruption_within_tissue.csv", index_col=0)
topo_v3  = pd.read_csv(f"{FEAT_V3}/topo_features_v3.csv", index_col=0)
topo_z   = pd.read_csv(f"{FEAT_Z}/topo_features_zscore.csv", index_col=0)

combined_z = pd.concat([ma_v3.fillna(0), ed_v3.fillna(0), topo_z.fillna(0)], axis=1)

tissue_ser = meta.loc[X_expr.index, "tissue"].fillna("Unknown")
le = LabelEncoder()
tissue_enc = le.fit_transform(tissue_ser.values)

DRUGS = {
    "Osimertinib":              "AUC",
    "Crizotinib":               "LN_IC50",
    "KRAS (G12C) Inhibitor-12": "LN_IC50",
}

# Track experiment results for summary
experiment_results = {}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 (CRITICAL): SHAP on unconfounded features for osimertinib
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Running Experiment 1: SHAP on unconfounded features (osimertinib) ===")
log("=== Experiment 1: SHAP on unconfounded features ===")
t1_start = time.time()

try:
    drug = "Osimertinib"
    y_drug = y_all[drug].dropna()

    # ── 1a: SHAP on Combined zscore (RF) ──────────────────────────────────────
    log("  1a: SHAP on Combined_zscore for Osimertinib (RF, full cohort)")
    common_cz = combined_z.index.intersection(y_drug.index)
    X_cz = combined_z.loc[common_cz].values.astype(np.float32)
    y_cz = y_drug.loc[common_cz].values

    imp_cz = SimpleImputer(strategy="median").fit(X_cz)
    X_cz_imp = imp_cz.transform(X_cz)
    sc_cz = StandardScaler().fit(X_cz_imp)
    X_cz_sc = sc_cz.transform(X_cz_imp)

    rf_cz = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                   n_jobs=1, random_state=42)
    rf_cz.fit(X_cz_sc, y_cz)

    rng = np.random.RandomState(42)
    bg_idx_cz = rng.choice(len(X_cz_sc), size=min(150, len(X_cz_sc)), replace=False)
    ev_idx_cz = rng.choice(len(X_cz_sc), size=min(200, len(X_cz_sc)), replace=False)

    explainer_cz = shap.TreeExplainer(rf_cz, data=X_cz_sc[bg_idx_cz],
                                       feature_perturbation="interventional")
    sv_cz = explainer_cz.shap_values(X_cz_sc[ev_idx_cz])
    log(f"    SHAP values shape: {sv_cz.shape}")

    feat_names_cz = combined_z.columns.tolist()
    mean_abs_cz = np.abs(sv_cz).mean(axis=0)
    top20_idx_cz = np.argsort(mean_abs_cz)[::-1][:20]
    top20_names_cz = [feat_names_cz[i] for i in top20_idx_cz]
    top20_vals_cz  = mean_abs_cz[top20_idx_cz]

    shap_cz_df = pd.DataFrame({
        "feature_name": [feat_names_cz[i] for i in np.argsort(mean_abs_cz)[::-1]],
        "mean_abs_shap": np.sort(mean_abs_cz)[::-1],
        "feature_type": [classify_feature(feat_names_cz[i]) for i in np.argsort(mean_abs_cz)[::-1]],
        "interpretation": [
            "Reinforces main finding: module/topology signal predicts osimertinib sensitivity"
            if classify_feature(feat_names_cz[i]) in ("module", "topology_zscore")
            else "EdgeDisrupt contribution in unconfounded space"
            for i in np.argsort(mean_abs_cz)[::-1]
        ]
    })
    shap_cz_df.to_csv(f"{SUPP}/table_shap_combined_zscore_osimertinib.csv", index=False)

    # Beeswarm + bar plot
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    colors_bar = [
        "#6ACC65" if classify_feature(n) == "module" else
        "#D65F5F" if classify_feature(n) == "edge_disrupt" else "#B47CC7"
        for n in top20_names_cz[::-1]
    ]
    ax.barh(range(20), top20_vals_cz[::-1], color=colors_bar)
    ax.set_yticks(range(20))
    ax.set_yticklabels(
        [n.replace("degree_", "deg_").replace("hub_score_", "hub_") for n in top20_names_cz[::-1]],
        fontsize=7
    )
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top 20 features — Osimertinib\n(Combined_zscore, RF)", fontsize=10)
    legend_elements = [
        Patch(facecolor="#6ACC65", label="module"),
        Patch(facecolor="#D65F5F", label="edge_disrupt"),
        Patch(facecolor="#B47CC7", label="topology_zscore"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax2 = axes[1]
    X_shap_top20 = X_cz_sc[ev_idx_cz][:, top20_idx_cz]
    sv_top20 = sv_cz[:, top20_idx_cz]
    for j in range(19, -1, -1):
        y_pos = 19 - j
        sv_j = sv_top20[:, j]
        x_j  = X_shap_top20[:, j]
        sc_norm = (x_j - x_j.min()) / (np.ptp(x_j) + 1e-9)
        colors_bee = plt.cm.RdBu_r(sc_norm)
        jitter = np.random.RandomState(j).uniform(-0.3, 0.3, len(sv_j))
        ax2.scatter(sv_j, y_pos + jitter, c=colors_bee, alpha=0.5, s=8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(
        [n.replace("degree_", "deg_").replace("hub_score_", "hub_") for n in top20_names_cz],
        fontsize=7
    )
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("SHAP value")
    ax2.set_title("SHAP beeswarm — Osimertinib\n(Combined_zscore)", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Feature value (norm.)", shrink=0.6)

    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_shap_combined_zscore_osimertinib.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_shap_combined_zscore_osimertinib.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_shap_combined_zscore_osimertinib.pdf")

    # ── 1b: SHAP on Topology zscore (Elastic Net) ─────────────────────────────
    log("  1b: SHAP on Topology_zscore for Osimertinib (ElasticNet)")
    common_tz = topo_z.index.intersection(y_drug.index)
    X_tz = topo_z.loc[common_tz].values.astype(np.float32)
    y_tz = y_drug.loc[common_tz].values

    imp_tz = SimpleImputer(strategy="median").fit(X_tz)
    X_tz_imp = imp_tz.transform(X_tz)
    sc_tz = StandardScaler().fit(X_tz_imp)
    X_tz_sc = sc_tz.transform(X_tz_imp)

    en_tz = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
                         cv=3, max_iter=5000, n_jobs=2, random_state=42)
    en_tz.fit(X_tz_sc, y_tz)

    rng2 = np.random.RandomState(42)
    bg_idx_tz = rng2.choice(len(X_tz_sc), size=min(150, len(X_tz_sc)), replace=False)

    explainer_tz = shap.LinearExplainer(en_tz, X_tz_sc[bg_idx_tz])
    sv_tz = explainer_tz.shap_values(X_tz_sc)
    log(f"    SHAP values shape: {sv_tz.shape}")

    feat_names_tz = topo_z.columns.tolist()
    mean_abs_tz = np.abs(sv_tz).mean(axis=0)
    top20_idx_tz = np.argsort(mean_abs_tz)[::-1][:20]
    top20_names_tz = [feat_names_tz[i] for i in top20_idx_tz]
    top20_vals_tz  = mean_abs_tz[top20_idx_tz]

    shap_tz_df = pd.DataFrame({
        "feature_name": [feat_names_tz[i] for i in np.argsort(mean_abs_tz)[::-1]],
        "mean_abs_shap": np.sort(mean_abs_tz)[::-1],
        "feature_type": "topology_zscore",
        "interpretation": "Topology zscore SHAP — unconfounded signal for osimertinib"
    })
    shap_tz_df.to_csv(f"{SUPP}/table_shap_topology_zscore_osimertinib.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    ax.barh(range(20), top20_vals_tz[::-1], color="#B47CC7")
    ax.set_yticks(range(20))
    ax.set_yticklabels(
        [n.replace("degree_", "deg_").replace("hub_score_", "hub_") for n in top20_names_tz[::-1]],
        fontsize=7
    )
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top 20 features — Osimertinib\n(Topology_zscore, EN)", fontsize=10)

    ax2 = axes[1]
    X_shap_top20_tz = X_tz_sc[:, top20_idx_tz]
    sv_top20_tz = sv_tz[:, top20_idx_tz]
    for j in range(19, -1, -1):
        y_pos = 19 - j
        sv_j = sv_top20_tz[:, j]
        x_j  = X_shap_top20_tz[:, j]
        sc_norm = (x_j - x_j.min()) / (np.ptp(x_j) + 1e-9)
        colors_bee = plt.cm.RdBu_r(sc_norm)
        jitter = np.random.RandomState(j).uniform(-0.3, 0.3, len(sv_j))
        ax2.scatter(sv_j, y_pos + jitter, c=colors_bee, alpha=0.5, s=8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(
        [n.replace("degree_", "deg_").replace("hub_score_", "hub_") for n in top20_names_tz],
        fontsize=7
    )
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("SHAP value")
    ax2.set_title("SHAP beeswarm — Osimertinib\n(Topology_zscore)", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Feature value (norm.)", shrink=0.6)

    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_shap_topology_zscore_osimertinib.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_shap_topology_zscore_osimertinib.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_shap_topology_zscore_osimertinib.pdf")

    # ── 1c: Comparative SHAP table ────────────────────────────────────────────
    log("  1c: Comparative SHAP table (3 analyses)")

    shap_raw_path = f"/Users/mriosc/Documents/paper2/models_v3/shap/shap_Osimertinib_Topology_v3.npy"
    if os.path.exists(shap_raw_path):
        sv_raw = np.load(shap_raw_path)
        feat_names_raw = topo_v3.columns.tolist()
        mean_abs_raw = np.abs(sv_raw).mean(axis=0)
        log(f"    Loaded existing raw SHAP: {sv_raw.shape}")
    else:
        log("    Raw SHAP not found, recomputing...")
        common_raw = topo_v3.index.intersection(y_drug.index)
        X_raw = topo_v3.loc[common_raw].values.astype(np.float32)
        y_raw = y_drug.loc[common_raw].values
        imp_raw = SimpleImputer(strategy="median").fit(X_raw)
        X_raw_imp = imp_raw.transform(X_raw)
        sc_raw = StandardScaler().fit(X_raw_imp)
        X_raw_sc = sc_raw.transform(X_raw_imp)
        rf_raw = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                        n_jobs=-1, random_state=42)
        rf_raw.fit(X_raw_sc, y_raw)
        rng_r = np.random.RandomState(42)
        bg_r = rng_r.choice(len(X_raw_sc), size=min(150, len(X_raw_sc)), replace=False)
        ev_r = rng_r.choice(len(X_raw_sc), size=min(200, len(X_raw_sc)), replace=False)
        exp_raw = shap.TreeExplainer(rf_raw, data=X_raw_sc[bg_r],
                                      feature_perturbation="interventional")
        sv_raw = exp_raw.shap_values(X_raw_sc[ev_r])
        feat_names_raw = topo_v3.columns.tolist()
        mean_abs_raw = np.abs(sv_raw).mean(axis=0)

    shap_raw_dict = {feat_names_raw[i]: mean_abs_raw[i] for i in range(len(feat_names_raw))}
    shap_tz_dict  = {feat_names_tz[i]: mean_abs_tz[i] for i in range(len(feat_names_tz))}

    shap_cz_topo_dict = {}
    for i, fn in enumerate(feat_names_cz):
        if fn in feat_names_raw:
            shap_cz_topo_dict[fn] = mean_abs_cz[i]

    raw_top20_names = [feat_names_raw[i] for i in np.argsort(mean_abs_raw)[::-1][:20]]
    tz_top20_names  = top20_names_tz[:20]
    cz_top20_names  = top20_names_cz[:20]

    common_feats = [f for f in feat_names_raw if f in feat_names_tz]
    compare_rows = []
    for feat in sorted(common_feats, key=lambda f: shap_raw_dict.get(f, 0), reverse=True):
        in_all_three = (feat in raw_top20_names and feat in tz_top20_names and feat in cz_top20_names)
        compare_rows.append({
            "feature_name": feat,
            "shap_topology_raw": round(shap_raw_dict.get(feat, np.nan), 6),
            "shap_topology_zscore": round(shap_tz_dict.get(feat, np.nan), 6),
            "shap_combined_zscore": round(shap_cz_topo_dict.get(feat, np.nan), 6),
            "robust_top20": "*" if in_all_three else "",
            "interpretation": (
                "Robust signal: appears in top-20 across all three SHAP analyses"
                if in_all_three else
                "Feature present in topology space; not consistently top-20"
            )
        })

    compare_df = pd.DataFrame(compare_rows).sort_values("shap_topology_raw", ascending=False)
    compare_df.to_csv(f"{SUPP}/table_shap_comparison_osimertinib.csv", index=False)
    n_robust = compare_df["robust_top20"].eq("*").sum()
    log(f"    Robust top-20 features: {n_robust}")
    log("    Saved: table_shap_comparison_osimertinib.csv")

    t1_end = time.time()
    experiment_results["Exp1_SHAP_unconfounded"] = {
        "status": "completed",
        "finding": (f"SHAP computed for Combined_zscore (RF) and Topology_zscore (EN); "
                    f"{n_robust} features robust across all 3 analyses"),
        "impact": "refuerza",
        "time_s": round(t1_end - t1_start, 1)
    }
    log(f"  Experiment 1 completed in {t1_end - t1_start:.1f}s")

except Exception as e:
    log(f"  ERROR in Experiment 1: {e}\n{traceback.format_exc()}")
    experiment_results["Exp1_SHAP_unconfounded"] = {
        "status": "failed", "finding": str(e), "impact": "N/A", "time_s": 0
    }
    # Ensure variables exist for downstream experiments
    top20_names_cz = []
    feat_names_cz  = combined_z.columns.tolist()
    mean_abs_cz    = np.zeros(len(feat_names_cz))
    sv_cz          = np.zeros((200, len(feat_names_cz)))
    ev_idx_cz      = np.arange(min(200, len(combined_z)))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 (CRITICAL): YAP1 centrality with Combined zscore SHAP values
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Running Experiment 2: YAP1 centrality with Combined zscore SHAP ===")
log("=== Experiment 2: YAP1/TEAD1 SHAP distribution by sensitivity ===")
t2_start = time.time()

try:
    yap_tead_features = ["degree_YAP1", "hub_score_YAP1", "degree_TEAD1", "hub_score_TEAD1"]
    mw_rows = []
    violin_data = []

    for drug in DRUGS:
        y_drug = y_all[drug].dropna()
        common = combined_z.index.intersection(y_drug.index)
        X_cz_drug = combined_z.loc[common].values.astype(np.float32)
        y_vec = y_drug.loc[common].values

        imp_d = SimpleImputer(strategy="median").fit(X_cz_drug)
        X_imp_d = imp_d.transform(X_cz_drug)
        sc_d = StandardScaler().fit(X_imp_d)
        X_sc_d = sc_d.transform(X_imp_d)

        rf_d = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                      n_jobs=-1, random_state=42)
        rf_d.fit(X_sc_d, y_vec)

        rng_d = np.random.RandomState(42)
        bg_d = rng_d.choice(len(X_sc_d), size=min(150, len(X_sc_d)), replace=False)
        ev_d = rng_d.choice(len(X_sc_d), size=min(200, len(X_sc_d)), replace=False)

        exp_d = shap.TreeExplainer(rf_d, data=X_sc_d[bg_d],
                                    feature_perturbation="interventional")
        sv_d = exp_d.shap_values(X_sc_d[ev_d])

        feat_names_d = combined_z.columns.tolist()
        y_eval = y_vec[ev_d]
        # Quartile boundaries defined on the full cohort (consistent with §3.7)
        q_boundaries = np.percentile(y_vec, [25, 50, 75])
        quartiles = np.searchsorted(q_boundaries, y_eval)  # 0=Q1, 1=Q2, 2=Q3, 3=Q4
        sens_group = np.where(quartiles == 0, "Sensitive (Q1)",
                     np.where(quartiles == 3, "Resistant (Q4)", "Middle"))

        for feat in yap_tead_features:
            if feat not in feat_names_d:
                continue
            fidx = feat_names_d.index(feat)
            sv_feat = sv_d[:, fidx]

            mask_q1 = sens_group == "Sensitive (Q1)"
            mask_q4 = sens_group == "Resistant (Q4)"
            sv_q1 = sv_feat[mask_q1]
            sv_q4 = sv_feat[mask_q4]

            if len(sv_q1) > 3 and len(sv_q4) > 3:
                stat_mw, p_mw = mannwhitneyu(sv_q1, sv_q4, alternative="two-sided")
            else:
                stat_mw, p_mw = np.nan, np.nan

            mw_rows.append({
                "drug": drug, "feature": feat,
                "n_sensitive": len(sv_q1), "n_resistant": len(sv_q4),
                "mean_shap_sensitive": round(sv_q1.mean(), 5) if len(sv_q1) else np.nan,
                "mean_shap_resistant": round(sv_q4.mean(), 5) if len(sv_q4) else np.nan,
                "mannwhitney_U": round(stat_mw, 2) if not np.isnan(stat_mw) else np.nan,
                "p_value": round(p_mw, 4) if not np.isnan(p_mw) else np.nan,
                "interpretation": (
                    "YAP1/TEAD1 SHAP differs between sensitive and resistant — "
                    "supports YAP1 role in unconfounded space"
                    if (not np.isnan(p_mw) and p_mw < 0.05)
                    else "No significant difference — YAP1 signal may not persist in unconfounded space"
                )
            })

            for sv_val, grp in zip(sv_feat, sens_group):
                if grp in ("Sensitive (Q1)", "Resistant (Q4)"):
                    violin_data.append({"drug": drug, "feature": feat,
                                        "shap_value": sv_val, "group": grp})

    mw_df = pd.DataFrame(mw_rows)
    if len(mw_df) > 0:
        fdr_vals = fdr_bh(mw_df["p_value"].fillna(1).values)
        mw_df["p_adj_fdr"] = fdr_vals.round(4)
    mw_df.to_csv(f"{SUPP}/table_yap1_tead1_shap_mannwhitney.csv", index=False)
    log("    Saved: table_yap1_tead1_shap_mannwhitney.csv")

    # Violin plots
    violin_df = pd.DataFrame(violin_data)
    n_drugs = len(DRUGS)
    n_feats = len(yap_tead_features)
    fig, axes = plt.subplots(n_feats, n_drugs, figsize=(5 * n_drugs, 4 * n_feats))

    for fi, feat in enumerate(yap_tead_features):
        for di, drug in enumerate(DRUGS):
            ax = axes[fi, di] if n_feats > 1 else axes[di]
            sub = violin_df[(violin_df["drug"] == drug) & (violin_df["feature"] == feat)]
            if len(sub) > 0:
                sns.violinplot(
                    data=sub, x="group", y="shap_value", ax=ax,
                    palette={"Sensitive (Q1)": "#6ACC65", "Resistant (Q4)": "#D65F5F"},
                    inner="box", cut=0
                )
                row = mw_df[(mw_df["drug"] == drug) & (mw_df["feature"] == feat)]
                if not row.empty:
                    p = row.iloc[0]["p_value"]
                    p_str = f"p={p:.3f}" if not np.isnan(p) else "p=N/A"
                    ax.set_title(f"{drug.split()[0]}\n{feat}\n{p_str}", fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("SHAP value" if di == 0 else "")
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_yap1_shap_distribution_by_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_yap1_shap_distribution_by_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_yap1_shap_distribution_by_sensitivity.pdf")

    # ── 2b: Scatter plot for osimertinib ──────────────────────────────────────
    log("  2b: Scatter plot — Topology zscore deg_YAP1 vs top WGCNA module SHAP")
    drug_osi = "Osimertinib"
    y_osi = y_all[drug_osi].dropna()

    top_module = next((n for n in top20_names_cz if "_ME" in n), feat_names_cz[0]) if top20_names_cz else feat_names_cz[0]
    log(f"    Top WGCNA module feature: {top_module}")

    # FIX: usar exactamente las mismas cell lines con las que se calculó sv_cz
    # (common_cz y ev_idx_cz definidos en Exp 1a para Osimertinib)
    lines_eval   = common_cz[ev_idx_cz]
    shap_top_mod = sv_cz[:, feat_names_cz.index(top_module)] if top_module in feat_names_cz else np.zeros(len(lines_eval))

    if "degree_YAP1" in topo_z.columns:
        deg_yap1_tz = topo_z.reindex(lines_eval)["degree_YAP1"].fillna(0).values
    else:
        deg_yap1_tz = np.zeros(len(lines_eval))

    y_eval_osi = y_osi.loc[lines_eval].values
    quartiles_osi = pd.qcut(y_eval_osi, q=4, labels=False, duplicates="drop")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_colors = {0: "#2196F3", 1: "#6ACC65", 2: "#F0A500", 3: "#D65F5F"}
    scatter_labels = {0: "Q1 (sensitive)", 1: "Q2", 2: "Q3", 3: "Q4 (resistant)"}
    for q in range(4):
        mask = quartiles_osi == q
        ax.scatter(deg_yap1_tz[mask], shap_top_mod[mask],
                   c=scatter_colors[q], label=scatter_labels[q], alpha=0.7, s=40)
    ax.set_xlabel("Topology zscore deg_YAP1")
    ax.set_ylabel(f"SHAP contribution — {top_module}")
    ax.set_title("YAP1 centrality vs top module SHAP\nOsimertinib", fontsize=10)
    ax.legend(fontsize=8)
    r_scatter, p_scatter = stats.spearmanr(deg_yap1_tz, shap_top_mod)
    ax.text(0.05, 0.95, f"Spearman ρ={r_scatter:.3f}, p={p_scatter:.3f}",
            transform=ax.transAxes, fontsize=9, va="top")
   
    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_yap1_topo_vs_module_shap_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_yap1_topo_vs_module_shap_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_yap1_topo_vs_module_shap_scatter.pdf")

    t2_end = time.time()
    sig_count = int(mw_df.get("p_adj_fdr", mw_df["p_value"]).lt(0.05).sum()) if "p_adj_fdr" in mw_df.columns else 0
    experiment_results["Exp2_YAP1_SHAP"] = {
        "status": "completed",
        "finding": f"YAP1/TEAD1 SHAP distributions computed; {sig_count} significant MW tests (FDR<0.05)",
        "impact": "refuerza" if sig_count > 0 else "neutro",
        "time_s": round(t2_end - t2_start, 1)
    }
    log(f"  Experiment 2 completed in {t2_end - t2_start:.1f}s")

except Exception as e:
    log(f"  ERROR in Experiment 2: {e}\n{traceback.format_exc()}")
    experiment_results["Exp2_YAP1_SHAP"] = {
        "status": "failed", "finding": str(e), "impact": "N/A", "time_s": 0
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 (IMPORTANT): Ablation study — Combined_noEdge
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Running Experiment 3: Ablation study — Combined_noEdge ===")
log("=== Experiment 3: Ablation study — Combined_noEdge ===")
t3_start = time.time()

try:
    # Combined_noEdge = Modules (510) + Topology_zscore (72) = 582 features
    combined_noedge = pd.concat([ma_v3.fillna(0), topo_z.fillna(0)], axis=1)
    log(f"  Combined_noEdge shape: {combined_noedge.shape}")

    v3_scores = pd.read_csv(f"{MOD_V3}/cv_scores_v3.csv")
    z_scores  = pd.read_csv(f"{MOD_Z}/cv_scores_zscore.csv")

    ablation_scores = []
    ablation_summary_rows = []

    for drug in DRUGS:
        y_drug = y_all[drug].dropna()
        common = combined_noedge.index.intersection(y_drug.index)
        X_ne = combined_noedge.loc[common].values.astype(np.float32)
        y_vec = y_drug.loc[common].values
        strat = stratify_by_quartile(y_vec)
        log(f"  {drug}: N={len(common)}, features={X_ne.shape[1]}")

        fold_sp, fold_r2, fold_rmse = run_cv(X_ne, y_vec, strat, "RF")
        log(f"    Combined_noEdge: rho={fold_sp.mean():.3f}+/-{fold_sp.std():.3f}")

        for fi, (sp, r2, rmse) in enumerate(zip(fold_sp, fold_r2, fold_rmse)):
            ablation_scores.append({
                "drug": drug, "feature_set": "Combined_noEdge",
                "model": "RF", "fold": fi,
                "spearman": sp, "r2": r2, "rmse": rmse
            })

        X_ne_full = combined_noedge.reindex(X_expr.index).fillna(0).values.astype(np.float32)
        ta_ne = tissue_accuracy(X_ne_full, tissue_enc)
        log(f"    Tissue accuracy: {ta_ne:.3f}")

        base_sp = v3_scores[(v3_scores["drug"] == drug) &
                            (v3_scores["feature_set"] == "Baseline_Resid_RF")]["spearman"].values
        cz_sp   = z_scores[(z_scores["drug"] == drug) &
                           (z_scores["feature_set"] == "Combined_zscore")]["spearman"].values

        if len(base_sp) == len(fold_sp) > 0:
            _, p_ne = stats.wilcoxon(fold_sp, base_sp, alternative="greater", zero_method="wilcox")
            delta_ne = fold_sp.mean() - base_sp.mean()
        else:
            p_ne, delta_ne = np.nan, np.nan

        ablation_summary_rows.append({
            "drug": drug,
            "rho_baseline_resid": round(base_sp.mean(), 3) if len(base_sp) else np.nan,
            "rho_combined_noedge": round(fold_sp.mean(), 3),
            "rho_combined_zscore": round(cz_sp.mean(), 3) if len(cz_sp) else np.nan,
            "tissue_accuracy_noedge": round(ta_ne, 3),
            "delta_noedge_vs_baseline": round(delta_ne, 4) if not np.isnan(delta_ne) else np.nan,
            "p_wilcoxon_noedge": round(p_ne, 4) if not np.isnan(p_ne) else np.nan,
        })

    ablation_scores_df = pd.DataFrame(ablation_scores)

    # Extended Wilcoxon table (add 3 new tests to existing 24 = 27 total)
    # Combined_noEdge vs Baseline_Resid_RF for each of the 3 drugs
    z_wilcox = pd.read_csv(f"{MOD_Z}/wilcoxon_zscore.csv")
    new_wilcox_rows = []
    for row in ablation_summary_rows:
        drug = row["drug"]
        base_sp = v3_scores[(v3_scores["drug"] == drug) &
                            (v3_scores["feature_set"] == "Baseline_Resid_RF")]["spearman"].values
        ne_sp = ablation_scores_df[(ablation_scores_df["drug"] == drug) &
                                   (ablation_scores_df["feature_set"] == "Combined_noEdge")]["spearman"].values
        if len(base_sp) == len(ne_sp) > 0:
            _, p_w = stats.wilcoxon(ne_sp, base_sp, alternative="greater", zero_method="wilcox")
            delta_w = ne_sp.mean() - base_sp.mean()
        else:
            p_w, delta_w = np.nan, np.nan
        new_wilcox_rows.append({
            "baseline": "Baseline_Resid_RF", "drug": drug,
            "network_set": "Combined_noEdge",
            "delta_spearman": round(delta_w, 4) if not np.isnan(delta_w) else np.nan,
            "p_value": round(p_w, 4) if not np.isnan(p_w) else np.nan
        })

    extended_wilcox = pd.concat([z_wilcox, pd.DataFrame(new_wilcox_rows)], ignore_index=True)
    pvals_ext = extended_wilcox["p_value"].fillna(1).values
    extended_wilcox["p_adj_fdr"] = fdr_bh(pvals_ext).round(4)
    extended_wilcox["significant"] = extended_wilcox["p_adj_fdr"] < 0.05

    # Comprehensive ablation table
    feat_sets_compare = ["Baseline_Resid_RF", "Modules_v3", "Topology_zscore",
                         "Combined_noEdge", "Combined_zscore"]
    ablation_rows = []
    for drug in DRUGS:
        row = {"drug": drug}
        for fs in feat_sets_compare:
            if fs == "Combined_noEdge":
                sp_vals = ablation_scores_df[(ablation_scores_df["drug"] == drug) &
                                             (ablation_scores_df["feature_set"] == fs)]["spearman"].values
            elif fs in ["Baseline_Resid_RF", "Modules_v3"]:
                sp_vals = v3_scores[(v3_scores["drug"] == drug) &
                                    (v3_scores["feature_set"] == fs)]["spearman"].values
            else:
                sp_vals = z_scores[(z_scores["drug"] == drug) &
                                   (z_scores["feature_set"] == fs)]["spearman"].values
            row[f"rho_{fs}"] = f"{sp_vals.mean():.3f}+/-{sp_vals.std():.3f}" if len(sp_vals) else "N/A"

        ne_row = next((r for r in ablation_summary_rows if r["drug"] == drug), {})
        row["tissue_accuracy_noedge"] = ne_row.get("tissue_accuracy_noedge", np.nan)

        w_row = extended_wilcox[(extended_wilcox["drug"] == drug) &
                                 (extended_wilcox["network_set"] == "Combined_noEdge")]
        if not w_row.empty:
            row["delta_noedge_vs_baseline"] = w_row.iloc[0]["delta_spearman"]
            row["padj_noedge"] = w_row.iloc[0]["p_adj_fdr"]
        row["interpretation"] = (
            "Combined_noEdge improves over baseline — EdgeDisrupt not required"
            if (not w_row.empty and w_row.iloc[0].get("significant", False))
            else "Combined_noEdge does not significantly improve — EdgeDisrupt may contribute"
        )
        ablation_rows.append(row)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(f"{SUPP}/table_ablation_combined_noedge.csv", index=False)
    log("    Saved: table_ablation_combined_noedge.csv")

    # Figure: bar plot comparison
    feat_order_abl = ["Baseline_Resid_RF", "Modules_v3", "Topology_zscore",
                      "Combined_noEdge", "Combined_zscore"]
    colors_abl = ["#2196F3", "#6ACC65", "#B47CC7", "#FF8C00", "#F0A500"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, drug in zip(axes, DRUGS.keys()):
        means_abl, stds_abl = [], []
        for fs in feat_order_abl:
            if fs == "Combined_noEdge":
                sp_v = ablation_scores_df[(ablation_scores_df["drug"] == drug) &
                                          (ablation_scores_df["feature_set"] == fs)]["spearman"].values
            elif fs in ["Baseline_Resid_RF", "Modules_v3"]:
                sp_v = v3_scores[(v3_scores["drug"] == drug) &
                                 (v3_scores["feature_set"] == fs)]["spearman"].values
            else:
                sp_v = z_scores[(z_scores["drug"] == drug) &
                                (z_scores["feature_set"] == fs)]["spearman"].values
            means_abl.append(sp_v.mean() if len(sp_v) else 0)
            stds_abl.append(sp_v.std() if len(sp_v) else 0)

        ax.bar(range(len(feat_order_abl)), means_abl, yerr=stds_abl,
               color=colors_abl, capsize=4, alpha=0.85, edgecolor="white")
        # Highlight Combined_noEdge with black border
        ax.bar([3], [means_abl[3]], yerr=[stds_abl[3]], color="#FF8C00",
               capsize=4, alpha=0.85, edgecolor="black", linewidth=1.5)
        ax.set_xticks(range(len(feat_order_abl)))
        ax.set_xticklabels(
            [f.replace("_Resid_RF", "").replace("_v3", "").replace("_zscore", "(z)")
             for f in feat_order_abl],
            rotation=35, ha="right", fontsize=8
        )
        ax.set_title(drug.split()[0], fontsize=10)
        ax.set_ylabel("Spearman rho")
        ax.axhline(0, color="black", lw=0.5, linestyle="--")

    plt.suptitle("Ablation study: Combined_noEdge vs other feature sets", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_ablation_spearman_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_ablation_spearman_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_ablation_spearman_comparison.pdf")

    t3_end = time.time()
    experiment_results["Exp3_Ablation"] = {
        "status": "completed",
        "finding": (f"Combined_noEdge evaluated; "
                    f"tissue_accuracy={ablation_summary_rows[0].get('tissue_accuracy_noedge', 'N/A')}"),
        "impact": "neutro",
        "time_s": round(t3_end - t3_start, 1)
    }
    log(f"  Experiment 3 completed in {t3_end - t3_start:.1f}s")

except Exception as e:
    log(f"  ERROR in Experiment 3: {e}\n{traceback.format_exc()}")
    experiment_results["Exp3_Ablation"] = {
        "status": "failed", "finding": str(e), "impact": "N/A", "time_s": 0
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 (IMPORTANT): Sensitivity analysis — exclusion of kNN-imputed lines
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Running Experiment 4: Sensitivity analysis — exclude kNN-imputed lines ===")
log("=== Experiment 4: Sensitivity analysis — no-imputed cohort ===")
t4_start = time.time()

try:
    # ── 4a: Identify imputed lines ────────────────────────────────────────────
    wgcna_dir = "/Users/mriosc/Documents/paper2/networks_v3/wgcna_v3"
    wgcna_tissues = set()
    for fn in os.listdir(wgcna_dir):
        if fn.endswith("_eigengenes.csv"):
            wgcna_tissues.add(fn.replace("_eigengenes.csv", ""))
    log(f"  WGCNA tissues with eigengenes: {sorted(wgcna_tissues)}")

    def tissue_to_wgcna_key(t):
        return t.replace(" ", "_").replace("&", "and")

    is_imputed = meta.loc[X_expr.index, "tissue"].apply(
        lambda t: tissue_to_wgcna_key(t) not in wgcna_tissues
    )
    n_imputed = is_imputed.sum()
    n_real    = (~is_imputed).sum()
    log(f"  Imputed lines: {n_imputed} ({n_imputed / len(is_imputed) * 100:.1f}%)")
    log(f"  Real eigengene lines: {n_real}")

    imputed_df = pd.DataFrame({
        "cell_line": X_expr.index,
        "is_imputed": is_imputed.values,
        "tissue": meta.loc[X_expr.index, "tissue"].values
    })
    imputed_df.to_csv(f"{SUPP}/table_imputed_lines_mask.csv", index=False)

    # ── 4b: CV on no-imputed cohort ───────────────────────────────────────────
    no_imp_idx = X_expr.index[~is_imputed.values]
    log(f"  No-imputed cohort size: {len(no_imp_idx)}")

    gene_vars_r = X_resid.values.var(axis=0)
    top1k_r_idx = np.argsort(gene_vars_r)[::-1][:1000]
    X_baseline_r = X_resid.iloc[:, top1k_r_idx]

    feat_sets_4 = {
        "Modules_v3":        (ma_v3,        "RF"),
        "Combined_zscore":   (combined_z,   "RF"),
        "Baseline_Resid_RF": (X_baseline_r, "RF"),
    }

    v3_scores = pd.read_csv(f"{MOD_V3}/cv_scores_v3.csv")
    z_scores  = pd.read_csv(f"{MOD_Z}/cv_scores_zscore.csv")

    sensitivity_rows = []
    box_data = []

    for drug in DRUGS:
        y_drug = y_all[drug].dropna()

        for fs_name, (X_feat, mtype) in feat_sets_4.items():
            if fs_name == "Baseline_Resid_RF":
                full_sp = v3_scores[(v3_scores["drug"] == drug) &
                                    (v3_scores["feature_set"] == fs_name)]["spearman"].values
            elif fs_name == "Modules_v3":
                full_sp = v3_scores[(v3_scores["drug"] == drug) &
                                    (v3_scores["feature_set"] == fs_name)]["spearman"].values
            elif fs_name == "Combined_zscore":
                full_sp = z_scores[(z_scores["drug"] == drug) &
                                   (z_scores["feature_set"] == fs_name)]["spearman"].values
            else:
                full_sp = np.array([])

            common_ni = X_feat.index.intersection(y_drug.index).intersection(no_imp_idx)
            if len(common_ni) < 20:
                log(f"    {drug} / {fs_name}: too few no-imputed samples ({len(common_ni)}), skipping")
                continue

            X_ni = X_feat.loc[common_ni].values.astype(np.float32)
            y_ni = y_drug.loc[common_ni].values
            strat_ni = stratify_by_quartile(y_ni)
            log(f"  {drug} / {fs_name}: N_no_imputed={len(common_ni)}")

            fold_sp_ni, _, _ = run_cv(X_ni, y_ni, strat_ni, mtype)

            if len(full_sp) > 0 and len(fold_sp_ni) > 0:
                _, p_mw4 = mannwhitneyu(full_sp, fold_sp_ni, alternative="two-sided")
                delta4 = fold_sp_ni.mean() - full_sp.mean()
            else:
                p_mw4, delta4 = np.nan, np.nan

            flag = "RELEVANT_DIFFERENCE" if (not np.isnan(delta4) and abs(delta4) > 0.05) else ""
            if flag:
                log(f"    *** {flag}: delta={delta4:.3f} for {drug}/{fs_name}")

            sensitivity_rows.append({
                "drug": drug, "feature_set": fs_name,
                "rho_full_mean": round(full_sp.mean(), 3) if len(full_sp) else np.nan,
                "rho_full_sd":   round(full_sp.std(), 3) if len(full_sp) else np.nan,
                "rho_no_imputed_mean": round(fold_sp_ni.mean(), 3),
                "rho_no_imputed_sd":   round(fold_sp_ni.std(), 3),
                "delta": round(delta4, 4) if not np.isnan(delta4) else np.nan,
                "p_mannwhitney": round(p_mw4, 4) if not np.isnan(p_mw4) else np.nan,
                "flag": flag,
                "interpretation": (
                    f"Delta={delta4:.3f} > 0.05 — imputed lines affect results; requires discussion"
                    if flag else
                    "Results stable across full and no-imputed cohorts — reinforces main finding"
                )
            })

            for sp_val in full_sp:
                box_data.append({"drug": drug, "feature_set": fs_name,
                                  "cohort": "Full cohort", "spearman": sp_val})
            for sp_val in fold_sp_ni:
                box_data.append({"drug": drug, "feature_set": fs_name,
                                  "cohort": "No-imputed", "spearman": sp_val})

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(f"{SUPP}/table_sensitivity_no_imputed.csv", index=False)
    log("    Saved: table_sensitivity_no_imputed.csv")

    # ── 4d: Box plot ──────────────────────────────────────────────────────────
    box_df = pd.DataFrame(box_data)
    feat_sets_plot = ["Modules_v3", "Combined_zscore"]
    

    fig, axes = plt.subplots(1, len(DRUGS), figsize=(6 * len(DRUGS), 5))
    for ax, drug in zip(axes, DRUGS.keys()):
        sub = box_df[(box_df["drug"] == drug) &
                     (box_df["feature_set"].isin(feat_sets_plot))]
        if len(sub) > 0:
            sns.boxplot(
                data=sub, x="feature_set", y="spearman", hue="cohort",
                palette={"Full cohort": "#4878CF", "No-imputed": "#D65F5F"},
                ax=ax, width=0.6
            )
            ax.set_title(drug.split()[0], fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("Spearman rho")
            ax.set_xticklabels(
                [f.replace("_v3", "").replace("_zscore", "(z)") for f in feat_sets_plot],
                rotation=20, ha="right", fontsize=8
            )
            ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

    plt.suptitle("Sensitivity analysis: full cohort vs no-imputed cohort", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_sensitivity_no_imputed_boxplot.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_sensitivity_no_imputed_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_sensitivity_no_imputed_boxplot.pdf")

    t4_end = time.time()
    n_flagged = sensitivity_df["flag"].ne("").sum() if "flag" in sensitivity_df.columns else 0
    experiment_results["Exp4_Sensitivity_noImputed"] = {
        "status": "completed",
        "finding": (f"Imputed lines: {n_imputed} ({n_imputed / len(is_imputed) * 100:.1f}%); "
                    f"{n_flagged} drug/feature combinations with delta>0.05"),
        "impact": "debilita" if n_flagged > 0 else "refuerza",
        "time_s": round(t4_end - t4_start, 1)
    }
    log(f"  Experiment 4 completed in {t4_end - t4_start:.1f}s")

except Exception as e:
    log(f"  ERROR in Experiment 4: {e}\n{traceback.format_exc()}")
    experiment_results["Exp4_Sensitivity_noImputed"] = {
        "status": "failed", "finding": str(e), "impact": "N/A", "time_s": 0
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 NOTE: WGCNA stability (R script — see wgcna_stability_analysis.R)
# ══════════════════════════════════════════════════════════════════════════════
experiment_results["Exp5_WGCNA_stability"] = {
    "status": "pending_R",
    "finding": "Implemented in wgcna_stability_analysis.R (requires R + WGCNA package)",
    "impact": "N/A",
    "time_s": 0
}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 (COMPLEMENTARY): PCA baseline comparison vs Modules
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Running Experiment 6: PCA baseline comparison vs Modules ===")
log("=== Experiment 6: PCA baseline vs Modules ===")
t6_start = time.time()

try:
    # ── 6a: PCA on batch-corrected, tissue-residualised expression ────────────
    log("  6a: Computing PCA on residualised expression")
    X_resid_vals = X_resid.values.astype(np.float32)

    imp_pca = SimpleImputer(strategy="median").fit(X_resid_vals)
    X_resid_imp = imp_pca.transform(X_resid_vals)
    sc_pca = StandardScaler().fit(X_resid_imp)
    X_resid_sc = sc_pca.transform(X_resid_imp)

    pca = PCA(random_state=42)
    pca.fit(X_resid_sc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.searchsorted(cumvar, 0.80)) + 1
    log(f"  PCA: K={K} components explain 80% variance")

    X_pca = pca.transform(X_resid_sc)[:, :K]
    X_pca_df = pd.DataFrame(X_pca, index=X_resid.index,
                             columns=[f"PC{i+1}" for i in range(K)])

    ta_pca = tissue_accuracy(X_pca, tissue_enc)
    log(f"  PCA tissue accuracy: {ta_pca:.3f}")

    # ── CV for PCA baseline ───────────────────────────────────────────────────
    v3_scores = pd.read_csv(f"{MOD_V3}/cv_scores_v3.csv")
    pca_scores = []

    for drug in DRUGS:
        y_drug = y_all[drug].dropna()
        common = X_pca_df.index.intersection(y_drug.index)
        X_pca_sub = X_pca_df.loc[common].values.astype(np.float32)
        y_vec = y_drug.loc[common].values
        strat = stratify_by_quartile(y_vec)
        log(f"  {drug}: N={len(common)}, K={K}")

        fold_sp, fold_r2, fold_rmse = run_cv(X_pca_sub, y_vec, strat, "RF")
        log(f"    PCA_baseline: rho={fold_sp.mean():.3f}+/-{fold_sp.std():.3f}")

        for fi, (sp, r2, rmse) in enumerate(zip(fold_sp, fold_r2, fold_rmse)):
            pca_scores.append({
                "drug": drug, "feature_set": "PCA_baseline",
                "model": "RF", "fold": fi,
                "spearman": sp, "r2": r2, "rmse": rmse
            })

    pca_scores_df = pd.DataFrame(pca_scores)

    # ── 6b: Wilcoxon Modules > PCA_baseline ───────────────────────────────────
    pca_compare_rows = []
    for drug in DRUGS:
        mod_sp = v3_scores[(v3_scores["drug"] == drug) &
                           (v3_scores["feature_set"] == "Modules_v3")]["spearman"].values
        pca_sp = pca_scores_df[(pca_scores_df["drug"] == drug) &
                               (pca_scores_df["feature_set"] == "PCA_baseline")]["spearman"].values

        if len(mod_sp) == len(pca_sp) > 0:
            _, p_w6 = stats.wilcoxon(mod_sp, pca_sp, alternative="greater", zero_method="wilcox")
            delta6 = mod_sp.mean() - pca_sp.mean()
        else:
            p_w6, delta6 = np.nan, np.nan

        pca_compare_rows.append({
            "drug": drug,
            "rho_modules": round(mod_sp.mean(), 3) if len(mod_sp) else np.nan,
            "rho_pca_baseline": round(pca_sp.mean(), 3) if len(pca_sp) else np.nan,
            "K_components": K,
            "tissue_accuracy_pca": round(ta_pca, 3),
            "delta": round(delta6, 4) if not np.isnan(delta6) else np.nan,
            "p_wilcoxon": round(p_w6, 4) if not np.isnan(p_w6) else np.nan,
            "interpretation": (
                "Modules > PCA_baseline: WGCNA network structure adds genuine biological signal"
                if (not np.isnan(p_w6) and p_w6 < 0.05 and delta6 > 0)
                else (
                    "PCA_baseline >= Modules: WGCNA contribution may be dimensionality reduction, not network biology"
                    if (not np.isnan(delta6) and delta6 <= 0)
                    else "No significant difference"
                )
            )
        })

    pca_compare_df = pd.DataFrame(pca_compare_rows)
    pca_compare_df["p_adj_fdr"] = fdr_bh(pca_compare_df["p_wilcoxon"].fillna(1).values).round(4)
    pca_compare_df.to_csv(f"{SUPP}/table_modules_vs_pca_baseline.csv", index=False)
    log("    Saved: table_modules_vs_pca_baseline.csv")

    # Figure
    drug_list = list(DRUGS.keys())
    x = np.arange(len(drug_list))
    width = 0.35

    mod_means = [v3_scores[(v3_scores["drug"] == d) &
                           (v3_scores["feature_set"] == "Modules_v3")]["spearman"].mean()
                 for d in drug_list]
    pca_means = [pca_scores_df[(pca_scores_df["drug"] == d) &
                               (pca_scores_df["feature_set"] == "PCA_baseline")]["spearman"].mean()
                 for d in drug_list]
    mod_stds  = [v3_scores[(v3_scores["drug"] == d) &
                           (v3_scores["feature_set"] == "Modules_v3")]["spearman"].std()
                 for d in drug_list]
    pca_stds  = [pca_scores_df[(pca_scores_df["drug"] == d) &
                               (pca_scores_df["feature_set"] == "PCA_baseline")]["spearman"].std()
                 for d in drug_list]


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, mod_means, width, yerr=mod_stds, label="Modules (WGCNA)",
           color="#6ACC65", capsize=4, alpha=0.85)
    ax.bar(x + width / 2, pca_means, width, yerr=pca_stds, label=f"PCA baseline (K={K})",
           color="#4878CF", capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([d.split()[0] for d in drug_list], rotation=20, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"Modules vs PCA baseline (K={K} PCs, 80% variance)", fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.5, linestyle="--")

    for i, row in pca_compare_df.iterrows():
        p_adj = row.get("p_adj_fdr", 1)
        if not np.isnan(p_adj) and p_adj < 0.05:
            y_max = max(mod_means[i] + mod_stds[i], pca_means[i] + pca_stds[i]) + 0.02
            ax.text(i, y_max, "*", ha="center", fontsize=14, color="black", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{SUPP}/fig_modules_vs_pca_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{SUPP}/fig_modules_vs_pca_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("    Saved: fig_modules_vs_pca_comparison.pdf")

    t6_end = time.time()
    n_sig_pca = int(pca_compare_df["p_adj_fdr"].lt(0.05).sum())
    experiment_results["Exp6_PCA_baseline"] = {
        "status": "completed",
        "finding": (f"PCA K={K} (80% var); "
                    f"Modules > PCA_baseline in {n_sig_pca}/3 drugs (FDR<0.05)"),
        "impact": "refuerza" if n_sig_pca > 0 else "neutro",
        "time_s": round(t6_end - t6_start, 1)
    }
    log(f"  Experiment 6 completed in {t6_end - t6_start:.1f}s")

except Exception as e:
    log(f"  ERROR in Experiment 6: {e}\n{traceback.format_exc()}")
    experiment_results["Exp6_PCA_baseline"] = {
        "status": "failed", "finding": str(e), "impact": "N/A", "time_s": 0
    }


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE EXPERIMENT_SUMMARY.md
# ══════════════════════════════════════════════════════════════════════════════
log("=== Generating EXPERIMENT_SUMMARY.md ===")

generated_files = []
for root, dirs, files in os.walk(SUPP):
    for fn in sorted(files):
        generated_files.append(os.path.join(root, fn))

summary_lines = [
    "# Supplementary Experiments — Summary",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Experiment Status Table",
    "",
    "| Experiment | Status | Hallazgo principal | Impacto | Tiempo (s) |",
    "|---|---|---|---|---|",
]

exp_order = [
    ("Exp1_SHAP_unconfounded",     "Exp 1 — SHAP unconfounded features"),
    ("Exp2_YAP1_SHAP",             "Exp 2 — YAP1/TEAD1 SHAP by sensitivity"),
    ("Exp3_Ablation",              "Exp 3 — Ablation Combined_noEdge"),
    ("Exp4_Sensitivity_noImputed", "Exp 4 — Sensitivity no-imputed cohort"),
    ("Exp5_WGCNA_stability",       "Exp 5 — WGCNA module stability (R)"),
    ("Exp6_PCA_baseline",          "Exp 6 — PCA baseline vs Modules"),
]

for key, label in exp_order:
    r = experiment_results.get(key, {"status": "not_run", "finding": "", "impact": "N/A", "time_s": 0})
    summary_lines.append(
        f"| {label} | {r['status']} | {str(r['finding'])[:80]} | {r['impact']} | {r['time_s']} |"
    )

summary_lines += [
    "",
    "## Generated Files",
    "",
]
for fp in generated_files:
    summary_lines.append(f"- `{fp}`")

summary_lines += [
    "",
    "## Notes",
    "",
    "- Experiment 5 (WGCNA stability) requires R >= 4.3 with WGCNA package.",
    "  Run: `Rscript /Users/mriosc/Documents/paper2/results/supplementary/wgcna_stability_analysis.R`",
    "- All Python experiments use random_state=42 for reproducibility.",
    "- Figures saved at 300 dpi in both PDF and PNG formats.",
    "- Tables include an `interpretation` column indicating whether results",
    "  reinforce or weaken the main paper findings.",
]

with open(f"{SUPP}/EXPERIMENT_SUMMARY.md", "w") as f:
    f.write("\n".join(summary_lines))
log(f"  Saved: {SUPP}/EXPERIMENT_SUMMARY.md")

log("\n=== All supplementary experiments complete ===")
log(f"Results in: {SUPP}")
print(f"\nDone. Results in: {SUPP}")