"""
Paso 4 v3.0: SHAP sobre mejor modelo de red + tissue-stratified.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap, warnings, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from scipy import stats
warnings.filterwarnings("ignore")

PROC=f"/Users/mriosc/Documents/paper2/data/processed"; FEAT=f"/Users/mriosc/Documents/paper2/features_v3"
MOD=f"/Users/mriosc/Documents/paper2/models_v3"; FIG=f"/Users/mriosc/Documents/paper2/figures_v3"
LOG=f"/Users/mriosc/Documents/paper2/notebook_log_v3.txt"
os.makedirs(f"{MOD}/shap", exist_ok=True)

def log(m):
    print(m, flush=True)
    open(LOG,"a").write(m+"\n")

def spearman_r(a,b):
    r,_=stats.spearmanr(a,b); return r if not np.isnan(r) else 0.0

def stratify_q(y): return pd.qcut(y,q=4,labels=False,duplicates="drop")

log("\n=== PASO 4 v3.0: SHAP + tissue-stratified ===")

X_expr  = pd.read_csv(f"{PROC}/X_expr_matched.csv", index_col=0)
y_all   = pd.read_csv(f"{PROC}/y_matched.csv",      index_col=0)
meta    = pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)
topo_v3 = pd.read_csv(f"{FEAT}/topo_features_v3.csv",  index_col=0)
ma_v3   = pd.read_csv(f"{FEAT}/module_activity_v3.csv", index_col=0)
ed_v3   = pd.read_csv(f"{FEAT}/edge_disruption_within_tissue.csv", index_col=0)

gene_vars = X_expr.values.var(axis=0)
top1k_idx = np.argsort(gene_vars)[::-1][:1000]
X_base    = X_expr.iloc[:, top1k_idx]
top1k_genes = X_base.columns.tolist()

DRUGS = {"Osimertinib":"AUC","Crizotinib":"LN_IC50","KRAS (G12C) Inhibitor-12":"LN_IC50"}

# ── 4A: SHAP on Topology_v3 (best network model for Osimertinib) ──────────────
log("\n--- 4A: SHAP on Topology_v3 (Osimertinib) and Combined_v3 (Crizotinib) ---")

shap_targets = {
    "Osimertinib":  ("Topology_v3",  topo_v3,  "EN"),
    "Crizotinib":   ("Combined_v3",  pd.concat([ma_v3.fillna(0), ed_v3.fillna(0),
                                                 topo_v3.fillna(topo_v3.median())], axis=1), "RF"),
}

for drug, (feat_name, X_feat, mtype) in shap_targets.items():
    log(f"\n  {drug} — {feat_name}:")
    y_drug = y_all[drug].dropna()
    common = X_feat.index.intersection(y_drug.index)
    X_sub  = X_feat.loc[common].values.astype(np.float32)
    y_vec  = y_drug.loc[common].values

    imp = SimpleImputer(strategy="median").fit(X_sub)
    X_imp = imp.transform(X_sub)
    sc  = StandardScaler().fit(X_imp)
    X_sc = sc.transform(X_imp)

    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                n_jobs=4, random_state=42)
    rf.fit(X_sc, y_vec)

    bg_idx = np.random.RandomState(42).choice(len(X_sc), size=min(150,len(X_sc)), replace=False)
    ev_idx = np.random.RandomState(43).choice(len(X_sc), size=min(200,len(X_sc)), replace=False)
    explainer = shap.TreeExplainer(rf, data=X_sc[bg_idx], feature_perturbation="interventional")
    sv = explainer.shap_values(X_sc[ev_idx])
    log(f"    SHAP values: {sv.shape}")

    mean_abs = np.abs(sv).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[::-1][:20]
    feat_names = X_feat.columns.tolist()
    top20_names = [feat_names[i] for i in top20_idx]
    top20_vals  = mean_abs[top20_idx]
    log(f"    Top 5: {top20_names[:5]}")

    # Check YAP1/PTK2 in topology features
    for gene in ["YAP1","PTK2","EGFR","CDH1"]:
        hits = [(i,n) for i,n in enumerate(feat_names) if gene in n]
        for idx_g, name_g in hits:
            rank = np.where(np.argsort(mean_abs)[::-1]==idx_g)[0]
            if len(rank):
                log(f"    {name_g}: SHAP rank {rank[0]+1}, |SHAP|={mean_abs[idx_g]:.5f}")

    safe = drug.replace(" ","_").replace("(","").replace(")","").replace("/","_")
    np.save(f"{MOD}/shap/shap_{safe}_{feat_name}.npy", sv)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(range(20), top20_vals[::-1], color="#B47CC7")
    ax.set_yticks(range(20))
    labels = [n.replace("degree_","deg_").replace("hub_score_","hub_") for n in top20_names[::-1]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top 20 features — {drug}\n({feat_name})", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG}/04_shap_{safe}_{feat_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"    Figure saved.")

# ── 4B: Tissue-stratified for all tissues with N>=20 ─────────────────────────
log("\n--- 4B: Tissue-stratified modeling (all tissues N>=20) ---")

tissue_results = []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

for drug in DRUGS:
    y_drug = y_all[drug].dropna()
    common = X_base.index.intersection(y_drug.index).intersection(meta.index)
    meta_sub = meta.loc[common]
    tc = meta_sub["tissue"].value_counts()
    tissues_eval = tc[tc >= 20].index.tolist()
    log(f"\n  {drug}: evaluating {len(tissues_eval)} tissues")

    for tissue in tissues_eval:
        t_idx = common[meta_sub["tissue"] == tissue]
        if len(t_idx) < 20: continue
        X_t = X_base.loc[t_idx].values.astype(np.float32)
        y_t = y_drug.loc[t_idx].values
        strat_t = stratify_q(y_t)
        fold_sp = []
        for tr, te in cv.split(X_t, strat_t):
            pipe = Pipeline([("sc",StandardScaler()),
                             ("rf",RandomForestRegressor(n_estimators=150,
                              max_features="sqrt",n_jobs=4,random_state=42))])
            pipe.fit(X_t[tr], y_t[tr])
            fold_sp.append(spearman_r(y_t[te], pipe.predict(X_t[te])))
        sp_t = np.mean(fold_sp); sp_std = np.std(fold_sp)
        # Pan-cancer reference
        scores_df = pd.read_csv(f"{MOD}/cv_results/cv_scores_v3.csv")
        pan_sp = scores_df[(scores_df["drug"]==drug) &
                           (scores_df["feature_set"]=="Baseline_RF")]["spearman"].mean()
        log(f"    {tissue} (N={len(t_idx)}): ρ={sp_t:.3f}±{sp_std:.3f} (pan={pan_sp:.3f}, Δ={sp_t-pan_sp:+.3f})")
        tissue_results.append({"drug":drug,"tissue":tissue,"n":len(t_idx),
                                "spearman_tissue":round(sp_t,3),
                                "spearman_std":round(sp_std,3),
                                "spearman_pancancer":round(pan_sp,3),
                                "delta":round(sp_t-pan_sp,3)})

tissue_df = pd.DataFrame(tissue_results)
tissue_df.to_csv(f"{MOD}/cv_results/tissue_stratified_v3.csv", index=False)
log(f"\nSaved: models_v3/cv_results/tissue_stratified_v3.csv")

# Figure: tissue-stratified heatmap
if len(tissue_df):
    pivot_t = tissue_df.pivot_table(index="tissue", columns="drug",
                                     values="delta", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot_t)*0.4)))
    import seaborn as sns
    sns.heatmap(pivot_t, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Δ Spearman ρ: tissue-stratified vs pan-cancer\n(Baseline RF)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG}/04_tissue_stratified_v3.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Figure saved: 04_tissue_stratified_v3.png")

log("\n=== PASO 4 v3.0 COMPLETO ===")
