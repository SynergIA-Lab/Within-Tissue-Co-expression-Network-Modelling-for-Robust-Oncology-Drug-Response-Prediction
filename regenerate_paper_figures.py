"""
Regenera las figuras del paper afectadas por los cambios z-score:
  - fig_tissue_qc.png        (tissue accuracy con Topology_zscore/Combined_zscore)
  - fig_heatmap_spearman.png (heatmap con nuevos feature sets)
  - fig_barplot_spearman.png (barplot con nuevos feature sets)
  - supp_wilcoxon_delta.png  (Wilcoxon con zscore sets)
  - supp_v3_vs_zscore.png    (nueva figura comparativa → S7 Fig)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

MOD_V3  = "/Users/mriosc/Documents/paper2/models_v3/cv_results"
MOD_Z   = "/Users/mriosc/Documents/paper2/zscore/models/cv_results"
OUT     = "/Users/mriosc/Documents/paper2/figures"

# ── Load data ─────────────────────────────────────────────────────────────────
scores_z   = pd.read_csv(f"{MOD_Z}/cv_scores_zscore.csv")
wilcox_z   = pd.read_csv(f"{MOD_Z}/wilcoxon_zscore.csv")
tissue_z   = pd.read_csv(f"{MOD_Z}/tissue_accuracy_zscore.csv")
summary_v3 = pd.read_csv(f"{MOD_V3}/cv_summary_v3.csv")

DRUGS = ["Osimertinib", "Crizotinib", "KRAS (G12C) Inhibitor-12"]
FEAT_ORDER = ["Baseline_RF", "Baseline_Resid_RF", "Modules_v3",
              "EdgeDisrupt_v3", "Topology_zscore", "Combined_zscore"]
COLORS = ["#4878CF", "#2196F3", "#6ACC65", "#D65F5F", "#B47CC7", "#F0A500"]
LABELS = ["Baseline\nRF", "Baseline\nResid RF", "Modules",
          "EdgeDisrupt", "Topology\nzscore", "Combined\nzscore"]

# ── 1. fig_tissue_qc.png ──────────────────────────────────────────────────────
print("Generating fig_tissue_qc.png ...")

# Use zscore tissue accuracy (includes all 6 sets)
ta = tissue_z.set_index("feature_set")["tissue_accuracy"]
# Ensure order
ta = ta.reindex(FEAT_ORDER)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(len(FEAT_ORDER)), ta.values, color=COLORS,
              edgecolor="white", linewidth=0.5, alpha=0.88)
ax.axhline(0.15, color="red", linestyle="--", linewidth=1.2,
           label="Acceptance threshold (0.15)")
ax.axhline(0.35, color="orange", linestyle=":", linewidth=1.2,
           label="Confounded threshold (0.35)")
ax.set_xticks(range(len(FEAT_ORDER)))
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylabel("Tissue prediction accuracy\n(Logistic Regression, 5-fold CV)", fontsize=10)
ax.set_title("Tissue-of-origin confounding across all feature sets", fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8, loc="upper right")

# Annotate bars
for i, (v, fs) in enumerate(zip(ta.values, FEAT_ORDER)):
    color = "red" if v > 0.35 else ("darkorange" if v > 0.15 else "darkgreen")
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
            fontsize=8, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_tissue_qc.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved fig_tissue_qc.png")

# ── 2. fig_heatmap_spearman.png ───────────────────────────────────────────────
print("Generating fig_heatmap_spearman.png ...")

pivot = (scores_z.groupby(["drug", "feature_set"])["spearman"]
         .mean().unstack("feature_set"))
pivot = pivot.reindex(columns=[c for c in FEAT_ORDER if c in pivot.columns])
pivot = pivot.reindex(DRUGS)

# Short drug labels
pivot.index = ["Osimertinib", "Crizotinib", "KRAS G12C"]

fig, ax = plt.subplots(figsize=(11, 3.5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=0.7,
            annot_kws={"size": 9})
ax.set_title("Spearman ρ — z-score (mean, 5-fold × 3 rep)", fontsize=10)
ax.set_xlabel("Feature set", fontsize=9)
ax.set_ylabel("Drug", fontsize=9)
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_heatmap_spearman.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved fig_heatmap_spearman.png")

# ── 3. fig_barplot_spearman.png ───────────────────────────────────────────────
print("Generating fig_barplot_spearman.png ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
drug_labels = ["Osimertinib", "Crizotinib", "KRAS G12C"]

for ax, drug, dlabel in zip(axes, DRUGS, drug_labels):
    sub = scores_z[scores_z["drug"] == drug]
    means = sub.groupby("feature_set")["spearman"].mean().reindex(FEAT_ORDER)
    stds  = sub.groupby("feature_set")["spearman"].std().reindex(FEAT_ORDER)

    bars = ax.bar(range(len(FEAT_ORDER)), means.values, yerr=stds.values,
                  color=COLORS, capsize=4, alpha=0.85, edgecolor="white",
                  linewidth=0.5)

    ax.set_xticks(range(len(FEAT_ORDER)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_title(dlabel, fontsize=10)
    ax.set_ylabel("Spearman ρ", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.05, 0.75)

    # Stars for significant + effect (criterion: vs Baseline_Resid_RF)
    for i, fs in enumerate(FEAT_ORDER):
        row = wilcox_z[(wilcox_z["drug"] == drug) &
                       (wilcox_z["network_set"] == fs) &
                       (wilcox_z["baseline"] == "Baseline_Resid_RF")]
        if not row.empty and row.iloc[0]["significant"] and row.iloc[0]["meets_effect"]:
            ax.text(i, means.iloc[i] + stds.iloc[i] + 0.015, "★",
                    ha="center", fontsize=13, color="gold")

plt.suptitle("Spearman ρ z-score — ★ = significantly better than Baseline Resid RF",
             fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_barplot_spearman.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved fig_barplot_spearman.png")

# ── 4. supp_wilcoxon_delta.png ────────────────────────────────────────────────
print("Generating supp_wilcoxon_delta.png ...")

network_sets = ["Modules_v3", "EdgeDisrupt_v3", "Topology_zscore", "Combined_zscore"]
net_labels   = ["Modules\nv3", "EdgeDisrupt\nv3", "Topology\nzscore", "Combined\nzscore"]
net_colors   = ["#6ACC65", "#D65F5F", "#B47CC7", "#F0A500"]
baselines    = ["Baseline_RF", "Baseline_Resid_RF"]
base_labels  = ["vs Baseline RF", "vs Baseline Resid RF"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, baseline, blabel in zip(axes, baselines, base_labels):
    x = np.arange(len(DRUGS))
    width = 0.18
    offsets = np.linspace(-(len(network_sets)-1)/2, (len(network_sets)-1)/2, len(network_sets)) * width

    for j, (ns, nl, nc) in enumerate(zip(network_sets, net_labels, net_colors)):
        deltas = []
        sig_effect = []
        for drug in DRUGS:
            row = wilcox_z[(wilcox_z["baseline"] == baseline) &
                           (wilcox_z["drug"] == drug) &
                           (wilcox_z["network_set"] == ns)]
            if not row.empty:
                deltas.append(row.iloc[0]["delta_spearman"])
                sig_effect.append(row.iloc[0]["significant"] and row.iloc[0]["meets_effect"])
            else:
                deltas.append(0); sig_effect.append(False)

        bars = ax.bar(x + offsets[j], deltas, width, label=nl,
                      color=nc, alpha=0.85, edgecolor="white", linewidth=0.5)
        for i, (b, se) in enumerate(zip(bars, sig_effect)):
            if se:
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_height() + 0.003, "★",
                        ha="center", fontsize=11, color="gold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.05, color="gray", linewidth=0.8, linestyle="--",
               label="Effect threshold (Δρ=0.05)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Osimertinib", "Crizotinib", "KRAS G12C"], fontsize=9)
    ax.set_ylabel("ΔSpearman ρ (network − baseline)", fontsize=9)
    ax.set_title(f"Wilcoxon signed-rank: {blabel}", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(-0.45, 0.20)

plt.suptitle("S3 Fig. Wilcoxon test results — z-score feature sets\n"
             "★ = FDR p<0.05 & Δρ>0.05",
             fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/supp_wilcoxon_delta.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved supp_wilcoxon_delta.png")

# ── 5. supp_v3_vs_zscore.png (nueva → S7 Fig) ────────────────────────────────
print("Generating supp_v3_vs_zscore.png ...")

# Load v3 summary
sum_v3 = summary_v3.set_index(["drug", "feature_set"])
sum_z  = scores_z.groupby(["drug", "feature_set"])["spearman"].agg(["mean","std"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
drug_labels_short = ["Osimertinib", "Crizotinib", "KRAS G12C"]
x = np.arange(len(DRUGS))
width = 0.35

for ax, (fs_v3, fs_z, title) in zip(axes, [
    ("Topology_v3",  "Topology_zscore", "Topology"),
    ("Combined_v3",  "Combined_zscore",  "Combined"),
]):
    rho_v3, std_v3, rho_z, std_z = [], [], [], []
    for drug in DRUGS:
        try:
            rho_v3.append(sum_v3.loc[(drug, fs_v3), "mean"])
            std_v3.append(sum_v3.loc[(drug, fs_v3), "std"])
        except KeyError:
            rho_v3.append(np.nan); std_v3.append(0)
        try:
            rho_z.append(sum_z.loc[(drug, fs_z), "mean"])
            std_z.append(sum_z.loc[(drug, fs_z), "std"])
        except KeyError:
            rho_z.append(np.nan); std_z.append(0)

    b1 = ax.bar(x - width/2, rho_v3, width, yerr=std_v3,
                label="v3.0 (raw degree)", color="steelblue",
                capsize=4, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, rho_z, width, yerr=std_z,
                label="z-score (within tissue)", color="#B47CC7",
                capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Delta annotations
    for i in range(len(DRUGS)):
        if not np.isnan(rho_v3[i]) and not np.isnan(rho_z[i]):
            delta = rho_z[i] - rho_v3[i]
            ypos  = max(rho_v3[i] + std_v3[i], rho_z[i] + std_z[i]) + 0.025
            color = "darkgreen" if delta > 0 else "red"
            ax.text(i, ypos, f"Δ={delta:+.3f}", ha="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(drug_labels_short, fontsize=9)
    ax.set_ylabel("Spearman ρ (mean ± sd, 15 folds)", fontsize=9)
    ax.set_title(f"{title}: v3.0 raw vs within-tissue z-score", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.05, 0.75)

plt.suptitle("S7 Fig. Effect of within-tissue standardisation on Topology and Combined\n"
             "Δ = z-score minus raw",
             fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/supp_v3_vs_zscore.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved supp_v3_vs_zscore.png")

print("\nAll figures regenerated successfully.")