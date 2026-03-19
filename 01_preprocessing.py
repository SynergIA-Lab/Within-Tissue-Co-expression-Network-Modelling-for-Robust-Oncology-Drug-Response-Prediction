"""
Paso 1 v3.0: Residualizar expresión sobre tejido.

Crea X_expr_residualized.csv eliminando la señal de tejido de la expresión,
para usarla como baseline "justo" comparable a EdgeDisrupt residualizado.

Verifica también tissue accuracy antes y después para confirmar la corrección.

Outputs:
  data/processed/X_expr_residualized.csv   (660 × 14658)
  figures_v3/01_tissue_accuracy_check.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import os, warnings
warnings.filterwarnings("ignore")

PROC  = "/Users/mriosc/Documents/paper2/data/processed"
FIG   = "/Users/mriosc/Documents/paper2/figures_v3"
LOG   = "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"
os.makedirs(FIG, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log("\n=== PASO 1: Residualizar expresión sobre tejido ===")

X    = pd.read_csv(f"{PROC}/X_expr_matched.csv", index_col=0)
meta = pd.read_csv(f"{PROC}/cell_line_metadata.csv", index_col=0)

log(f"X: {X.shape}, meta: {meta.shape}")

tissue = meta.loc[X.index, "tissue"].fillna("Unknown")
le = LabelEncoder()
tissue_enc = le.fit_transform(tissue)
tissue_dummies = pd.get_dummies(tissue, drop_first=True).values.astype(float)
log(f"Tissue groups: {len(np.unique(tissue_enc))}")

# Tissue accuracy BEFORE residualization (top 1000 genes)
gene_vars = X.values.var(axis=0)
top1k = np.argsort(gene_vars)[::-1][:1000]
X_top = StandardScaler().fit_transform(X.iloc[:, top1k].values)
acc_before = cross_val_score(RidgeClassifier(), X_top, tissue_enc,
                              cv=5, scoring="accuracy").mean()
log(f"Tissue accuracy BEFORE residualization (top 1k): {acc_before:.3f}")

# Residualize: fit Ridge(expression ~ tissue_dummies), take residuals
log("Fitting Ridge regression (expression ~ tissue) ...")
ridge = Ridge(alpha=1.0)
ridge.fit(tissue_dummies, X.values)
X_resid = X.values - ridge.predict(tissue_dummies)
X_resid_df = pd.DataFrame(X_resid, index=X.index, columns=X.columns)

# Tissue accuracy AFTER residualization
X_top_resid = StandardScaler().fit_transform(X_resid_df.iloc[:, top1k].values)
acc_after = cross_val_score(RidgeClassifier(), X_top_resid, tissue_enc,
                             cv=5, scoring="accuracy").mean()
log(f"Tissue accuracy AFTER residualization (top 1k):  {acc_after:.3f}")
log(f"Reduction: {acc_before:.3f} → {acc_after:.3f} (Δ={acc_after-acc_before:+.3f})")

# Save
out = f"{PROC}/X_expr_residualized.csv"
X_resid_df.to_csv(out)
log(f"Saved: {out}  {X_resid_df.shape}")

# Figure: tissue accuracy before/after
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Before\nresidualiz.", "After\nresidualiz."],
       [acc_before, acc_after],
       color=["#D65F5F", "#6ACC65"], alpha=0.85, width=0.5)
ax.axhline(0.15, color="black", linestyle="--", linewidth=1,
           label="Acceptance threshold (0.15)")
ax.set_ylabel("Tissue prediction accuracy (5-fold CV)")
ax.set_title("Tissue signal in expression\nbefore vs after residualization")
ax.set_ylim(0, max(acc_before, 0.2) * 1.2)
ax.legend(fontsize=8)
for i, v in enumerate([acc_before, acc_after]):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG}/01_tissue_accuracy_check.png", dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: figures_v3/01_tissue_accuracy_check.png")
log("=== PASO 1 COMPLETO ===")
