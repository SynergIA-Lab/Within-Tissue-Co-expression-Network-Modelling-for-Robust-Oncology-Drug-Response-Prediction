#!/usr/bin/env Rscript
# Paso 2A v3.0: WGCNA estratificado por tejido — umbral N>=15, kNN imputation
#
# Cambios vs v2.0:
#   - Umbral N: 30 → 15 (más tejidos cubiertos)
#   - minModuleSize: 30 → 15
#   - kNN imputation (k=5) para líneas en tejidos no procesados
#   - Reportar cobertura real vs imputada
#
# Outputs:
#   networks_v3/wgcna_v3/<tissue>_eigengenes.csv
#   networks_v3/wgcna_v3/<tissue>_module_colors.csv
#   networks_v3/wgcna_v3/<tissue>_soft_threshold.png
#   features_v3/module_activity_v3.csv
#   figures_v3/02a_wgcna_coverage.png

suppressPackageStartupMessages({
  library(WGCNA)
  library(dynamicTreeCut)
  library(FNN)
  library(ggplot2)
})

options(stringsAsFactors = FALSE)
allowWGCNAThreads(nThreads = 4)

PROC_DIR <- "/Users/mriosc/Documents/paper2/data/processed"
NET_DIR  <- "/Users/mriosc/Documents/paper2/networks_v3/wgcna_v3"
FEAT_DIR <- "/Users/mriosc/Documents/paper2/features_v3"
FIG_DIR  <- "/Users/mriosc/Documents/paper2/figures_v3"
LOG_FILE <- "/Users/mriosc/Documents/paper2/notebook_log_v3.txt"
dir.create(NET_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(FEAT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR,  recursive = TRUE, showWarnings = FALSE)

log_msg <- function(msg) {
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("\n=== PASO 2A v3.0: WGCNA (N>=15, kNN imputation) ===")

X    <- read.csv(file.path(PROC_DIR, "X_expr_matched.csv"),    row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(PROC_DIR, "cell_line_metadata.csv"), row.names = 1)

common <- intersect(rownames(X), rownames(meta))
X    <- X[common, ]
meta <- meta[common, ]
log_msg(sprintf("Data: %d lines × %d genes", nrow(X), ncol(X)))

# Tissues with N >= 15
tissue_counts <- table(meta$tissue)
MIN_N <- 15
tissues_use <- names(tissue_counts[tissue_counts >= MIN_N])
log_msg(sprintf("Tissues with N>=%d (%d total):", MIN_N, length(tissues_use)))
for (t in tissues_use) log_msg(sprintf("  %s: N=%d", t, tissue_counts[t]))

# ── WGCNA per tissue ──────────────────────────────────────────────────────────
run_wgcna <- function(tissue_name) {
  safe_name <- gsub("[^A-Za-z0-9]", "_", tissue_name)
  N_t <- sum(meta$tissue == tissue_name)
  log_msg(sprintf("\n--- %s (N=%d) ---", tissue_name, N_t))

  idx <- rownames(meta)[meta$tissue == tissue_name]
  Xt  <- X[idx, ]

  # Top 5000 genes by variance within tissue (or fewer if N is small)
  n_genes <- ifelse(N_t >= 30, 5000, 3000)
  gene_vars <- apply(Xt, 2, var)
  top_genes <- names(sort(gene_vars, decreasing = TRUE))[1:min(n_genes, ncol(Xt))]
  Xt <- Xt[, top_genes]

  # goodSamplesGenes check
  gsg <- goodSamplesGenes(Xt, verbose = 0)
  if (!gsg$allOK) {
    Xt <- Xt[gsg$goodSamples, gsg$goodGenes]
    log_msg(sprintf("  Post-QC: %d × %d", nrow(Xt), ncol(Xt)))
  }

  # Soft threshold
  powers <- c(4, 6, 8, 10, 12, 14)
  sft    <- pickSoftThreshold(Xt, powerVector = powers,
                              networkType = "signed hybrid", verbose = 0)
  sft_df <- sft$fitIndices
  good_p <- sft_df$Power[sft_df$SFT.R.sq >= 0.80]
  chosen_power <- if (length(good_p) > 0) min(good_p) else sft_df$Power[which.max(sft_df$SFT.R.sq)]
  best_r2 <- sft_df$SFT.R.sq[sft_df$Power == chosen_power]
  log_msg(sprintf("  Soft threshold: power=%d  R²=%.3f", chosen_power, best_r2))

  # Soft threshold plot
  png(file.path(NET_DIR, sprintf("%s_soft_threshold.png", safe_name)), width=800, height=400)
  par(mfrow=c(1,2))
  plot(sft_df$Power, sft_df$SFT.R.sq, type="b", pch=19,
       xlab="Power", ylab="Scale-free R²", main=tissue_name)
  abline(h=0.80, col="red", lty=2); abline(v=chosen_power, col="blue", lty=2)
  plot(sft_df$Power, sft_df$mean.k., type="b", pch=19,
       xlab="Power", ylab="Mean connectivity", main=tissue_name)
  abline(v=chosen_power, col="blue", lty=2)
  dev.off()

  # minModuleSize: proportional to N
  minModSize <- max(10, round(N_t / 5))
  log_msg(sprintf("  minModuleSize: %d", minModSize))

  net <- blockwiseModules(
    Xt,
    power             = chosen_power,
    networkType       = "signed hybrid",
    TOMType           = "signed",
    minModuleSize     = minModSize,
    mergeCutHeight    = 0.25,
    numericLabels     = FALSE,
    pamRespectsDendro = FALSE,
    saveTOMs          = FALSE,
    verbose           = 0,
    maxBlockSize      = ncol(Xt) + 1
  )

  module_colors <- net$colors
  n_modules <- length(unique(module_colors)) - 1
  log_msg(sprintf("  Modules: %d  (grey/unassigned: %d genes)",
                  n_modules, sum(module_colors == "grey")))

  MEs <- moduleEigengenes(Xt, colors = module_colors)$eigengenes
  MEs <- MEs[, !grepl("grey", colnames(MEs)), drop = FALSE]
  colnames(MEs) <- paste0(safe_name, "_", colnames(MEs))

  write.csv(data.frame(gene=names(module_colors), module=module_colors),
            file.path(NET_DIR, sprintf("%s_module_colors.csv", safe_name)), row.names=FALSE)
  write.csv(MEs, file.path(NET_DIR, sprintf("%s_eigengenes.csv", safe_name)))
  log_msg(sprintf("  Saved: %d modules × %d lines", ncol(MEs), nrow(MEs)))
  return(MEs)
}

all_MEs <- list()
for (tissue in tissues_use) {
  tryCatch({
    MEs <- run_wgcna(tissue)
    all_MEs[[tissue]] <- MEs
  }, error = function(e) {
    log_msg(sprintf("  ERROR in %s: %s", tissue, conditionMessage(e)))
  })
}

# ── Combine eigengenes ────────────────────────────────────────────────────────
log_msg("\n--- Combining eigengenes across tissues ---")

all_lines <- rownames(X)
module_activity <- data.frame(depmap_id = all_lines, stringsAsFactors = FALSE)
for (me_df in all_MEs) {
  df <- as.data.frame(me_df); df$depmap_id <- rownames(df)
  module_activity <- merge(module_activity, df, by = "depmap_id", all.x = TRUE)
}
rownames(module_activity) <- module_activity$depmap_id
module_activity$depmap_id <- NULL

n_mods  <- ncol(module_activity)
nan_pct <- mean(is.na(module_activity)) * 100
lines_with_real <- sum(apply(module_activity, 1, function(r) any(!is.na(r))))
coverage_pct <- lines_with_real / nrow(module_activity) * 100

log_msg(sprintf("Combined: %d lines × %d modules", nrow(module_activity), n_mods))
log_msg(sprintf("NaN: %.1f%% (by design — tissue-specific)", nan_pct))
log_msg(sprintf("Lines with real eigengenes: %d / %d (%.1f%%)",
                lines_with_real, nrow(module_activity), coverage_pct))

# ── kNN imputation for lines without modules ──────────────────────────────────
log_msg("\n--- kNN imputation (k=5) for lines without modules ---")

# PCA on full expression for kNN distance
pca_res <- prcomp(X, rank. = 50, scale. = FALSE)
pca_coords <- pca_res$x  # all lines in PCA space

lines_with    <- rownames(module_activity)[apply(module_activity, 1, function(r) any(!is.na(r)))]
lines_without <- rownames(module_activity)[apply(module_activity, 1, function(r) all(is.na(r)))]
log_msg(sprintf("Lines needing imputation: %d", length(lines_without)))

if (length(lines_without) > 0 && length(lines_with) >= 5) {
  coords_with    <- pca_coords[lines_with, , drop = FALSE]
  coords_without <- pca_coords[lines_without, , drop = FALSE]

  k <- min(5, length(lines_with))
  nn_result <- get.knnx(coords_with, coords_without, k = k)

  for (i in seq_along(lines_without)) {
    neighbor_lines <- lines_with[nn_result$nn.index[i, ]]
    neighbor_MEs   <- module_activity[neighbor_lines, , drop = FALSE]
    # Impute with weighted mean (weight = 1/distance)
    dists <- nn_result$nn.dist[i, ]
    weights <- 1 / (dists + 1e-6)
    weights <- weights / sum(weights)
    imputed <- colSums(sweep(neighbor_MEs, 1, weights, "*"), na.rm = TRUE)
    module_activity[lines_without[i], ] <- imputed
  }
  log_msg(sprintf("Imputed %d lines using kNN (k=%d)", length(lines_without), k))
}

# Final coverage
nan_pct_final <- mean(is.na(module_activity)) * 100
log_msg(sprintf("NaN after imputation: %.1f%%", nan_pct_final))

write.csv(module_activity, file.path(FEAT_DIR, "module_activity_v3.csv"))
log_msg(sprintf("Saved: features_v3/module_activity_v3.csv  %d × %d",
                nrow(module_activity), ncol(module_activity)))

# ── Coverage figure ───────────────────────────────────────────────────────────
tissue_coverage <- data.frame(
  tissue    = c(tissues_use, "kNN imputed", "No coverage"),
  n_lines   = c(as.integer(tissue_counts[tissues_use]),
                length(lines_without),
                nrow(module_activity) - lines_with_real - length(lines_without)),
  type      = c(rep("WGCNA", length(tissues_use)), "kNN", "None")
)
tissue_coverage <- tissue_coverage[tissue_coverage$n_lines > 0, ]

png(file.path(FIG_DIR, "02a_wgcna_coverage.png"), width=900, height=500)
par(mar=c(10,4,3,1))
cols <- ifelse(tissue_coverage$type == "WGCNA", "#4878CF",
        ifelse(tissue_coverage$type == "kNN",   "#F0A500", "#D65F5F"))
barplot(tissue_coverage$n_lines, names.arg=tissue_coverage$tissue,
        col=cols, las=2, cex.names=0.7,
        main=sprintf("WGCNA v3.0 coverage — %d modules, %.1f%% real eigengenes",
                     n_mods, coverage_pct),
        ylab="N cell lines")
legend("topright", legend=c("WGCNA real","kNN imputed"),
       fill=c("#4878CF","#F0A500"), bty="n")
dev.off()

log_msg("\n=== PASO 2A v3.0 COMPLETO ===")
log_msg(sprintf("Tissues processed: %d", length(all_MEs)))
log_msg(sprintf("Total modules: %d", n_mods))
log_msg(sprintf("Coverage: %.1f%% real + %.1f%% kNN imputed",
                coverage_pct, length(lines_without)/nrow(module_activity)*100))

