#!/usr/bin/env Rscript
# wgcna_stability_analysis.R
# ==========================
# Experiment 5: WGCNA module stability bootstrap for tissues with 15 ≤ N < 30
#
# For each small-N tissue, runs 100 bootstrap iterations (80% subsampling without
# replacement) and computes Module Preservation Score (Zsummary, medianRank)
# using WGCNA::modulePreservation().
#
# Outputs (in <SUPP>/):
#   table_wgcna_stability_by_tissue.csv
#   fig_wgcna_zsummary_heatmap.pdf
#   table_predictive_sensitivity_stable_modules_only.csv
#
# Usage (paths as positional arguments — no more hardcoded paths):
#   Rscript wgcna_stability_analysis.R <PROC> <WGCNA_DIR> <SUPP>
#
# Example:
#   Rscript wgcna_stability_analysis.R \
#     /path/to/data/processed \
#     /path/to/networks_v3/wgcna_v3 \
#     /path/to/results/supplementary/wgcna_stability

cat("=== Running Experiment 5: WGCNA Module Stability Bootstrap ===\n")

# ── Dependencies ──────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(WGCNA)
  library(ggplot2)
  library(pheatmap)
})
options(stringsAsFactors = FALSE)
enableWGCNAThreads(nThreads = 4)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Arguments are optional: run as  Rscript wgcna_stability_analysis.R
# or override any path:           Rscript wgcna_stability_analysis.R <PROC> <WGCNA_DIR> <SUPP>
args <- commandArgs(trailingOnly = TRUE)

PROC      <- if (length(args) >= 1) args[1] else "/Users/mriosc/Documents/paper2/data/processed"
WGCNA_DIR <- if (length(args) >= 2) args[2] else "/Users/mriosc/Documents/paper2/networks_v3/wgcna_v3"
SUPP      <- if (length(args) >= 3) args[3] else "/Users/mriosc/Documents/paper2/results/supplementary/wgcna_stability"

cat("Paths in use:\n")
cat("  PROC:      ", PROC,      "\n")
cat("  WGCNA_DIR: ", WGCNA_DIR, "\n")
cat("  SUPP:      ", SUPP,      "\n\n")

# Validate input directories exist before wasting time
if (!dir.exists(PROC))      stop("PROC directory not found: ", PROC)
if (!dir.exists(WGCNA_DIR)) stop("WGCNA_DIR directory not found: ", WGCNA_DIR)

dir.create(SUPP, recursive = TRUE, showWarnings = FALSE)

LOG_PATH <- file.path(SUPP, "wgcna_stability_log.txt")
log_msg <- function(msg) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- paste0("[", ts, "] ", msg)
  cat(line, "\n")
  cat(line, "\n", file = LOG_PATH, append = TRUE)
}

# ── Load expression data ───────────────────────────────────────────────────────
log_msg("Loading expression and metadata...")
X_expr <- read.csv(file.path(PROC, "X_expr_matched.csv"), row.names = 1, check.names = FALSE)
meta   <- read.csv(file.path(PROC, "cell_line_metadata.csv"), row.names = 1)

# Align metadata
common_lines <- intersect(rownames(X_expr), rownames(meta))
X_expr <- X_expr[common_lines, ]
meta   <- meta[common_lines, ]

tissue_counts <- table(meta$tissue)
log_msg(paste("Total tissues:", length(tissue_counts)))

# Tissues with 15 ≤ N < 30 that have WGCNA networks
small_tissues <- names(tissue_counts[tissue_counts >= 15 & tissue_counts < 30])
log_msg(paste("Small-N tissues (15 ≤ N < 30):", paste(small_tissues, collapse = ", ")))

# ── Bootstrap stability analysis ───────────────────────────────────────────────
N_BOOT   <- 100
SUBSAMP  <- 0.80
SEED     <- 42
set.seed(SEED)

stability_rows <- list()
all_zsummary   <- list()

for (tissue in small_tissues) {
  log_msg(paste("Processing tissue:", tissue))
  
  # Get tissue cell lines
  t_lines <- rownames(meta)[meta$tissue == tissue]
  n_t <- length(t_lines)
  log_msg(paste("  N =", n_t))
  
  # Load WGCNA network for this tissue
  tissue_key <- gsub(" ", "_", tissue)
  tissue_key <- gsub("&", "and", tissue_key)
  eig_file   <- file.path(WGCNA_DIR, paste0(tissue_key, "_eigengenes.csv"))
  col_file   <- file.path(WGCNA_DIR, paste0(tissue_key, "_module_colors.csv"))
  
  if (!file.exists(eig_file) || !file.exists(col_file)) {
    log_msg(paste("  SKIP: WGCNA files not found for", tissue_key))
    next
  }
  
  # Expression matrix for this tissue
  X_t <- as.matrix(X_expr[t_lines, ])
  n_genes <- ncol(X_t)
  log_msg(paste("  Expression matrix:", nrow(X_t), "x", n_genes))
  
  # Load module colors
  mod_colors <- read.csv(col_file, row.names = 1)
  gene_colors <- setNames(mod_colors[, 1], rownames(mod_colors))
  
  # Align genes
  common_genes <- intersect(colnames(X_t), names(gene_colors))
  if (length(common_genes) < 100) {
    log_msg(paste("  SKIP: too few common genes:", length(common_genes)))
    next
  }
  X_t_sub    <- X_t[, common_genes]
  colors_sub <- gene_colors[common_genes]
  
  # Reference network (full tissue)
  log_msg("  Computing reference network (full tissue)...")
  
  # Bootstrap iterations
  zsummary_boot <- list()
  n_sub <- max(floor(n_t * SUBSAMP), 5)
  
  for (b in seq_len(N_BOOT)) {
    boot_idx <- sample(seq_len(n_t), size = n_sub, replace = FALSE)
    X_boot   <- X_t_sub[boot_idx, ]
    
    # FIX: remove genes with zero variance or excessive NAs from the bootstrap
    # subsample before calling modulePreservation(). With small N (16-29),
    # random subsampling can produce constant genes that cause the function to fail.
    gsg <- goodSamplesGenes(X_boot, verbose = 0)
    if (!gsg$allOK) {
      X_boot     <- X_boot[gsg$goodSamples, gsg$goodGenes]
      colors_boot <- colors_sub[gsg$goodGenes]
    } else {
      colors_boot <- colors_sub
    }
    
    # Compute module preservation
    tryCatch({
      mp <- modulePreservation(
        multiData = list(
          ref  = list(data = X_t_sub),
          test = list(data = X_boot[, names(colors_boot)])
        ),
        multiColor = list(
          ref  = colors_sub,
          test = colors_boot       # gene-level vector filtered to match X_boot genes
        ),
        referenceNetworks = 1,
        testNetworks       = 2,
        nPermutations      = 10,
        randomSeed         = SEED + b,
        quickCor           = 1,
        verbose            = 0
      )
      
      # mp$preservation$Z is a list with one element named literally "ref.ref".
      # Inside it, the Z-scores are in $inColumnsAlsoPresentIn.test (not $modulePreservation).
      # mp$preservation$observed has the same structure for medianRank.
      z_df <- mp$preservation$Z[["ref.ref"]][["inColumnsAlsoPresentIn.test"]]
      r_df <- mp$preservation$observed[["ref.ref"]][["inColumnsAlsoPresentIn.test"]]
      
      if (!is.null(z_df) && "Zsummary.pres" %in% colnames(z_df)) {
        for (mod in rownames(z_df)) {
          if (mod %in% c("gold", "grey")) next
          zsummary_boot[[length(zsummary_boot) + 1]] <- data.frame(
            tissue     = tissue,
            module     = mod,
            boot_iter  = b,
            Zsummary   = z_df[mod, "Zsummary.pres"],
            medianRank = r_df[mod, "medianRank.pres"],
            stringsAsFactors = FALSE
          )
        }
      }
    }, error = function(e) {
      log_msg(paste0("  [boot ", b, " ERROR] ", conditionMessage(e)))
    })
    
    if (b %% 20 == 0) log_msg(paste("  Bootstrap iteration", b, "/", N_BOOT))
  }
  
  if (length(zsummary_boot) == 0) {
    log_msg(paste("  WARNING: No preservation scores computed for", tissue))
    next
  }
  
  boot_df <- do.call(rbind, zsummary_boot)
  all_zsummary[[tissue]] <- boot_df
  
  # Summarize per module
  mod_summary <- aggregate(
    cbind(Zsummary, medianRank) ~ module,
    data = boot_df,
    FUN  = mean
  )
  mod_summary$Zsummary_sd   <- aggregate(Zsummary ~ module, data = boot_df, FUN = sd)$Zsummary
  mod_summary$preservation  <- ifelse(mod_summary$Zsummary > 10, "highly_preserved",
                                      ifelse(mod_summary$Zsummary > 2,  "moderately_preserved",
                                             "not_preserved"))
  n_high <- sum(mod_summary$preservation == "highly_preserved")
  n_mod  <- sum(mod_summary$preservation == "moderately_preserved")
  n_not  <- sum(mod_summary$preservation == "not_preserved")
  pct_pres <- round((n_high + n_mod) / nrow(mod_summary) * 100, 1)
  
  stability_rows[[length(stability_rows) + 1]] <- data.frame(
    tissue              = tissue,
    N                   = n_t,
    n_modules           = nrow(mod_summary),
    n_highly_preserved  = n_high,
    n_moderately_preserved = n_mod,
    n_not_preserved     = n_not,
    pct_preserved       = pct_pres,
    stringsAsFactors    = FALSE
  )
  
  log_msg(paste0("  Modules: ", nrow(mod_summary),
                 " | highly=", n_high, " | mod=", n_mod, " | not=", n_not,
                 " | pct_preserved=", pct_pres, "%"))
}

# ── 5c: Summary table ─────────────────────────────────────────────────────────
if (length(stability_rows) > 0) {
  stab_df <- do.call(rbind, stability_rows)
  write.csv(stab_df, file.path(SUPP, "table_wgcna_stability_by_tissue.csv"), row.names = FALSE)
  log_msg("Saved: table_wgcna_stability_by_tissue.csv")
} else {
  log_msg("WARNING: No stability results computed (WGCNA modulePreservation may require more data)")
  # Write placeholder
  stab_df <- data.frame(
    tissue = small_tissues,
    N = as.integer(tissue_counts[small_tissues]),
    n_modules = NA, n_highly_preserved = NA, n_moderately_preserved = NA,
    n_not_preserved = NA, pct_preserved = NA
  )
  write.csv(stab_df, file.path(SUPP, "table_wgcna_stability_by_tissue.csv"), row.names = FALSE)
}

# ── 5d: Heatmap of Zsummary ───────────────────────────────────────────────────
if (length(all_zsummary) > 0) {
  all_boot_df <- do.call(rbind, all_zsummary)
  
  # FIX 2: build the tissue × module matrix manually instead of using base
  # reshape(), which silently produces wrong column names when tissue×module
  # combinations are missing (not all modules appear in every tissue).
  mean_z  <- aggregate(Zsummary ~ tissue + module, data = all_boot_df, FUN = mean)
  tissues <- unique(mean_z$tissue)
  modules <- unique(mean_z$module)
  mat_num <- matrix(
    NA_real_,
    nrow = length(tissues),
    ncol = length(modules),
    dimnames = list(tissues, modules)
  )
  for (i in seq_len(nrow(mean_z))) {
    mat_num[mean_z$tissue[i], mean_z$module[i]] <- mean_z$Zsummary[i]
  }
  
  # FIX 3 & 4: drop all-NA columns before computing column means, then order.
  # colMeans() on an all-NA column returns NaN; order(NaN) returns NA and
  # breaks the column indexing silently.
  valid_cols <- colSums(!is.na(mat_num)) > 0   # keep columns with ≥1 real value
  mat_num    <- mat_num[, valid_cols, drop = FALSE]
  col_means  <- colMeans(mat_num, na.rm = TRUE)
  col_order  <- order(col_means, decreasing = TRUE)
  mat_num    <- mat_num[, col_order, drop = FALSE]
  
  pdf(file.path(SUPP, "fig_wgcna_zsummary_heatmap.pdf"), width = 14, height = 6)
  pheatmap(
    mat_num,
    color           = colorRampPalette(c("#D65F5F", "#FFFFFF", "#6ACC65"))(100),
    breaks          = seq(0, 15, length.out = 101),
    cluster_rows    = FALSE,
    cluster_cols    = FALSE,
    na_col          = "grey90",
    main            = "WGCNA Module Preservation Zsummary\n(100 bootstrap iterations, 80% subsampling)",
    fontsize_row    = 10,
    fontsize_col    = 6,
    angle_col       = 45
  )
  dev.off()
  log_msg("Saved: fig_wgcna_zsummary_heatmap.pdf")
} else {
  log_msg("WARNING: No Zsummary data for heatmap")
}

# ── 5e: Predictive sensitivity — stable modules only ─────────────────────────
# This requires Python CV; write a note table instead
note_df <- data.frame(
  note = paste("Experiment 5e requires re-running Python CV with stable-modules-only",
               "eigengene matrix. Run supplementary_experiments_5e.py after this R script."),
  status = "pending_python"
)
write.csv(note_df, file.path(SUPP, "table_predictive_sensitivity_stable_modules_only.csv"),
          row.names = FALSE)

log_msg("=== Experiment 5 (R) complete ===")
cat("Results saved to:", SUPP, "\n")

