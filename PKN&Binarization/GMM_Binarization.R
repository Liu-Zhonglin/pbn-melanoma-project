# ===================================================================
# SCRIPT: GMM_Binarization_v2.R (Anti-PD-1 Study)
#
# PURPOSE:
# Produces binarized expression matrices for the anti-PD-1 study
# using a Gaussian Mixture Model (GMM) approach.
#
# METHOD:
# 1. Loads the final core gene list and raw count matrix.
# 2. Applies Variance Stabilizing Transformation (VST).
# 3. Correctly separates data into responder and non-responder groups.
# 4. For each gene, fits a two-component GMM to identify 'low' and
#    'high' expression clusters. The binarization threshold is
#    derived from the model.
# 5. Includes robust fallbacks for genes with low variance or where
#    the GMM fails to converge.
#
# v2 Update: Adapted for the GSE78220 anti-PD-1 study.
# - Uses the correct "responder" / "non_responder" labels.
# - Updated all variable names and output filenames for consistency.
# ===================================================================

# --- 1. Load Required Libraries ---
# You may need to run install.packages("mclust") once in your console.
library(DESeq2)
library(mclust) # The library for Gaussian Mixture Models

cat("--- Starting Binarization Workflow (GMM Method for Anti-PD-1 Study) ---\n")

# --- 2. Load and Filter Data ---

cat("\nStep 1: Loading core gene list and raw count data...\n")

tryCatch({
  final_gene_list <- read.table("core_gene_list.txt", stringsAsFactors = FALSE)$V1
}, error = function(e) { stop("Could not load 'core_gene_list.txt'. Please ensure this file is in the working directory.") })

tryCatch({
  count_data <- read.csv("final_clean_counts.csv", header = TRUE, row.names = 1)
}, error = function(e) { stop("Could not load 'final_clean_counts.csv'. Please ensure this file is in the working directory.") })

filtered_counts <- count_data[rownames(count_data) %in% final_gene_list, ]
cat(sprintf("Filtered raw counts to %d core genes.\n", nrow(filtered_counts)))


# --- 3. Normalize and Separate Data ---

cat("\nStep 2: Normalizing and Separating Responder vs. Non-Responder Samples...\n")

# --- MODIFIED: Use the robust classification logic ---
sample_names <- colnames(filtered_counts)
sample_conditions <- character(length = length(sample_names))
sample_conditions[grepl("_responder$", sample_names)] <- "responder"
sample_conditions[grepl("_non_responder$", sample_names)] <- "non_responder"

coldata <- data.frame(condition = factor(sample_conditions))
rownames(coldata) <- sample_names

dds <- DESeqDataSetFromMatrix(countData = filtered_counts, colData = coldata, design = ~ 1)
vst_data <- assay(varianceStabilizingTransformation(dds, blind = FALSE))

# --- MODIFIED: Use correct variable names ---
responder_samples <- sample_names[grepl("_responder$", sample_names)]
non_responder_samples <- sample_names[grepl("_non_responder$", sample_names)]

vst_responder <- vst_data[, responder_samples, drop = FALSE]
vst_non_responder <- vst_data[, non_responder_samples, drop = FALSE]
cat(sprintf("Data successfully separated into %d responder and %d non-responder samples.\n",
            ncol(vst_responder), ncol(vst_non_responder)))


# --- 4. Binarization Function using GMM ---

cat("\nStep 3: Performing GMM-based Binarization on each group...\n")

perform_gmm_binarization <- function(expression_matrix, condition_name) {
  
  cat(sprintf("-> Binarizing for '%s' condition:\n", condition_name))
  
  binarized_matrix <- matrix(0, nrow = nrow(expression_matrix), ncol = ncol(expression_matrix))
  rownames(binarized_matrix) <- rownames(expression_matrix)
  colnames(binarized_matrix) <- colnames(expression_matrix)
  
  for (i in 1:nrow(expression_matrix)) {
    gene_name <- rownames(expression_matrix)[i]
    gene_expression <- as.numeric(expression_matrix[i, ])
    
    # Robustness Check 1: Handle genes with no variance
    if (sd(gene_expression) < 1e-6) {
      binarized_matrix[i, ] <- ifelse(mean(gene_expression) < 7.0, 0, 1)
      cat(sprintf("  - Gene '%s': Classified as Constant (low variance).\n", gene_name))
      next
    }
    
    # Robustness Check 2: Use a tryCatch block for GMM fitting
    gmm_model <- tryCatch({
      Mclust(gene_expression, G = 2, modelNames = "E", verbose = FALSE)
    }, error = function(e) { NULL })
    
    # Apply Binarization Logic
    if (!is.null(gmm_model)) {
      # SUCCESS: GMM fitting worked.
      threshold <- mean(gmm_model$parameters$mean)
      binarized_matrix[i, ] <- ifelse(gene_expression > threshold, 1, 0)
      cat(sprintf("  - Gene '%s': Binarized with GMM. Threshold = %.2f\n", gene_name, threshold))
      
    } else {
      # FAILURE: GMM fitting failed. Fall back to median split.
      threshold <- median(gene_expression)
      binarized_matrix[i, ] <- ifelse(gene_expression > threshold, 1, 0)
      cat(sprintf("  - Gene '%s': GMM failed. Used Fallback (Median Split). Threshold = %.2f\n", gene_name, threshold))
    }
  }
  return(binarized_matrix)
}

# --- MODIFIED: Apply function with new variable names ---
final_binarized_responder <- perform_gmm_binarization(vst_responder, "responder")
final_binarized_non_responder <- perform_gmm_binarization(vst_non_responder, "non_responder")
cat("Binarization complete.\n")


# --- 5. Save the Final Output Files ---
cat("\nStep 4: Saving the two final binarized data files...\n")

# --- MODIFIED: Use new, consistent filenames ---
write.csv(final_binarized_responder, file = "binarized_responder_final.csv", row.names = TRUE)
write.csv(final_binarized_non_responder, file = "binarized_non_responder_final.csv", row.names = TRUE)

cat("Successfully generated: 'binarized_responder_final.csv' and 'binarized_non_responder_final.csv'\n")
cat("\n--- Binarization Complete! ---\n")