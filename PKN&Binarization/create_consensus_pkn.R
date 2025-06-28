# ===================================================================
# SCRIPT: create_consensus_pkn_v4.2.R
#
# PURPOSE:
# Creates a final PKN and core gene list for the Anti-PD-1 Resistance
# study. It uses a hybrid selection strategy, combining expert-defined
# IPRES genes with top genes from a purely data-driven ranking.
#
# v4.2 Update (Error Handling):
# - Added a robust pre-check to detect unclassified samples *before*
#   they are converted to NA by the factor() function.
# - If problematic sample names are found, the script now stops and
#   prints the exact column names that need to be fixed in the source
#   CSV file, preventing the "cannot contain NA" error.
# ===================================================================

# --- 1. Load Libraries ---
library(DESeq2)
library(doParallel)
library(doRNG)
library(reshape2)

cat("--- Starting Advanced PKN Generation (v4.2 - Robust Error Handling) ---\n")

# --- 2. Define Parameters and Biological Gene Lists ---
cat("\nStep 1: Defining parameters and IPRES signature genes...\n")
ipres_signature_genes <- c("AXL", "ROR2", "WNT5A", "LOXL2", "TAGLN")
cat("Defined", length(ipres_signature_genes), "core IPRES signature genes from Hugo et al. for forced inclusion.\n")
weights <- list(de_score = 0.5, net_score = 0.5)
cat(sprintf("Using weights: DE=%.2f, Network=%.2f. Ranking is purely data-driven.\n",
            weights$de_score, weights$net_score))

# --- 3. Load Data and Identify Common Genes ---
cat("\nStep 2: Loading data and identifying common genes...\n")
tryCatch({
  pkn_data <- read.table("network.sif", sep = " ", header = FALSE, stringsAsFactors = FALSE)
  colnames(pkn_data) <- c("source", "interaction", "target")
  pkn_gene_list <- unique(c(pkn_data$source, pkn_data$target))
}, error = function(e) { stop("Could not load 'network.sif'. Please run 'convertToSIF.m' first.") })
count_data <- read.csv("final_clean_counts.csv", header = TRUE, row.names = 1)
experimental_gene_list <- rownames(count_data)
genes_to_consider <- union(pkn_gene_list, ipres_signature_genes)
common_genes <- intersect(genes_to_consider, experimental_gene_list)
cat(sprintf("Found %d common genes to analyze between the PKN, IPRES list, and expression data.\n", length(common_genes)))

# --- 4. Filter, Normalize, and Run Analyses ---
cat("\nStep 3: Filtering data and running DESeq2 & dynGENIE3...\n")
filtered_counts <- count_data[rownames(count_data) %in% common_genes, ]

# --- MODIFIED: Added a robust pre-check for unclassified samples ---
sample_names <- colnames(filtered_counts)
sample_conditions <- character(length = length(sample_names))
sample_conditions[grepl("_responder$", sample_names)] <- "responder"
sample_conditions[grepl("_non_responder$", sample_names)] <- "non_responder"

# --- ROBUSTNESS CHECK ---
# Check for any samples that were not classified (i.e., are still an empty string).
unclassified_indices <- which(sample_conditions == "")
if (length(unclassified_indices) > 0) {
  problematic_names <- sample_names[unclassified_indices]
  error_message <- paste(
    "Error: The following sample columns in 'final_clean_counts.csv' could not be classified.",
    "They do not end in '_responder' or '_non_responder'. Please correct them:",
    paste(problematic_names, collapse = "\n- "),
    sep = "\n- "
  )
  stop(error_message)
}
# --- END OF CHECK ---

# Now it is safe to create the coldata object
coldata <- data.frame(condition = factor(sample_conditions, levels = c("non_responder", "responder")))
rownames(coldata) <- sample_names

# Proceed with DESeq2 analysis
dds_for_vst <- DESeqDataSetFromMatrix(countData = filtered_counts, colData = coldata, design = ~ 1)
vst_data <- assay(varianceStabilizingTransformation(dds_for_vst, blind = FALSE))
dds_for_deg <- DESeqDataSetFromMatrix(countData = filtered_counts, colData = coldata, design = ~ condition)
dds_for_deg <- DESeq(dds_for_deg)
res <- results(dds_for_deg)
res_df <- as.data.frame(res)
res_df$gene <- rownames(res_df)
res_df$de_score <- -log10(res_df$padj)
res_df$de_score[is.na(res_df$de_score)] <- 0
res_df$de_score[is.infinite(res_df$de_score)] <- max(res_df$de_score[is.finite(res_df$de_score)], na.rm = TRUE) * 1.1

# dynGENIE3 Analysis
source("dynGENIE3.R")
cl <- makeCluster(4)
registerDoParallel(cl)
dynGENIE3_results <- dynGENIE3(list(vst_data), list(1:ncol(vst_data)))
stopCluster(cl)
weight_matrix <- dynGENIE3_results$weight.matrix
rownames(weight_matrix) <- rownames(vst_data)
colnames(weight_matrix) <- rownames(vst_data)
influence_outgoing <- rowSums(weight_matrix)
influence_incoming <- colSums(weight_matrix)
total_influence_scores <- influence_outgoing + influence_incoming
influence_df <- data.frame(gene = names(total_influence_scores), net_score = total_influence_scores)

# --- 5. Combine Scores Using Weighted, Data-Driven System ---
cat("\nStep 4: Combining scores using weighted system...\n")
combined_scores_df <- merge(res_df[, c("gene", "de_score")], influence_df, by = "gene")
normalize <- function(x) { (x - min(x, na.rm=T)) / (max(x, na.rm=T) - min(x, na.rm=T)) }
combined_scores_df$de_norm <- normalize(combined_scores_df$de_score)
combined_scores_df$net_norm <- normalize(combined_scores_df$net_score)
combined_scores_df$final_score <- (weights$de_score * combined_scores_df$de_norm) +
  (weights$net_score * combined_scores_df$net_norm)
ranked_genes <- combined_scores_df[order(combined_scores_df$final_score, decreasing = TRUE), ]
cat("Full data-driven ranking table (top 20):\n")
print(head(ranked_genes, 20))

# --- 6. Finalize Gene List and Create the Core PKN using Hybrid Selection ---
cat("\nStep 5: Creating final gene list and PKN using hybrid selection...\n")
top_data_driven_genes <- head(ranked_genes$gene, 12)
cat("Top 10 data-driven genes from ranking:\n")
print(top_data_driven_genes)
final_gene_list <- union(ipres_signature_genes, top_data_driven_genes)
cat(sprintf("\nFinal hybrid list contains %d unique genes.\n", length(final_gene_list)))
print(final_gene_list)
final_pkn_matrix <- weight_matrix[final_gene_list, final_gene_list]
link_list <- get.link.list(final_pkn_matrix)
final_sif <- data.frame(
  source = link_list$regulatory.gene,
  interaction = "pd",
  target = link_list$target.gene
)
final_sif <- final_sif[link_list$weight > 0, ]
cat(sprintf("Final core PKN contains %d interactions.\n", nrow(final_sif)))

# --- 7. Save Final Outputs ---
cat("\nStep 6: Saving final outputs...\n")
output_pkn_file <- "core_pkn.sif"
output_genelist_file <- "core_gene_list.txt"
write.table(final_sif, file = output_pkn_file, sep = " ", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat(sprintf("\nSuccessfully generated final core PKN file: %s\n", output_pkn_file))
write.table(final_gene_list, file = output_genelist_file, row.names = FALSE, col.names = FALSE, quote = FALSE)
cat(sprintf("Final list of %d genes saved to: %s\n", length(final_gene_list), output_genelist_file))

# --- 8. Export dynGENIE3 Rankings for MATLAB Inference ---
cat("\nStep 7: Exporting dynGENIE3 interaction rankings for MATLAB...\n")
link_list_full <- get.link.list(weight_matrix)
ranked_interactions_df <- data.frame(
  regulatory_gene = link_list_full$regulatory.gene,
  target_gene = link_list_full$target.gene,
  weight = link_list_full$weight
)
ranked_interactions_df <- ranked_interactions_df[order(ranked_interactions_df$weight, decreasing = TRUE), ]
output_ranking_file <- "dynGENIE3_ranked_interactions.csv"
write.csv(ranked_interactions_df, file = output_ranking_file, row.names = FALSE, quote = FALSE)
cat(sprintf("Successfully exported %d ranked interactions to: %s\n", nrow(ranked_interactions_df), output_ranking_file))

cat("\n--- Advanced PKN Generation Complete! ---\n")