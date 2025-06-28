import pandas as pd
from datetime import datetime


# =========================================================================
# SCRIPT: run_comparison_analysis.py
#
# PURPOSE:
#   Loads and analyzes PBN attractor data for responder and non-responder
#   models, then generates a clean, formatted comparison report.
#
# REVISION:
#   - Fixed a SyntaxError related to illegal backslashes in f-strings by
#     simplifying the formatting logic.
# =========================================================================


def analyze_landscape(df: pd.DataFrame) -> dict:
    """
    Calculates key metrics for a given attractor landscape DataFrame.

    Args:
        df: A pandas DataFrame with attractor data.

    Returns:
        A dictionary containing calculated statistics.
    """
    if df.empty:
        return {
            "total_attractors": 0,
            "top_1_prob": 0,
            "size_distribution": {},
            "top_5_raw_string": "No attractors found."
        }

    # Ensure data is sorted by probability for correct analysis
    df = df.sort_values(by='Estimated_Probability', ascending=False).reset_index(drop=True)

    # --- Calculate Primary Metrics ---
    stats = {}
    stats['total_attractors'] = len(df)
    stats['top_1_prob'] = df['Estimated_Probability'].iloc[0]

    # --- Calculate Attractor Size Distribution (weighted by frequency) ---
    total_frequency = df['Frequency_Count'].sum()
    size_dist = (df.groupby('Attractor_Size')['Frequency_Count'].sum() / total_frequency) * 100
    stats['size_distribution'] = size_dist.to_dict()

    # --- Format the Top 5 Attractors into a string for the report ---
    top_5_df = df.head(5)
    header = f"{'Attractor_Size':<15} | {'Frequency_Count':<15} | {'Estimated_Probability':<21} | {'AttractorStates_Decimal'}"
    lines = [header, '-' * len(header)]

    for _, row in top_5_df.iterrows():
        lines.append(
            f"{row['Attractor_Size']:<15} | {row['Frequency_Count']:<15} | "
            f"{row['Estimated_Probability']:<21.5f} | \"{row['AttractorStates_Decimal']}\""
        )
    stats['top_5_raw_string'] = "\n".join(lines)

    return stats


def format_size_distribution(dist_map: dict) -> str:
    """
    Formats the size distribution dictionary into a readable string.
    """
    lines = []
    for size, percentage in sorted(dist_map.items()):
        type_desc = "Fixed-Point (Stable State)" if size == 1 else f"Cyclic (of length {size})"
        lines.append(f"  - Size {size} ({type_desc}): {percentage:.2f}%")
    return "\n".join(lines)


def main():
    """
    Main function to coordinate loading, analysis, and report generation.
    """
    # --- 1. Configuration ---
    responder_file = 'attractors_responder.csv'
    non_responder_file = 'attractors_non_responder.csv'
    output_file = 'comparison_report_final.txt'

    # --- 2. Load Data ---
    try:
        responder_df = pd.read_csv(responder_file)
        non_responder_df = pd.read_csv(non_responder_file)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}.")
        print("Please ensure both CSV files are in the same directory as the script.")
        return

    # --- 3. Analyze Datasets ---
    print("Analyzing attractor landscapes...")
    responder_stats = analyze_landscape(responder_df)
    non_responder_stats = analyze_landscape(non_responder_df)

    # --- 4. Build the Report ---
    print("Generating report...")
    report_content = []

    # Header
    report_content.append('=' * 80)
    report_content.append('PBN Attractor Landscape Comparison Report')
    report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append('=' * 80)
    report_content.append('')

    # Summary
    report_content.append('1. HIGH-LEVEL SUMMARY')
    report_content.append('-------------------------')
    report_content.append('The analysis reveals a stark contrast between the two models. The drug-responder\n'
                          'landscape is fragmented and plastic, characterized by a high diversity of\n'
                          'attractors with no single dominant phenotype. In contrast, the non-responder\n'
                          'landscape is highly consolidated and rigid, canalized into an extremely small\n'
                          'number of dominant, stable states. This suggests that drug resistance is\n'
                          'achieved via a network rewiring that sacrifices plasticity for robust stability.')
    report_content.append('')

    # Quantitative Table
    quant_header = f"{'Metric':<45} | {'Responder (Sensitive)':<25} | {'Non-Responder (Resistant)':<25}"
    report_content.append('2. QUANTITATIVE COMPARISON')
    report_content.append('-------------------------')
    report_content.append(quant_header)
    report_content.append('-' * len(quant_header))
    report_content.append(
        f"{'Total Unique Attractors Found':<45} | {responder_stats['total_attractors']:<25} | {non_responder_stats['total_attractors']:<25}")

    # ============================ FIX IS HERE ============================
    # The f-string formatting has been corrected to be valid Python syntax.
    report_content.append(
        f"{'Probability of Most Dominant Attractor':<45} | "
        f"{responder_stats['top_1_prob']:<25.2%} | "
        f"{non_responder_stats['top_1_prob']:<25.2%}"
    )
    # =====================================================================

    report_content.append('')

    # Size Distributions
    report_content.append('3. ATTRACTOR SIZE DISTRIBUTION (PHENOTYPE COMPLEXITY)')
    report_content.append('-------------------------------------------------------')
    report_content.append('Responder Model:')
    report_content.append(format_size_distribution(responder_stats['size_distribution']))
    report_content.append('\nNon-Responder Model:')
    report_content.append(format_size_distribution(non_responder_stats['size_distribution']))
    report_content.append('')

    # Top 5 Attractor Details
    report_content.append('4. TOP 5 ATTRACTORS - RESPONDER MODEL')
    report_content.append('----------------------------------------')
    report_content.append(responder_stats['top_5_raw_string'])
    report_content.append('\n5. TOP 5 ATTRACTORS - NON-RESPONDER MODEL')
    report_content.append('---------------------------------------------')
    report_content.append(non_responder_stats['top_5_raw_string'])
    report_content.append('')

    # --- 5. Write to File ---
    with open(output_file, 'w') as f:
        f.write("\n".join(report_content))

    print(f"Success! Comparison report generated: {output_file}")


if __name__ == "__main__":
    main()