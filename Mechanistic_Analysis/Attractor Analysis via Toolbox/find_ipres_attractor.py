import json
import pandas as pd

# =========================================================================
# SCRIPT: find_ipres_attractor.py
#
# PURPOSE:
#   Systematically searches the attractor landscape of a PBN model to find
#   the specific attractor state that best matches a predefined biological
#   signature (e.g., the IPRES signature). It decodes all attractor
#   states and ranks them by their match score.
# =========================================================================


def load_gene_list_from_model(model_filepath: str) -> list:
    """Loads a PBN model from a JSON file and returns its list of gene names."""
    try:
        with open(model_filepath, 'r') as f:
            model_data = json.load(f)
        gene_names = list(model_data['nodes'].keys())
        return gene_names
    except (FileNotFoundError, KeyError, TypeError) as e:
        print(f"FATAL ERROR loading or parsing '{model_filepath}': {e}")
        return None


def decode_state_to_dict(decimal_state: int, gene_names: list) -> dict:
    """Decodes a single decimal state into a dictionary of gene_name: state."""
    num_genes = len(gene_names)
    binary_state = bin(decimal_state)[2:].zfill(num_genes)
    return dict(zip(gene_names, list(binary_state)))


def main():
    """
    Main function to coordinate the search for the IPRES signature attractor.
    """
    # --- 1. Configuration ---
    NON_RESPONDER_MODEL_FILE = 'non_responder_PBN_model_mi_loose.json'
    NON_RESPONDER_ATTRACTORS_FILE = 'attractors_non_responder.csv'

    # Define the core IPRES signature based on the Hugo et al. paper.
    # We are looking for a state where these genes are ON ('1').
    IPRES_SIGNATURE = {
        'AXL': '1',
        'WNT5A': '1',
        'ROR2': '1'
    }

    print("--- Searching for IPRES Signature Attractor ---")
    print(f"Target Signature: {IPRES_SIGNATURE}\n")

    # --- 2. Load Model Genes and Attractor Data ---
    gene_names = load_gene_list_from_model(NON_RESPONDER_MODEL_FILE)
    if not gene_names:
        return

    try:
        attractors_df = pd.read_csv(NON_RESPONDER_ATTRACTORS_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: Attractor file not found at '{NON_RESPONDER_ATTRACTORS_FILE}'")
        return

    # --- 3. Iterate, Decode, and Score Each Attractor State ---
    found_attractors = []

    for _, row in attractors_df.iterrows():
        # The 'AttractorStates_Decimal' can be a single number or a comma-separated string
        decimal_states_str = str(row['AttractorStates_Decimal'])
        decimal_states = [int(s.strip()) for s in decimal_states_str.split(',')]

        for state in decimal_states:
            # Decode the state into a gene expression dictionary
            decoded_pattern = decode_state_to_dict(state, gene_names)

            # Calculate how well this state matches the IPRES signature
            match_score = 0
            for gene, required_state in IPRES_SIGNATURE.items():
                if decoded_pattern.get(gene) == required_state:
                    match_score += 1

            # Store the results for this specific state
            found_attractors.append({
                'Match_Score': match_score,
                'Probability': row['Estimated_Probability'],
                'Attractor_Size': row['Attractor_Size'],
                'Decimal_State': state,
                'Decoded_Pattern': decoded_pattern
            })

    # --- 4. Rank and Display the Best Matches ---
    if not found_attractors:
        print("No attractors were found or processed.")
        return

    # Sort results: best match score first, then highest probability
    sorted_results = sorted(
        found_attractors,
        key=lambda x: (x['Match_Score'], x['Probability']),
        reverse=True
    )

    print("--- Top Matches Found ---")
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 120)

    # Create a clean DataFrame for display
    display_list = []
    for result in sorted_results:
        # Show the state of only the key IPRES genes for a cleaner report
        ipres_gene_states = {gene: result['Decoded_Pattern'][gene] for gene in IPRES_SIGNATURE}
        display_list.append({
            'Match Score': f"{result['Match_Score']} / {len(IPRES_SIGNATURE)}",
            'Probability': f"{result['Probability']:.2%}",
            'Attractor Size': result['Attractor_Size'],
            'Decimal State': result['Decimal_State'],
            'IPRES Gene States': ipres_gene_states
        })

    display_df = pd.DataFrame(display_list)
    print(display_df)

    # --- 5. Final Conclusion ---
    best_match = sorted_results[0]
    if best_match['Match_Score'] == len(IPRES_SIGNATURE):
        print("\n--- CONCLUSION ---")
        print("SUCCESS: A perfect match for the IPRES signature was found!")
        print(f"The IPRES phenotype corresponds to the attractor state: {best_match['Decimal_State']}")
        print(f"This state belongs to an attractor of size {best_match['Attractor_Size']} with a basin of attraction of {best_match['Probability']:.2%}.")
        print("\nThis state should now be used for the in silico knockout experiments (Step 2).")
    else:
        print("\n--- CONCLUSION ---")
        print("NOTE: No perfect match was found.")
        print(f"The closest match had a score of {best_match['Match_Score']} / {len(IPRES_SIGNATURE)}.")
        print("Please review the table above to see the gene expression patterns of the most probable attractors.")


if __name__ == "__main__":
    main()