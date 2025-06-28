import json
import pandas as pd


# =========================================================================
# SCRIPT: run_knockout_experiments.py
#
# PURPOSE:
#   Performs in silico knockout experiments on specific, predefined
#   attractor states (phenotypes). It simulates the effect of forcing a
#   gene to be 'OFF' and determines if this intervention can break the
#   stability of the resistant phenotype, potentially shifting it to a
#   more responder-like state.
# =========================================================================


def load_pbn_model(model_filepath: str) -> (list, dict):
    """Loads a PBN model, returning the gene list and the parsed model."""
    try:
        with open(model_filepath, 'r') as f:
            model_data = json.load(f)
        gene_names = list(model_data['nodes'].keys())
        return gene_names, model_data
    except (FileNotFoundError, KeyError, TypeError) as e:
        print(f"FATAL ERROR loading or parsing '{model_filepath}': {e}")
        return None, None


def decode_state_to_dict(decimal_state: int, gene_names: list) -> dict:
    """Decodes a decimal state into a dictionary of {gene_name: state}."""
    num_genes = len(gene_names)
    binary_state = bin(decimal_state)[2:].zfill(num_genes)
    return dict(zip(gene_names, list(binary_state)))


def evaluate_boolean_function(func_str: str, current_states: dict) -> bool:
    """
    Evaluates a single Boolean function string (e.g., "AXL & ~JUN")
    against the current state of the network.
    """
    # Replace gene names with their current values (0 or 1)
    for gene, state in current_states.items():
        # Use word boundaries to avoid replacing 'AXL' in 'LOXL2'
        func_str = func_str.replace(f' {gene} ', f' {state} ')
        func_str = func_str.replace(f' {gene}|', f' {state}|')
        if func_str.startswith(f'{gene} '):
            func_str = f'{state} ' + func_str[len(gene):]

    # Replace logic operators with Python equivalents
    func_str = func_str.replace(' & ', ' and ')
    func_str = func_str.replace(' | ', ' or ')
    func_str = func_str.replace('~', ' not ')

    try:
        return eval(func_str)
    except Exception:
        # Handles cases like "0" or "1" which are not valid eval expressions
        if func_str.strip() == "0": return False
        if func_str.strip() == "1": return True
        return False


def run_simulation_step(current_states: dict, model: dict, knockout_gene: str = None) -> dict:
    """
    Calculates the next state of the network based on the current state.
    """
    next_states = {}
    for gene, node_data in model['nodes'].items():
        # For now, we simplify by using only the first Boolean function for each gene.
        # A full PBN simulation would involve probabilities.
        func = node_data['functions'][0]['function']
        next_states[gene] = '1' if evaluate_boolean_function(func, current_states) else '0'

    # Apply the knockout condition *after* calculating the next state
    if knockout_gene:
        next_states[knockout_gene] = '0'

    return next_states


def main():
    """Main function to run and report on knockout experiments."""
    # --- 1. Configuration ---
    MODEL_FILE = 'non_responder_PBN_model_mi_loose.json'
    RESPONDER_DOMINANT_STATE = 13  # The most stable "good" phenotype

    # === ACTION REQUIRED ===
    # Update these decimal values based on your 'find_ipres_attractor.py' output.
    # Use the 'Decimal State' for the highest probability attractors of each type.
    TARGET_PHENOTYPES = {
        "WNT_Driven_Resistance": 4078,  # Placeholder: Replace with your actual state for (WNT5A=1, ROR2=1, AXL=0)
        "AXL_Driven_Resistance": 832,  # Placeholder: Replace with your actual state for (AXL=1, ROR2=1, WNT5A=0)
    }

    # Genes we want to test as potential "master switches"
    GENES_TO_KNOCKOUT = ['JUN', 'PIK3CB', 'RELA', 'MAPK3']

    SIMULATION_STEPS = 20  # Number of steps to run to see if the network stabilizes

    # --- 2. Load Model ---
    gene_names, model = load_pbn_model(MODEL_FILE)
    if not model: return

    responder_pattern = decode_state_to_dict(RESPONDER_DOMINANT_STATE, gene_names)

    print("--- Running In Silico Knockout Experiments ---")

    # --- 3. Run Experiment for Each Phenotype and Knockout ---
    results = []
    for phenotype_name, decimal_state in TARGET_PHENOTYPES.items():
        print(f"\n--- Testing Phenotype: {phenotype_name} (State: {decimal_state}) ---")

        initial_state = decode_state_to_dict(decimal_state, gene_names)

        for ko_gene in GENES_TO_KNOCKOUT:
            # Run the simulation with the knockout
            current_state = initial_state.copy()
            current_state[ko_gene] = '0'  # Apply initial knockout

            for _ in range(SIMULATION_STEPS):
                current_state = run_simulation_step(current_state, model, ko_gene)

            final_state = current_state

            # --- 4. Analyze the Outcome ---
            is_broken = (final_state != initial_state)
            resembles_responder = (final_state == responder_pattern)

            outcome = "FAILED"
            if is_broken and resembles_responder:
                outcome = "SUCCESS (Shifted to Responder Phenotype)"
            elif is_broken:
                outcome = "PARTIAL (Shifted to a new, non-responder state)"

            results.append({
                "Resistant Phenotype": phenotype_name,
                "Knockout Target": ko_gene,
                "Outcome": outcome,
                "Initial State": decimal_state,
                "Final State (Decimal)": int("".join(final_state.values()), 2)
            })

    # --- 5. Display Final Report ---
    report_df = pd.DataFrame(results)
    print("\n\n--- FINAL REPORT: Knockout Experiment Summary ---")
    print(report_df)

    print("\n--- INTERPRETATION ---")
    print("A 'SUCCESS' outcome means the simulated knockout of that gene is a high-priority")
    print("candidate for a combination therapy to overcome that specific type of resistance.")


if __name__ == "__main__":
    main()