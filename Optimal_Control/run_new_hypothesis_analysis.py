import gymnasium as gym
import gym_PBN
import json
import time
import numpy as np
from collections import Counter
import os
import csv

try:
    from stable_baselines3 import PPO
except ImportError:
    print("stable-baselines3 is not installed. Please run: pip install stable-baselines3")
    exit()


# =========================================================================
# SCRIPT: run_temporal_hypothesis_analysis.py
#
# PURPOSE:
#   To test the hypothesis that a brief, temporary inhibition of JUN can be
#   as effective as inhibiting MAPK3, and to explore the optimal duration
#   of this "priming" intervention.
#
# WORKFLOW:
#   1. Check for and load/train a PPO agent.
#   2. Define the four experimental conditions (Target: MAPK3/JUN, Duration: 1/2).
#   3. Loop through each condition and run a full evaluation.
#   4. For each condition, evaluate the agent's success rate and action frequencies.
#   5. Save a comprehensive summary to 'temporal_hypothesis_summary.txt'.
#   6. Save all results in a structured format to 'temporal_hypothesis_results.csv'.
#
# USAGE:
#   Run the script: python3 run_temporal_hypothesis_analysis.py
# =========================================================================

def load_pbn_from_json(json_file):
    """Loads a PBN model from the specified JSON file format."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    nodes_data = data['nodes']
    node_names = list(nodes_data.keys())
    logic_funcs = []
    for name in node_names:
        node_funcs = []
        for func_info in nodes_data[name]['functions']:
            logic_str = func_info['function'].replace('&', ' and ').replace('|', ' or ').replace('~', 'not ')
            prob = func_info['probability']
            if prob > 0:
                node_funcs.append((logic_str, prob))
        logic_funcs.append(node_funcs)
    return (node_names, logic_funcs)


def get_states_for_phenotype(gene_list, phenotype_definition):
    """Finds all network states (binary tuples) that match a phenotype definition."""
    matching_states = set()
    num_genes = len(gene_list)
    gene_indices = {name: i for i, name in enumerate(gene_list)}
    for i in range(2 ** num_genes):
        bin_state = tuple(int(c) for c in format(i, f'0{num_genes}b'))
        is_match = True
        for gene, required_val in phenotype_definition.items():
            idx = gene_indices.get(gene)
            if idx is not None and bin_state[idx] != required_val:
                is_match = False
                break
        if is_match:
            matching_states.add(bin_state)
    return matching_states


def to_binary_tuple(dec_state, num_bits):
    """Converts a decimal integer to a tuple of binary states."""
    return tuple(int(c) for c in format(dec_state, f'0{num_bits}b'))


def evaluate_strategy(env, model, source_states, gene_list, priming_target_name, priming_duration):
    """
    Evaluates a specific priming strategy.

    Args:
        env: The Gymnasium environment.
        model: The trained PPO agent.
        source_states (set): The set of starting resistant attractor states.
        gene_list (list): The list of gene names.
        priming_target_name (str): The name of the gene to inhibit (e.g., 'MAPK3').
        priming_duration (int): The number of steps to inhibit the gene (e.g., 1 or 2).

    Returns:
        tuple: A tuple containing the success rate (float) and a Counter of action frequencies.
    """
    print(f"\n--- Running Evaluation: Target={priming_target_name}, Duration={priming_duration} step(s) ---")

    try:
        target_index = gene_list.index(priming_target_name)
    except ValueError:
        print(f"ERROR: Priming target '{priming_target_name}' not found in gene list.")
        return 0.0, Counter()

    N_EVAL_PER_ATTRACTOR = 25
    N_EVAL_EPISODES = len(source_states) * N_EVAL_PER_ATTRACTOR

    action_counts = Counter()
    success_count = 0
    eval_start_time = time.time()

    for start_state in source_states:
        for _ in range(N_EVAL_PER_ATTRACTOR):
            # === PHASE 1: TEMPORARY INHIBITION (PRIMING) ===
            obs, _ = env.reset(options={"state": np.array(start_state)})

            # Apply inhibition for the specified duration
            for _ in range(priming_duration):
                current_state = list(obs)
                current_state[target_index] = 0  # Clamp the target gene to 0

                # We need to manually step the environment with a 'Do Nothing' action
                # while ensuring the clamped state is respected.
                # The most direct way is to reset the environment to this clamped state
                # then let it evolve one natural step.
                env.reset(options={"state": np.array(current_state)})
                obs, _, _, _, _ = env.step(0)

            # === PHASE 2: RL AGENT CONTROL ===
            # The agent takes over from the state resulting from the priming phase
            destabilized_obs = obs

            for step in range(15):  # Agent gets 15 steps to succeed
                action, _ = model.predict(destabilized_obs, deterministic=True)
                action_int = action.item()
                action_counts[action_int] += 1
                destabilized_obs, _, terminated, truncated, _ = env.step(action_int)
                if terminated:
                    success_count += 1
                    break
                if truncated:
                    break

    eval_end_time = time.time()
    success_rate = (success_count / N_EVAL_EPISODES) * 100

    print(f"   Evaluation complete in {eval_end_time - eval_start_time:.2f} seconds.")
    print(f"   Success Rate: {success_rate:.2f}%")

    return success_rate, action_counts


def main():
    """Main function to run the temporal hypothesis analysis."""
    print("====== Starting Temporal Hypothesis Analysis ======\n")

    # --- Setup Environment ---
    print("1. Defining the Non-Responder PBN Environment...")
    NON_RESPONDER_JSON_FILE = 'non_responder_PBN_model_mi_loose.json'
    gene_list, logic_funcs = load_pbn_from_json(NON_RESPONDER_JSON_FILE)
    num_genes = len(gene_list)
    logic_func_data = (gene_list, logic_funcs)

    source_attractor_1 = {to_binary_tuple(712, num_genes)}
    source_attractor_2 = {to_binary_tuple(4076, num_genes)}
    source_attractor_3 = {to_binary_tuple(4078, num_genes)}
    source_attractor_4 = {to_binary_tuple(4014, num_genes)}
    source_states = source_attractor_1.union(source_attractor_2, source_attractor_3, source_attractor_4)

    SENSITIVE_STATE_DEF = {'AXL': 0, 'WNT5A': 0, 'ROR2': 1}
    target_states = get_states_for_phenotype(gene_list, SENSITIVE_STATE_DEF)

    goal_config = {"all_attractors": [source_states, target_states], "target": target_states}
    env = gym.make('gym-PBN/PBN-v0', logic_func_data=logic_func_data, goal_config=goal_config, render_mode=None)
    print("   Environment created successfully.")
    print("-" * 60)

    # --- Train or Load Agent ---
    MODEL_FILE = "ppo_pbn_controller.zip"
    if os.path.exists(MODEL_FILE):
        print(f"\n2. Found pre-trained model ('{MODEL_FILE}'). Loading agent...")
        model = PPO.load(MODEL_FILE, env=env)
    else:
        print(f"\n2. No pre-trained model found. Training new agent...")
        start_time = time.time()
        model = PPO("MlpPolicy", env, verbose=0, n_steps=2048)
        model.learn(total_timesteps=500000)
        model.save(MODEL_FILE)
        print(f"   Training finished in {time.time() - start_time:.2f}s. Model saved.")
    print("-" * 60)

    # --- Define and Run Experiments ---
    experiments = [
        {"target": "MAPK3", "duration": 1},
        {"target": "MAPK3", "duration": 2},
        {"target": "JUN", "duration": 1},
        {"target": "JUN", "duration": 2},
    ]

    results = []
    action_map = {0: "Do Nothing"}
    for i, name in enumerate(gene_list):
        action_map[i + 1] = f"Flip Gene '{name}'"

    for exp in experiments:
        success_rate, action_counts = evaluate_strategy(
            env, model, source_states, gene_list, exp["target"], exp["duration"]
        )
        results.append({
            "target": exp["target"],
            "duration": exp["duration"],
            "success_rate": success_rate,
            "action_counts": action_counts
        })

    # --- Save Reports ---
    print("\n--- Saving Final Reports ---")
    TXT_REPORT_FILE = 'temporal_hypothesis_summary.txt'
    CSV_REPORT_FILE = 'temporal_hypothesis_results.csv'

    # Save detailed TXT report
    with open(TXT_REPORT_FILE, 'w') as f:
        f.write("====== Temporal Hypothesis Analysis Report ======\n\n")
        f.write("This report details the results of testing different temporary 'priming' interventions.\n")
        f.write("Each experiment involved inhibiting a target gene for a set duration, then allowing\n")
        f.write("the pre-trained RL agent to take control.\n\n")

        for res in results:
            f.write("--------------------------------------------------\n")
            f.write(f"  EXPERIMENT: Target={res['target']}, Duration={res['duration']} step(s)\n")
            f.write("--------------------------------------------------\n")
            f.write(f"  - Success Rate: {res['success_rate']:.2f}%\n")
            f.write("  - Action Frequencies:\n")
            for action_idx, count in res['action_counts'].most_common():
                action_name = action_map.get(action_idx, "Invalid Action")
                f.write(f"    - {action_name:<25}: {count}\n")
            f.write("\n")

        f.write("\n====== OVERALL CONCLUSION ======\n")
        # Add a summary conclusion here if desired
        f.write("The results can be compared to identify the most effective priming strategy\n")
        f.write("based on success rate and the frequency of the 'Do Nothing' action.\n")

    print(f"Detailed summary saved to: {TXT_REPORT_FILE}")

    # Save structured CSV report
    with open(CSV_REPORT_FILE, 'w', newline='') as f:
        # Create header with all possible actions
        header = ['Priming_Target', 'Inhibition_Duration', 'Success_Rate'] + list(action_map.values())
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for res in results:
            row = {
                'Priming_Target': res['target'],
                'Inhibition_Duration': res['duration'],
                'Success_Rate': f"{res['success_rate']:.2f}"
            }
            # Initialize all action counts to 0 for the row
            for action_name in action_map.values():
                row[action_name] = 0
            # Fill in the counts for actions that were actually taken
            for action_idx, count in res['action_counts'].items():
                action_name = action_map.get(action_idx)
                if action_name:
                    row[action_name] = count
            writer.writerow(row)

    print(f"Structured CSV results saved to: {CSV_REPORT_FILE}")

    env.close()
    print("\n====== Analysis Complete ======")


if __name__ == '__main__':
    main()
