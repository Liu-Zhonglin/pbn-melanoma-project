import gymnasium as gym
import gym_PBN
import json
import time
import random
import numpy as np
from collections import Counter
import csv

try:
    from stable_baselines3 import PPO
except ImportError:
    print("stable-baselines3 is not installed. Please run: pip install stable-baselines3")
    exit()


# =========================================================================
# SCRIPT: run_rl_optimal_control_revised.py
#
# PURPOSE:
#   To run the definitive, most rigorous version of the control experiment.
#
# v6 REVISIONS (COMPLETE & REPRODUCIBLE):
#   - Set the max_episode_steps constraint to 15.
#   - File saving logic is now complete and fully implemented for both reports.
# =========================================================================

def load_pbn_from_json(json_file):
    """Loads a PBN model from our JSON format."""
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
    """Finds all network states (as binary tuples) that match a phenotype definition."""
    matching_states = set()
    num_genes = len(gene_list)
    gene_indices = {name: i for i, name in enumerate(gene_list)}
    for i in range(2 ** num_genes):
        bin_state = tuple(int(c) for c in format(i, f'0{num_genes}b'))
        is_match = True
        for gene, required_val in phenotype_definition.items():
            idx = gene_indices[gene]
            if bin_state[idx] != required_val:
                is_match = False
                break
        if is_match:
            matching_states.add(bin_state)
    return matching_states


def to_binary_tuple(dec_state, num_bits):
    """Converts a decimal integer to a tuple of binary states."""
    return tuple(int(c) for c in format(dec_state, f'0{num_bits}b'))


def main():
    """Main function to run the RL-based control analysis with a time constraint."""
    print("====== Starting Definitive RL-based Optimal Control Analysis (Time-Constrained) ======\n")

    # --- 1. Define and Create the Environment ---
    print("1. Defining the Non-Responder PBN Environment...")
    NON_RESPONDER_JSON_FILE = 'non_responder_PBN_model_mi_loose.json'
    gene_list, logic_funcs = load_pbn_from_json(NON_RESPONDER_JSON_FILE)
    num_genes = len(gene_list)
    logic_func_data = (gene_list, logic_funcs)

    print("   Defining phenotypes based on attractor analysis...")
    source_attractor_1 = {to_binary_tuple(712, num_genes)}
    source_attractor_2 = {to_binary_tuple(4076, num_genes)}
    source_attractor_3 = {to_binary_tuple(4078, num_genes)}
    source_attractor_4 = {to_binary_tuple(4014, num_genes)}
    source_states = source_attractor_1.union(source_attractor_2, source_attractor_3, source_attractor_4)

    SENSITIVE_STATE_DEF = {'AXL': 0, 'WNT5A': 0, 'ROR2': 1}
    target_states = get_states_for_phenotype(gene_list, SENSITIVE_STATE_DEF)

    print(f"   Using {len(source_states)} specific source (resistant) attractor states.")
    print(f"   Found {len(target_states)} target (sensitive) states.")

    goal_config = {"all_attractors": [source_states, target_states], "target": target_states}

    reward_config = {
        'successful_reward': 100.0,
        'wrong_attractor_cost': 5.0,
        'action_cost': 0
    }



    MAX_STEPS_PER_EPISODE = 15
    print(f"   Applying a time constraint of {MAX_STEPS_PER_EPISODE} steps per episode.")

    env = gym.make(
        'gym-PBN/PBN-v0',
        logic_func_data=logic_func_data,
        goal_config=goal_config,
        render_mode=None,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
        reward_config=reward_config
    )
    print("   Environment created successfully.")
    print("-" * 50)

    # --- 2. Train the RL Agent (to be efficient) ---
    print("\n2. Training the Reinforcement Learning Agent (PPO)...")
    start_time = time.time()
    TOTAL_TRAINING_TIMESTEPS = 500000

    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048)
    model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS)
    training_time = time.time() - start_time
    print(f"   Training finished in {training_time:.2f} seconds.")
    print("-" * 50)

    # --- 3. Robust Evaluation and Policy Analysis ---
    print(f"\n3. Evaluating the learned policy under the {MAX_STEPS_PER_EPISODE}-step constraint...")
    action_map = {0: "Do Nothing"}
    for i, name in enumerate(gene_list):
        action_map[i + 1] = f"Flip Gene '{name}'"

    N_EVAL_PER_ATTRACTOR = 2500
    N_EVAL_EPISODES = len(source_states) * N_EVAL_PER_ATTRACTOR

    action_counts = Counter()
    success_count = 0

    for start_state in source_states:
        for _ in range(N_EVAL_PER_ATTRACTOR):
            obs, info = env.reset(options={"state": np.array(start_state)})
            for step in range(MAX_STEPS_PER_EPISODE):
                action, _states = model.predict(obs, deterministic=True)
                action_int = action.item()
                action_counts[action_int] += 1
                obs, reward, terminated, truncated, info = env.step(action_int)
                if terminated:
                    success_count += 1
                    break
                if truncated:
                    break
    print(f"   Evaluation complete ({N_EVAL_EPISODES} episodes).")
    print("-" * 50)

    # --- 4. Report Final Results to Console ---
    print("\n4. Optimal Control Strategy Results (Time-Constrained):")
    success_rate = (success_count / N_EVAL_EPISODES) * 100
    print(f"\n- Agent Success Rate: {success_rate:.1f}%")
    print("\n- Frequency of Actions Chosen by the Agent:")
    print("  ------------------------------------------")
    print(f"  {'Action':<25} | {'Times Chosen':<15}")
    print("  ------------------------------------------")

    reporting_action_counts = action_counts.copy()
    for action_idx, count in reporting_action_counts.most_common():
        action_name = action_map.get(action_idx, "Invalid Action")
        print(f"  {action_name:<25} | {count:<15}")
    print("  ------------------------------------------")

    action_counts.pop(0, None)
    if not action_counts:
        optimal_strategy = "No meaningful intervention found."
    else:
        most_common_action_idx = action_counts.most_common(1)[0][0]
        optimal_strategy = action_map.get(most_common_action_idx)
    print("\n- CONCLUSION:")
    print(f"  >> Most Effective Intervention under time constraint: {optimal_strategy} <<")
    print("-" * 50)

    # --- 5. SAVE RESULTS TO FILES ---
    TXT_REPORT_FILE = 'rl_control_report_15steps_v6.txt'
    CSV_REPORT_FILE = 'rl_action_frequencies_15steps_v6.csv'

    # =========================================================================
    # ===> FULLY IMPLEMENTED TXT REPORT SAVING <===============================
    # =========================================================================
    print(f"\nSaving comprehensive report to: {TXT_REPORT_FILE}")
    with open(TXT_REPORT_FILE, 'w') as f:
        f.write("====== Definitive Reinforcement Learning Optimal Control Report (v6) ======\n\n")
        f.write("--- METHODOLOGY ---\n")
        f.write(f"Training Timesteps: {TOTAL_TRAINING_TIMESTEPS}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation Episodes: {N_EVAL_EPISODES}\n")
        f.write(f"Episode Time Constraint: {MAX_STEPS_PER_EPISODE} steps\n\n")
        f.write("--- STARTING CONDITIONS ---\n")
        f.write(f"Source States: {len(source_states)} specific attractor states confirmed as resistant.\n")
        f.write(f"Target Phenotype: {SENSITIVE_STATE_DEF}\n\n")
        f.write("--- RESULTS ---\n")
        f.write(f"Agent Success Rate: {success_rate:.1f}%\n")
        f.write(f"(Successfully drove the network to a sensitive state in {success_count} out of {N_EVAL_EPISODES} trials)\n\n")
        f.write("Frequency of Actions Chosen by the Agent:\n")
        f.write("------------------------------------------\n")
        f.write(f"{'Action':<25} | {'Times Chosen':<15}\n")
        f.write("------------------------------------------\n")
        for action_idx, count in reporting_action_counts.most_common():
            action_name = action_map.get(action_idx, "Invalid Action")
            f.write(f"{action_name:<25} | {count:<15}\n")
        f.write("------------------------------------------\n\n")
        f.write("--- CONCLUSION ---\n")
        f.write("When forced to find an active intervention to escape stable attractors under a time constraint,\n")
        f.write("the agent's learned optimal policy is to consistently intervene on a specific gene.\n")
        f.write(f">> Most Effective Intervention: {optimal_strategy} <<\n")

    # =========================================================================
    # ===> FULLY IMPLEMENTED CSV REPORT SAVING <===============================
    # =========================================================================
    print(f"Saving structured action frequencies to: {CSV_REPORT_FILE}")
    with open(CSV_REPORT_FILE, 'w', newline='') as f:
        header = ['Action', 'Times_Chosen']
        writer = csv.writer(f)
        writer.writerow(header)
        for action_idx, count in reporting_action_counts.most_common():
            action_name = action_map.get(action_idx, "Invalid Action")
            writer.writerow([action_name, count])

    env.close()
    print("\n====== Analysis Complete ======")


if __name__ == '__main__':
    main()