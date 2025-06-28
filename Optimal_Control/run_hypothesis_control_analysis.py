import gymnasium as gym
import gym_PBN
import json
import time
import numpy as np
from collections import Counter
import os
import csv

from stable_baselines3 import PPO


# =========================================================================
# SCRIPT: run_sequential_control_hypothesis.py
#
# PURPOSE:
#   To test the Phase 3 hypothesis by simulating a sequential intervention.
#   This version saves all outputs to both TXT and CSV formats.
#
# WORKFLOW:
#   1. Check for and load/train a PPO agent.
#   2. Evaluate the sequential intervention strategy.
#   3. Save a detailed report to 'rl_sequential_hypothesis_report.txt'.
#   4. Save action frequency data to 'rl_sequential_action_frequencies.csv'.
#
# USAGE:
#   Run the script: python3 run_sequential_control_hypothesis.py
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
    """Finds all network states that match a phenotype definition."""
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


def main():
    """Main function to train/load agent and run the hypothesis test."""
    print("====== Starting Self-Contained Sequential Control Hypothesis Test ======\n")

    # --- 1. Define and Create the Environment ---
    print("1. Defining the Non-Responder PBN Environment...")
    NON_RESPONDER_JSON_FILE = 'non_responder_PBN_model_mi_loose.json'
    gene_list, logic_funcs = load_pbn_from_json(NON_RESPONDER_JSON_FILE)
    num_genes = len(gene_list)
    logic_func_data = (gene_list, logic_funcs)

    try:
        mapk3_index = gene_list.index('MAPK3')
    except ValueError:
        print("Error: MAPK3 not found in the gene list.")
        return

    print("   Defining phenotypes based on attractor analysis...")
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

    # --- 2. Train or Load the RL Agent ---
    MODEL_FILE = "ppo_pbn_controller.zip"
    TOTAL_TRAINING_TIMESTEPS = 500000
    model_was_loaded = False

    if os.path.exists(MODEL_FILE):
        print(f"\n2. Found pre-trained model ('{MODEL_FILE}'). Loading agent...")
        model = PPO.load(MODEL_FILE, env=env)
        model_was_loaded = True
        print("   Agent loaded successfully.")
    else:
        print(f"\n2. No pre-trained model found. Starting new training...")
        print("   This is the most time-consuming step and only needs to be done once.")
        start_time = time.time()

        model = PPO("MlpPolicy", env, verbose=0, n_steps=2048)
        model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS)
        model.save(MODEL_FILE)

        end_time = time.time()
        print(f"   Training finished in {end_time - start_time:.2f} seconds.")
        print(f"   Agent saved to '{MODEL_FILE}' for future runs.")
    print("-" * 60)

    # --- 3. Evaluate the Sequential Intervention Strategy ---
    print(f"\n3. Evaluating the sequential intervention strategy...")

    action_map = {0: "Do Nothing"}
    for i, name in enumerate(gene_list):
        action_map[i + 1] = f"Flip Gene '{name}'"

    N_EVAL_PER_ATTRACTOR = 25
    N_EVAL_EPISODES = len(source_states) * N_EVAL_PER_ATTRACTOR

    action_counts = Counter()
    success_count = 0
    eval_start_time = time.time()

    for start_state in source_states:
        for _ in range(N_EVAL_PER_ATTRACTOR):
            # === PHASE 1: TEMPORARY MAPK3 INHIBITION ===
            env.reset(options={"state": np.array(start_state)})
            next_state_array, _, _, _, _ = env.step(0)  # 'Do Nothing' action

            # === THE FIX: Directly convert the returned numpy array to a list ===
            destabilized_state = list(next_state_array)
            destabilized_state[mapk3_index] = 0
            destabilized_state_tuple = tuple(destabilized_state)

            # === PHASE 2: RL AGENT CONTROL ===
            obs, info = env.reset(options={"state": np.array(destabilized_state_tuple)})

            for step in range(15):
                action, _states = model.predict(obs, deterministic=True)
                action_int = action.item()
                action_counts[action_int] += 1
                obs, reward, terminated, truncated, info = env.step(action_int)
                if terminated:
                    success_count += 1
                    break
                if truncated:
                    break

    eval_end_time = time.time()
    print(f"   Evaluation complete in {eval_end_time - eval_start_time:.2f} seconds.")
    print("-" * 60)

    # --- 4. Report Final Results to Console ---
    print("\n4. Sequential Intervention Strategy Results:")
    success_rate = (success_count / N_EVAL_EPISODES) * 100

    RL_ONLY_SUCCESS_RATE = 88.0
    STATIC_KO_SUCCESS_RATE = 98.0

    print(f"\n- Strategy Success Rate: {success_rate:.2f}%")
    print(f"  (Successfully drove the network to a sensitive state in {success_count} out of {N_EVAL_EPISODES} trials)")

    print("\n- Comparison to Benchmarks from Paper:")
    print(f"  - RL-Only (JUN Flip) Success Rate: {RL_ONLY_SUCCESS_RATE:.2f}%")
    print(f"  - Static MAPK3 Knockout Efficacy: {STATIC_KO_SUCCESS_RATE:.2f}%")

    print("\n- Frequency of Actions Chosen by Agent (Post-Inhibition):")
    print("  ------------------------------------------")
    print(f"  {'Action':<25} | {'Times Chosen':<15}")
    print("  ------------------------------------------")
    for action_idx, count in action_counts.most_common():
        action_name = action_map.get(action_idx, "Invalid Action")
        print(f"  {action_name:<25} | {count:<15}")
    print("  ------------------------------------------")

    # --- 5. Generate Conclusion ---
    print("\n- CONCLUSION FROM THIS EXPERIMENT:")
    conclusion_text = ""
    if success_rate > RL_ONLY_SUCCESS_RATE and success_rate >= STATIC_KO_SUCCESS_RATE - 2:
        conclusion_text = "The results STRONGLY SUPPORT the hypothesis. The sequential strategy is highly effective,\nachieving an efficacy comparable to the clinically challenging permanent knockout."
        print("  >> The results STRONGLY SUPPORT the hypothesis.")
        print("  >> The sequential strategy (temp MAPK3 inhibition -> agent-guided perturbation)")
        print(f"     is not only more effective than the agent alone but achieves an")
        print(f"     efficacy comparable to a permanent, static MAPK3 knockout.")
    elif success_rate > RL_ONLY_SUCCESS_RATE:
        conclusion_text = "The results PARTIALLY SUPPORT the hypothesis. The sequential strategy is an improvement\nbut does not fully replicate the effects of a permanent MAPK3 knockout."
        print("  >> The results PARTIALLY SUPPORT the hypothesis.")
        print("  >> The sequential strategy improved outcomes compared to the agent acting alone,")
        print("     but did not reach the efficacy of a permanent knockout.")
    else:
        conclusion_text = "The results DO NOT SUPPORT the hypothesis. The initial MAPK3 inhibition did not\nmeaningfully improve the performance of the RL control agent."
        print("  >> The results DO NOT SUPPORT the hypothesis.")
        print("  >> The initial MAPK3 inhibition did not improve the performance of the RL agent.")
    print("-" * 60)

    # --- 6. SAVE RESULTS TO FILES ---
    TXT_REPORT_FILE = 'rl_sequential_hypothesis_report.txt'
    CSV_REPORT_FILE = 'rl_sequential_action_frequencies.csv'

    # Save the detailed TXT report
    print(f"\nSaving detailed text report to: {TXT_REPORT_FILE}")
    with open(TXT_REPORT_FILE, 'w') as f:
        f.write("====== Sequential Control Hypothesis Test Report ======\n\n")
        f.write("This report details the results of running the sequential control experiment.\n\n")
        f.write("--- MODEL STATUS ---\n")
        if model_was_loaded:
            f.write("Loaded pre-trained model from 'ppo_pbn_controller.zip'.\n\n")
        else:
            f.write(f"Trained a new model for {TOTAL_TRAINING_TIMESTEPS} timesteps.\n\n")
        f.write("--- EXPERIMENTAL SETUP ---\n")
        f.write(f"Evaluation Episodes: {N_EVAL_EPISODES}\n")
        f.write("Intervention Strategy:\n")
        f.write("  1. Evolve 1 step from a resistant attractor.\n")
        f.write("  2. Inhibit MAPK3 (set to 0) for one time step.\n")
        f.write("  3. Allow the trained RL agent to take control.\n\n")
        f.write("--- RESULTS ---\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")
        f.write(f"Benchmark (RL-Only): {RL_ONLY_SUCCESS_RATE:.2f}%\n")
        f.write(f"Benchmark (Permanent MAPK3 KO): {STATIC_KO_SUCCESS_RATE:.2f}%\n\n")
        f.write("--- CONCLUSION ---\n")
        f.write(conclusion_text + "\n")

    # Save the action frequencies to a CSV file
    print(f"Saving action frequency data to: {CSV_REPORT_FILE}")
    with open(CSV_REPORT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Action', 'Times Chosen'])
        for action_idx, count in action_counts.most_common():
            action_name = action_map.get(action_idx, "Invalid Action")
            writer.writerow([action_name, count])

    env.close()
    print("\n====== Analysis Complete ======")


if __name__ == '__main__':
    main()