import json
from random import random

import numpy as np
import shap
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nilearn.conftest import matplotlib
from stable_baselines3 import PPO
import gymnasium as gym
import gym_PBN




# =============================================================================
# SCRIPT: run_robust_xai_analysis.py
# =============================================================================
def save_heatmap_data(shap_dataframe, action_names_list, shap_csv_path="shap_averaged_data.csv", actions_json_path="shap_action_names.json"):
    """Saves the averaged SHAP DataFrame and action names to files."""
    print(f"\nSaving averaged SHAP values to {shap_csv_path}...")
    shap_dataframe.to_csv(shap_csv_path)

    print(f"Saving most frequent actions to {actions_json_path}...")
    with open(actions_json_path, 'w') as f:
        json.dump(action_names_list, f)
    print("Data saved successfully.")

def load_heatmap_data(shap_csv_path="shap_averaged_data.csv", actions_json_path="shap_action_names.json"):
    """Loads the averaged SHAP DataFrame and action names from files for plotting."""
    print(f"Loading SHAP values from {shap_csv_path}...")
    # index_col=0 ensures the first column (gene names) is used as the DataFrame index
    shap_dataframe = pd.read_csv(shap_csv_path, index_col=0)

    print(f"Loading most frequent actions from {actions_json_path}...")
    with open(actions_json_path, 'r') as f:
        action_names_list = json.load(f)
    print("Data loaded successfully.")
    return shap_dataframe, action_names_list
def load_pbn_from_json(json_file):
    """Loads and parses a PBN model from the project-specific JSON file format."""
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
    return node_names, logic_funcs

# Define paths
PBN_MODEL_PATH = 'non_responder_PBN_model_mi_loose.json'
PPO_AGENT_PATH = 'ppo_pbn_controller_15steps.zip'

# Load assets
print("Loading assets...")
gene_list, logic_funcs = load_pbn_from_json(PBN_MODEL_PATH)
logic_func_data = (gene_list, logic_funcs)
agent = PPO.load(PPO_AGENT_PATH)

# Create environment
print("Creating Gym environment...")
goal_config = {"all_attractors": {}, "target": set()}
env = gym.make(
    'gym-PBN/PBN-v0',
    logic_func_data=logic_func_data,
    goal_config=goal_config,
    reward_config={
        'successful_reward': 100.0,
        'wrong_attractor_cost': 5.0,
        'action_cost': 0
    }
)
print("Environment ready.")

# Define resistant attractor states for analysis
source_states = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Dominant Resistant State
])

# ---------- 2. SHAP EXPLAINER SETUP ----------

def agent_action_logits(states, action_idx):
    """Wrapper function to get the logit for a specific action."""
    logits = []
    for state in states:
        obs = np.array(state).astype(np.float32).reshape(1, -1)
        obs_torch = torch.from_numpy(obs).to(agent.policy.device)
        dist = agent.policy.get_distribution(obs_torch)
        logit = dist.distribution.logits[0, action_idx]
        logits.append(float(logit))
    return np.array(logits)

# Create a background dataset for the explainers
background_states = shap.sample(source_states, 50)

# ---------- PART 1: DECONSTRUCT THE PRIMARY INTERVENTION ----------
print("\n--- PART 1: DECONSTRUCTING THE PRIMARY INTERVENTION ---")

# --- 1a. Vulnerability Signature for 'Flip MAP2K3' ---
print("\n[1a] Explaining primary action: 'Flip MAP2K3'")
map2k3_action_idx = gene_list.index('MAP2K3') + 1
explainer_map2k3 = shap.KernelExplainer(lambda s: agent_action_logits(s, map2k3_action_idx), background_states)
shap_values_map2k3 = explainer_map2k3.shap_values(source_states, nsamples=100)

shap.summary_plot(shap_values_map2k3, source_states, feature_names=gene_list, show=False)
plt.title("SHAP Signature for 'Flip MAP2K3' Decision")
plt.tight_layout()
plt.savefig("shap_summary_flip_map2k3.png", dpi=300)
plt.close()
print("Saved shap_summary_flip_map2k3.png")

# --- 1b. Comparative Analysis: 'Why not Flip JUN?' ---
print("\n[1b] Comparative analysis: Explaining 'Flip JUN'")
jun_action_idx = gene_list.index('JUN') + 1
explainer_jun = shap.KernelExplainer(lambda s: agent_action_logits(s, jun_action_idx), background_states)
shap_values_jun = explainer_jun.shap_values(source_states, nsamples=100)

shap.summary_plot(shap_values_jun, source_states, feature_names=gene_list, show=False)
plt.title("SHAP Signature for 'Flip JUN' Decision (Comparative)")
plt.tight_layout()
plt.savefig("shap_summary_flip_jun_comparative.png", dpi=300)
plt.close()
print("Saved shap_summary_flip_jun_comparative.png")


# ---------- PART 2: VALIDATE THE "HIT-AND-RUN" MECHANISM ----------
print("\n--- PART 2: VALIDATING THE 'HIT-AND-RUN' MECHANISM ---")

def simulate_inhibition(start_state, target_gene, duration):
    obs = np.array(start_state).copy()
    target_idx = gene_list.index(target_gene)
    env.reset(options={"state": obs})
    for _ in range(duration):
        obs, _, _, _, _ = env.step(0)  # Do Nothing action
        obs[target_idx] = 0 # Clamp the gene to OFF
    return obs

# --- 2a. Optimal Priming: 4-step LOXL2 Inhibition ---
print("\n[2a] Analyzing state after optimal 4-step LOXL2 inhibition...")
post_loxl2_states = np.array([simulate_inhibition(s, 'LOXL2', 4) for s in source_states])
shap_values_post_loxl2 = explainer_map2k3.shap_values(post_loxl2_states, nsamples=100)

shap.summary_plot(shap_values_post_loxl2, post_loxl2_states, feature_names=gene_list, plot_type="bar", show=False)
plt.title("SHAP Impact After Optimal Priming (4-step LOXL2)")
plt.tight_layout()
plt.savefig("shap_summary_post_optimal_loxl2.png", dpi=300)
plt.close()
print("Saved shap_summary_post_optimal_loxl2.png")

# --- 2b. Sub-optimal Priming (Negative Control): 2-step JUN Inhibition ---
print("\n[2b] Analyzing state after sub-optimal 2-step JUN inhibition...")
post_jun_states = np.array([simulate_inhibition(s, 'JUN', 2) for s in source_states])
shap_values_post_jun = explainer_map2k3.shap_values(post_jun_states, nsamples=100)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Compute mean(|SHAP|) for each gene for both settings
mean_shap_loxl2 = np.abs(shap_values_post_loxl2).mean(axis=0)
mean_shap_jun = np.abs(shap_values_post_jun).mean(axis=0)

# Build a DataFrame for easier plotting
df = pd.DataFrame({
    'Gene': gene_list,
    'LOXL2 (Optimal, 4-step)': mean_shap_loxl2,
    'JUN (Sub-optimal, 2-step)': mean_shap_jun
}).set_index('Gene')

# Sort by the larger value for visual clarity
df = df.sort_values(by=['LOXL2 (Optimal, 4-step)', 'JUN (Sub-optimal, 2-step)'], ascending=False)

# Plot: grouped horizontal bar plot
fig, ax = plt.subplots(figsize=(10, 7))
bar_width = 0.4
y = np.arange(len(df))

ax.barh(y - bar_width/2, df['LOXL2 (Optimal, 4-step)'], height=bar_width, label='LOXL2 (Optimal, 4-step)', color='#1f77b4')
ax.barh(y + bar_width/2, df['JUN (Sub-optimal, 2-step)'], height=bar_width, label='JUN (Sub-optimal, 2-step)', color='#ff7f0e')

ax.set_yticks(y)
ax.set_yticklabels(df.index)
ax.invert_yaxis()
ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
ax.set_title('SHAP Feature Importance: Optimal vs Sub-Optimal Priming')
ax.legend()
plt.tight_layout()
plt.savefig("shap_bar_comparison_loxl2_vs_jun.png", dpi=300)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

n_genes = len(gene_list)
y_pos = np.arange(n_genes)

fig, ax = plt.subplots(figsize=(11, 7))

# MAP2K3: blue circles
ax.scatter(
    shap_values_map2k3.flatten(),
    np.repeat(y_pos - 0.18, shap_values_map2k3.shape[0]),
    color='royalblue',
    alpha=0.85,
    s=44,
    marker='o',
    label="Flip MAP2K3",
    edgecolor='k',
    linewidth=0.3,
)

# JUN: orange triangles
ax.scatter(
    shap_values_jun.flatten(),
    np.repeat(y_pos + 0.18, shap_values_jun.shape[0]),
    color='orange',
    alpha=0.85,
    s=44,
    marker='^',
    label="Flip JUN",
    edgecolor='k',
    linewidth=0.3,
)

ax.axvline(0, color="grey", lw=1, ls='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(gene_list, fontsize=13)
ax.invert_yaxis()
ax.set_xlabel("SHAP value (impact on model output)", fontsize=15)
ax.set_title("SHAP Signature: Flip MAP2K3 vs Flip JUN", fontsize=16)
ax.legend(fontsize=13, frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig("shap_combined_academic_nocolorbar.png", dpi=350)
plt.show()

# --- Make sure to add these imports at the top of your script ---
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter


# ---------------------------------------------------------------


# =============================================================================
# PART 3: ROBUST TRAJECTORY ANALYSIS (PARALLELIZED)
# =============================================================================

def run_single_simulation(run_num, agent, env, initial_state, explainers, gene_list):
    """
    Runs a single 15-step trajectory and returns its SHAP values and action sequence.
    This function is designed to be called in parallel.
    """
    # Important: Seed the environment reset for reproducibility of this specific run
    obs, info = env.reset(
        options={"state": initial_state},

    )

    trajectory = []
    for step in range(15):
        action_idx, _ = agent.predict(obs, deterministic=True)
        action_idx = int(action_idx)
        trajectory.append({'state': obs, 'action_idx': action_idx})
        obs, _, _, _, _ = env.step(action_idx)

    # Calculate SHAP values for this single trajectory
    single_run_shap_values = []
    for step_data in trajectory:
        state_for_shap = step_data['state'].reshape(1, -1)
        action_idx = step_data['action_idx']
        # Use the pre-made explainer for the action taken
        shap_values_for_step = explainers[action_idx].shap_values(state_for_shap, nsamples=100)
        single_run_shap_values.append(shap_values_for_step[0])

    # Extract the sequence of actions taken in this run
    actions_taken = [step['action_idx'] for step in trajectory]

    return np.array(single_run_shap_values), actions_taken


print("\n--- PART 3: VISUALIZING ROBUST DYNAMIC CONTROL TRAJECTORY (PARALLELIZED) ---")

num_robustness_runs = 10000  # Number of trajectories to average over
all_results = []

print(f"\n[3a] Generating and explaining {num_robustness_runs} trajectories in parallel...")

# Create explainers for all possible actions beforehand
explainers = {}
for i in range(env.action_space.n):
    explainers[i] = shap.KernelExplainer(lambda s: agent_action_logits(s, i), background_states)

# Use joblib.Parallel to run the simulations across all CPU cores (n_jobs=-1)
# tqdm is used to create a progress bar.
all_results = Parallel(n_jobs=-1)(
    delayed(run_single_simulation)(
        run_num, agent, env, source_states[0], explainers, gene_list
    )
    for run_num in tqdm(range(num_robustness_runs), desc="Running simulations")
)

# Unpack the results
all_shap_trajectories, all_action_sequences = zip(*all_results)

# --- 3b. Averaging and Plotting ---
print("\n[3b] Averaging SHAP values and generating robust heatmap...")

# Convert list of 2D arrays into a 3D numpy array and calculate the mean
average_shap_trajectory = np.mean(np.array(all_shap_trajectories), axis=0)

# Create a DataFrame for the heatmap
shap_df = pd.DataFrame(average_shap_trajectory.T, index=gene_list, columns=range(1, 16))

# Get the most frequent action at each step for labeling the x-axis
action_names = []
# Transpose the action sequences so we can analyze each step
actions_by_step = np.array(all_action_sequences).T
for step_idx in range(15):
    most_common_action_idx = Counter(actions_by_step[step_idx]).most_common(1)[0][0]
    if most_common_action_idx == 0:
        action_names.append("-")  # Do Nothing
    else:
        action_names.append(gene_list[most_common_action_idx - 1])

# --- ADD THIS LINE TO SAVE THE DATA ---
save_heatmap_data(shap_df, action_names)
# --------------------------------------
# ENHANCED PLOTTING FOR IEEE SINGLE-COLUMN FIGURE
# =============================================================================
# ... (insert the plotting code block here) ...

# Call the enhanced plotting function from the previous step
# (Make sure the IEEE plotting code from our last conversation is defined)
# Note: You can reuse the plotting code block I provided before.
# I am including it here again for completeness.

# =============================================================================
# ENHANCED PLOTTING FOR IEEE SINGLE-COLUMN FIGURE
# =============================================================================
print("Generating enhanced SHAP trajectory heatmap for IEEE publication...")

# 1. Style and Font Configuration
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300
})

# 2. Create the Plot
fig_width_inches = 3.5
fig_height_inches = 2.8
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

# Generate the heatmap of the AVERAGED data
sns.heatmap(
    shap_df, ax=ax, cmap="RdBu_r", center=0, annot=False, linewidths=0.2,
    cbar_kws={"label": "Mean SHAP Value", "pad": 0.02, "aspect": 40}
)

# 3. Customize Ticks and Labels for Clarity
ax.set_xlabel("Control Step (Most Frequent Intervention)", fontsize=8, labelpad=4)
ax.set_ylabel("Gene", fontsize=8, labelpad=2)
ax.set_title("Robust SHAP Trajectory (Averaged over 100 Runs)", fontsize=9, pad=5)

tick_spacing = 2
xticks_locs = np.arange(0, len(action_names), tick_spacing) + 0.5
xtick_labels = [action_names[i] for i in range(0, len(action_names), tick_spacing)]
ax.set_xticks(xticks_locs)
ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# 4. Final Layout Adjustment and Saving
plt.tight_layout(pad=0.5)
output_filename = "shap_trajectory_heatmap_robust_average.png"
plt.savefig(output_filename, dpi=600)
plt.close()

print(f"Saved robust, averaged heatmap to {output_filename}")

print("\n\nRobust XAI analysis complete.")