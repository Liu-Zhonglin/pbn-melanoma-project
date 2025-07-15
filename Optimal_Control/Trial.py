import gymnasium as gym
import gym_PBN
import numpy as np
from stable_baselines3 import PPO
import time
import random

# --- Step 1: Define Our Custom 3-Node PBN ---

print("1. Defining our custom 3-Node Probabilistic Boolean Control Network (PBCN)...")

# Node names. 'A' is the control node. 'B' and 'C' are the observable state nodes.
node_names = ["A", "B", "C"]

# Logic functions. 'A' has an empty list to designate it as the control node.
logic_funcs = [
    [],                        # Node A (Control Node)
    [("A and not C", 1.0)],    # Node B depends on A and C
    [("B or C", 1.0)],         # Node C depends on B and C
]
logic_func_data = (node_names, logic_funcs)


# --- Goal Configuration for our 3-Node PBN ---
# The environment's observable state consists of the non-control nodes: (B, C).
# We define the attractors based on the states of B and C.
#
# Full "Disease" State (A,B,C) = (0,0,0) -> Observable part (B,C) = (0,0)
# Full "Healthy" State (A,B,C) = (1,0,1) -> Observable part (B,C) = (0,1)

source_attractor = {(0, 0)}
target_attractor = {(0, 1)}

goal_config = {
    "all_attractors": [source_attractor, target_attractor],
    "target": target_attractor,
}

# Reward configuration remains the same.
reward_config = {
    'successful_reward': 100.0,
    'wrong_attractor_cost': -5.0,
    'action_cost': -0.1
}

# Create the Gymnasium environment.
env = gym.make('gym-PBN/PBCN-v0',
               logic_func_data=logic_func_data,
               goal_config=goal_config,
               reward_config=reward_config,
               render_mode=None)

# Set the action space to the correct discrete space.
env.action_space = env.unwrapped.discrete_action_space

print("   Custom 3-Node PBN environment created successfully.")
print(f"   Number of observable nodes: {env.unwrapped.PBN.N}")
print(f"   Action space: {env.action_space}")
print("-" * 30)


# --- Step 2: Train the Reinforcement Learning Agent ---

print("\n2. Training the Reinforcement Learning Agent (PPO)...")
start_time = time.time()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=20000) # 20k is plenty for this small network

end_time = time.time()
print(f"   Training finished in {end_time - start_time:.2f} seconds.")
print("-" * 30)


# --- Step 3: Evaluate the Learned Optimal Policy ---

print("\n3. Evaluating the learned optimal control strategy...")

# Reset the environment.
obs, info = env.reset(seed=123)

# Define our action map and observable node names for clarity.
action_map = {0: "Do Nothing", 1: "Flip Node A"}
observable_nodes = ["B", "C"]

# Run the simulation
for i in range(10):
    print(f"Step {i+1}:")
    print(f"  - Current State ({', '.join(observable_nodes)}): {obs}")

    # If we start in the target, reset until we are in the source state.
    if tuple(obs) in target_attractor:
        print("\nSUCCESS: Reached the target state!")
        break

    # Get the best action from the trained model
    action, _states = model.predict(obs, deterministic=True)
    action_int = action.item()
    print(f"  - Agent's Action: {action_map.get(action_int, f'Action {action_int}')}")

    # Apply the action to the environment
    obs, reward, terminated, truncated, info = env.step(action_int)

    print(f"  - Resulting State: {obs}\n")

    if terminated:
        print("SUCCESS: Episode terminated after reaching target.")
        break
    if truncated:
        print("EPISODE TRUNCATED: Time limit reached.")
        break

if not terminated and not truncated:
    print("EVALUATION FAILED: The agent did not reach the target within 10 steps.")

# Close the environment
env.close()
print("-" * 30)