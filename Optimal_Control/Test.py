import gymnasium as gym
import gym_PBN
import numpy as np
from stable_baselines3 import PPO
import time
import random

# --- Step 1: Define the PBN using the exact data from example.py ---

print("1. Defining the PBCN using the network from example.py...")

# The network definition is correct. 'u' is the control node.
logic_func_data=(
    ["u", "x1", "x2", "x3", "x4"],
    [
        [], # Control node 'u'
        [("not x2 and not x4", 1)],
        [("not x4 and not u and (x2 or x3)", 1)],
        [("not x2 and not x4 and x1", 0.7), ("False", 0.3)],
        [("not x2 and not x3", 1)],
    ],
)

# --- CONFIGURATION CORRECTION ---
# The goal_config MUST use 4-element tuples, because the environment's state
# only includes the 4 non-control nodes (x1, x2, x3, x4).
# This is the exact config from the library's author.
source_attractor = {(0, 1, 0, 0)} # State for (x1, x2, x3, x4)
target_attractor = {(0, 0, 0, 1)} # State for (x1, x2, x3, x4)

goal_config={
    "all_attractors": [source_attractor, target_attractor],
    "target": target_attractor,
}
# --- END OF CORRECTION ---

# Define a compatible reward configuration.
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

print("   Environment from example.py created successfully.")
print(f"   Number of observable nodes: {env.unwrapped.PBN.N}")
print(f"   Action space: {env.action_space}")
print("-" * 30)


# --- Step 2: Train the Reinforcement Learning Agent ---

print("\n2. Training the Reinforcement Learning Agent (PPO)...")
start_time = time.time()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=30000)

end_time = time.time()
print(f"   Training finished in {end_time - start_time:.2f} seconds.")
print("-" * 30)


# --- Step 3: Evaluate the Learned Optimal Policy ---

print("\n3. Evaluating the learned optimal control strategy...")

# Reset the environment. It will start in one of the two attractors.
obs, info = env.reset(seed=123)

# The action map is for the single control node 'u'.
action_map = {0: "Do Nothing", 1: "Flip Node u"}
node_names = ["x1", "x2", "x3", "x4"]

# Run the simulation
for i in range(15):
    print(f"Step {i+1}:")
    print(f"  - Current State ({', '.join(node_names)}): {obs}")

    # If we are in the target state, the episode is over.
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
        # This block will be hit on the step *after* reaching the target
        print("SUCCESS: Episode terminated after reaching target.")
        break
    if truncated:
        print("EPISODE TRUNCATED: Time limit reached.")
        break

if not terminated and not truncated:
    print("EVALUATION FAILED: The agent did not reach the target within 15 steps.")

# Close the environment
env.close()
print("-" * 30)