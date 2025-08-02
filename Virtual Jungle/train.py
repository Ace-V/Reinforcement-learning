import gymnasium as gym
import os
import ray
from food_chain_env import FoodChainEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium import spaces
import numpy as np
from collections import deque
import time
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv



print(f"Ray version: {ray.__version__}")

# Initialize Ray
ray.init(ignore_reinit_error=True, log_to_driver=False)

# Environment registration with PettingZoo wrapper
def env_creator(config=None):
    return PettingZooEnv(FoodChainEnv(width=800, height=600, max_steps=1000))
register_env("food_chain", env_creator)

# Policy mapping function based on animal type level
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    animal_type = agent_id.split('_')[0]
    temp_env = FoodChainEnv()
    level = temp_env.agent_types[animal_type]["level"]
    return f"level{level}_policy"

# Simple training monitor to decide convergence
class SimpleTrainingMonitor:
    def __init__(self, window_size=50):
        self.recent_rewards = {'level1': deque(maxlen=window_size),
                               'level2': deque(maxlen=window_size),
                               'level3': deque(maxlen=window_size)}
        self.iterations = []

    def update(self, iteration, result):
        rewards = result.get('policy_reward_mean', {})
        self.recent_rewards['level1'].append(rewards.get('level1_policy', 0))
        self.recent_rewards['level2'].append(rewards.get('level2_policy', 0))
        self.recent_rewards['level3'].append(rewards.get('level3_policy', 0))
        self.iterations.append(iteration)

    def should_stop(self, iteration, min_iterations=200):
        if iteration < min_iterations:
            return False, f"Need at least {min_iterations} iterations"
        if len(self.recent_rewards['level1']) < 20:
            return False, "Building reward history"
        def cv(data): return np.std(data) / (abs(np.mean(data)) + 1e-8)
        recent_l1 = list(self.recent_rewards['level1'])[-15:]
        recent_l2 = list(self.recent_rewards['level2'])[-15:]
        recent_l3 = list(self.recent_rewards['level3'])[-15:]
        stable = (cv(recent_l1) < 0.2 and cv(recent_l2) < 0.2 and cv(recent_l3) < 0.2)
        reasonable = (np.mean(recent_l1) > 1 and np.mean(recent_l3) > -8)
        if stable and reasonable:
            return True, "Converged"
        if iteration > 1500:
            return True, "Max iterations reached"
        return False, f"CVs: L1={cv(recent_l1):.3f}, L2={cv(recent_l2):.3f}, L3={cv(recent_l3):.3f}"

# Build PPOConfig with new RLlib API stack model config
config = (
    PPOConfig()
    .environment("food_chain")
    .framework("torch")
    # Enable the new API stack (default).
)

# Set model config using new api_stack method
config.rl_module(model_config={
    "fcnet_hiddens": [64, 64],
    "fcnet_activation": "relu",
})

# Multi-agent policies setup
obs_space = spaces.Box(0, 1, (8,), dtype=np.float32)
act_space = spaces.Discrete(5)
config = config.multi_agent(
    policies={
        "level1_policy": (None, obs_space, act_space, {"gamma": 0.95}),
        "level2_policy": (None, obs_space, act_space, {"gamma": 0.98}),
        "level3_policy": (None, obs_space, act_space, {"gamma": 0.99}),
    },
    policy_mapping_fn=policy_mapping_fn,
    policies_to_train=["level1_policy", "level2_policy", "level3_policy"]
)

# Assign training hyperparameters directly
config.gamma = 0.98
config.lr = 5e-4
config.train_batch_size = 2000
config.sgd_minibatch_size = 128
config.num_sgd_iter = 3
config.clip_param = 0.2
config.vf_loss_coeff = 0.5
config.entropy_coeff = 0.01

# Disable rollout workers on some Ray versions if needed
try:
    config.num_env_runners = 0
except Exception:
    try:
        config.num_rollout_workers = 0
    except Exception:
        pass
# Use no GPUs here
try:
    config.num_gpus = 0
except Exception:
    pass

# Build the PPO algorithm instance
try:
    algo = config.build()
except Exception as e:
    print(f"Error building algorithm: {e}")
    ray.shutdown()
    exit(1)

print("Starting training loop...")

monitor = SimpleTrainingMonitor()
try:
    for iteration in range(2000):
        result = algo.train()
        monitor.update(iteration, result)
        if iteration % 20 == 0:
            rewards = result.get('policy_reward_mean', {})
            print(f"Iter {iteration:4d} | "
                  f"L1: {rewards.get('level1_policy', 0):+.2f} | "
                  f"L2: {rewards.get('level2_policy', 0):+.2f} | "
                  f"L3: {rewards.get('level3_policy', 0):+.2f}")
        if iteration % 75 == 0 and iteration > 0:
            should_stop, reason = monitor.should_stop(iteration)
            print(f" Status: {reason}")
            if should_stop:
                print("Training stopping due to convergence or max iterations.")
                break
        if iteration % 300 == 0 and iteration > 0:
            checkpoint_dir = f"checkpoints/iter_{iteration}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_path}")
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
except Exception as e:
    print(f"Training error: {e}")

# Final save of the trained model
final_dir = "trained_models/food_chain_final"
os.makedirs(final_dir, exist_ok=True)
final_checkpoint = algo.save(final_dir)
print(f"Training completed. Model saved to {final_checkpoint}")

ray.shutdown()
