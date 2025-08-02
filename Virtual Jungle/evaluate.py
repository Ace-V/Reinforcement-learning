import os
import ray
import pygame
import numpy as np
import time
from food_chain_env import FoodChainEnv
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from collections import defaultdict
import matplotlib.pyplot as plt
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

ray.init(ignore_reinit_error=True, log_to_driver=False)

def env_creator(config=None):
    return PettingZooEnv(FoodChainEnv(width=800, height=600, max_steps=1000, render_mode="human"))
register_env("food_chain", env_creator)

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    animal_type = agent_id.split('_')[0]
    temp_env = FoodChainEnv()
    level = temp_env.agent_types[animal_type]["level"]
    return f"level{level}_policy"

class FoodChainEvaluator:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.algo = None
        self.env = None
        self.load_model()

    def load_model(self):
        try:
            self.algo = PPO.from_checkpoint(self.checkpoint_path)
            print(f"Successfully loaded model from {self.checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the model first using train.py")
            raise

    def evaluate_episode(self, render=True, max_steps=1000):
        env = FoodChainEnv(width=800, height=600, max_steps=max_steps, render_mode="human" if render else None)
        observations, _ = env.reset()
        episode_rewards = defaultdict(float)
        episode_stats = {
            'steps': 0, 'hunts_successful': 0, 'agents_died': 0,
            'survival_by_level': defaultdict(int),
            'final_populations': defaultdict(int),
        }
        running = True
        step_count = 0
        while running and step_count < max_steps:
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    animal_type = agent.split('_')[0]
                    level = env.agent_types[animal_type]["level"]
                    policy_id = f"level{level}_policy"
                    action = self.algo.compute_single_action(
                        observations[agent],
                        policy_id=policy_id
                    )[0]
                    actions[agent] = action
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent, reward in rewards.items():
                episode_rewards[agent] += reward
            for agent in terminations:
                if terminations[agent]:
                    episode_stats['agents_died'] += 1
            step_count += 1
            episode_stats['steps'] = step_count
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                time.sleep(0.05)
            if len(env.agents) == 0:
                break
            remaining_levels = set()
            for agent in env.agents:
                animal_type = agent.split('_')[0]
                level = env.agent_types[animal_type]["level"]
                remaining_levels.add(level)
            if len(remaining_levels) <= 1:
                break
        for agent in env.agents:
            animal_type = agent.split('_')[0]
            level = env.agent_types[animal_type]["level"]
            episode_stats['survival_by_level'][level] += 1
            episode_stats['final_populations'][animal_type] += 1
        env.close()
        return episode_rewards, episode_stats

    def run_evaluation(self, num_episodes=5, render_first=True):
        print(f"Running {num_episodes} evaluation episodes...")
        all_rewards, all_stats = [], []
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            render = render_first and (episode == 0)
            rewards, stats = self.evaluate_episode(render=render)
            all_rewards.append(rewards)
            all_stats.append(stats)
            print(f" Steps: {stats['steps']}")
            print(f" Agents died: {stats['agents_died']}")
            print(f" Final populations: {dict(stats['final_populations'])}")
            print(f" Survival by level: {dict(stats['survival_by_level'])}")
        return all_rewards, all_stats

    def analyze_results(self, all_rewards, all_stats):
        print("\n" + "=" * 60)
        print("EVALUATION ANALYSIS")
        print("=" * 60)
        avg_episode_length = np.mean([stats['steps'] for stats in all_stats])
        avg_deaths = np.mean([stats['agents_died'] for stats in all_stats])
        print(f"Average episode length: {avg_episode_length:.1f} steps")
        print(f"Average deaths per episode: {avg_deaths:.1f}")
        # Survival analysis by level
        survival_rates = defaultdict(list)
        for stats in all_stats:
            total_by_level = defaultdict(int)
            temp_env = FoodChainEnv()
            for animal, config in temp_env.agent_types.items():
                total_by_level[config['level']] += config['count']
            for level in [1,2,3]:
                survived = stats['survival_by_level'][level]
                initial = total_by_level[level]
                rate = survived / initial if initial > 0 else 0
                survival_rates[level].append(rate)
        print("\nSurvival Rates by Trophic Level:")
        for level in [1,2,3]:
            avg_survival = np.mean(survival_rates[level])
            std_survival = np.std(survival_rates[level])
            level_name = {1: "Apex Predators", 2: "Mid Predators", 3: "Herbivores"}[level]
            print(f" Level {level} ({level_name}): {avg_survival:.1%} ± {std_survival:.1%}")
        # Reward analysis
        reward_by_level = defaultdict(list)
        for episode_rewards in all_rewards:
            level_totals, level_counts = defaultdict(float), defaultdict(int)
            for agent, total_reward in episode_rewards.items():
                animal_type = agent.split('_')[0]
                temp_env = FoodChainEnv()
                level = temp_env.agent_types[animal_type]["level"]
                level_totals[level] += total_reward
                level_counts[level] += 1
            for level in [1,2,3]:
                if level_counts[level] > 0:
                    avg_reward = level_totals[level] / level_counts[level]
                    reward_by_level[level].append(avg_reward)
        print("\nAverage Rewards by Trophic Level:")
        for level in [1,2,3]:
            if reward_by_level[level]:
                avg_reward = np.mean(reward_by_level[level])
                std_reward = np.std(reward_by_level[level])
                level_name = {1: "Apex Predators", 2: "Mid Predators", 3: "Herbivores"}[level]
                print(f" Level {level} ({level_name}): {avg_reward:.2f} ± {std_reward:.2f}")
        self.create_evaluation_plots(survival_rates, reward_by_level, all_stats)
        return {
            'survival_rates': survival_rates,
            'reward_by_level': reward_by_level,
            'avg_episode_length': avg_episode_length,
            'avg_deaths': avg_deaths,
        }

    def create_evaluation_plots(self, survival_rates, reward_by_level, all_stats):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        levels = [1,2,3]
        level_names = ["Apex\nPredators", "Mid\nPredators", "Herbivores"]
        survival_means = [np.mean(survival_rates[level]) for level in levels]
        survival_stds = [np.std(survival_rates[level]) for level in levels]
        colors = ['red', 'orange', 'green']
        bars1 = ax1.bar(level_names, survival_means, yerr=survival_stds, color=colors, alpha=0.7, capsize=5)
        ax1.set_ylabel('Survival Rate')
        ax1.set_title('Average Survival Rates by Trophic Level')
        ax1.set_ylim(0, 1)
        for bar, mean in zip(bars1, survival_means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{mean:.1%}', ha='center', va='bottom')
        reward_means, reward_stds = [], []
        for level in levels:
            if reward_by_level[level]:
                reward_means.append(np.mean(reward_by_level[level]))
                reward_stds.append(np.std(reward_by_level[level]))
            else:
                reward_means.append(0)
                reward_stds.append(0)
        bars2 = ax2.bar(level_names, reward_means, yerr=reward_stds, color=colors, alpha=0.7, capsize=5)
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Rewards by Trophic Level')
        episode_lengths = [stats['steps'] for stats in all_stats]
        ax3.hist(episode_lengths, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Episode Length (Steps)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Episode Lengths')
        ax3.axvline(np.mean(episode_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(episode_lengths):.1f}')
        ax3.legend()
        deaths = [stats['agents_died'] for stats in all_stats]
        episodes = list(range(1, len(deaths) + 1))
        ax4.plot(episodes, deaths, 'o-', color='red', alpha=0.7)
