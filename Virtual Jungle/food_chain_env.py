import pygame
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import random
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict


class FoodChainEnv(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array'], 'name': "FoodChain_v1"}

    def __init__(self, width=800, height=600, max_steps=1000, render_mode=None):
        super().__init__()

        # Screen dimensions and rendering
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Agent definitions with enhanced configurations
        self.agent_types = {
            "lion": {"level": 1, "speed": 3, "color": (255, 165, 0), "size": 25, "count": 2, "hunt_bonus": 15},
            "cheetah": {"level": 1, "speed": 5, "color": (255, 255, 0), "size": 20, "count": 2, "hunt_bonus": 12},
            "hyena": {"level": 2, "speed": 4, "color": (128, 128, 128), "size": 18, "count": 3, "hunt_bonus": 10},
            "wolf": {"level": 2, "speed": 4, "color": (200, 200, 200), "size": 18, "count": 3, "hunt_bonus": 10},
            "deer": {"level": 3, "speed": 3, "color": (139, 69, 19), "size": 15, "count": 4, "hunt_bonus": 0},
            "zebra": {"level": 3, "speed": 4, "color": (255, 255, 255), "size": 18, "count": 4, "hunt_bonus": 0}
        }

        # Create agents
        self.possible_agents = []
        for animal, config in self.agent_types.items():
            self.possible_agents += [f"{animal}_{i}" for i in range(config["count"])]

        # Action space: [stay, up, right, down, left]
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

        # Enhanced observation space: [x, y, level, health, dist_to_nearest_predator, dist_to_nearest_prey, nearest_water_dist, energy_level]
        self.observation_spaces = {
            agent: spaces.Box(0, 1, (8,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Environment state
        self.max_steps = max_steps
        self.agents = self.possible_agents[:]
        self.positions = {}
        self.health = {}
        self.energy = {}  # Energy system for more realistic behavior
        self.steps = 0
        self.water_sources = self._generate_water_sources(5)
        self.food_sources = self._generate_food_sources(8)  # For herbivores

    def _generate_water_sources(self, count):
        sources = []
        for _ in range(count):
            sources.append((random.randint(50, self.width - 50),
                            random.randint(50, self.height - 50)))
        return sources

    def _generate_food_sources(self, count):
        """Generate food sources for herbivores"""
        sources = []
        for _ in range(count):
            sources.append({
                'pos': (random.randint(50, self.width - 50), random.randint(50, self.height - 50)),
                'amount': random.randint(50, 100)
            })
        return sources

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize positions randomly with better spacing
        self.positions = {}
        self.health = {}
        self.energy = {}

        for agent in self.possible_agents:
            animal = agent.split('_')[0]
            self.positions[agent] = np.array([
                random.randint(50, self.width - 50),
                random.randint(50, self.height - 50)
            ], dtype=np.float32)
            self.health[agent] = 100.0
            self.energy[agent] = 100.0

        # Reset environment state
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.water_sources = self._generate_water_sources(5)
        self.food_sources = self._generate_food_sources(8)

        if self.render_mode == "human":
            self.render()

        return self._observe(), {}

    def _observe(self):
        observations = {}
        for agent in self.agents:
            animal_type = agent.split('_')[0]
            level = self.agent_types[animal_type]["level"]

            # Find nearest predator and prey
            nearest_pred_dist = 1.0
            nearest_prey_dist = 1.0

            for other in self.agents:
                if agent == other:
                    continue

                other_type = other.split('_')[0]
                other_level = self.agent_types[other_type]["level"]
                dist = np.linalg.norm(self.positions[agent] - self.positions[other])
                norm_dist = min(dist / max(self.width, self.height), 1.0)

                # Predator detection (higher level can hunt lower)
                if other_level < level:
                    nearest_pred_dist = min(nearest_pred_dist, norm_dist)

                # Prey detection (lower level can be hunted)
                if other_level > level:
                    nearest_prey_dist = min(nearest_prey_dist, norm_dist)

            # Find nearest water source
            nearest_water_dist = 1.0
            for water in self.water_sources:
                dist = np.linalg.norm(self.positions[agent] - np.array(water))
                norm_dist = min(dist / max(self.width, self.height), 1.0)
                nearest_water_dist = min(nearest_water_dist, norm_dist)

            # Create enhanced observation vector
            obs = np.array([
                self.positions[agent][0] / self.width,  # Normalized x
                self.positions[agent][1] / self.height,  # Normalized y
                level / 3,  # Normalized level
                self.health[agent] / 100.0,  # Health percentage
                nearest_pred_dist,  # Dist to nearest predator
                nearest_prey_dist,  # Dist to nearest prey
                nearest_water_dist,  # Distance to nearest water
                self.energy[agent] / 100.0,  # Energy level
            ], dtype=np.float32)

            observations[agent] = obs

        return observations

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Process movements with energy cost
        for agent, action in actions.items():
            if agent not in self.agents:
                continue

            animal_type = agent.split('_')[0]
            config = self.agent_types[animal_type]  # Fixed: Define config here
            speed = config["speed"]
            level = config["level"]

            # Movement with energy cost
            move_vec = np.array([0.0, 0.0])
            energy_cost = 0.1

            if action == 1:  # Up
                move_vec[1] = -speed
                energy_cost = 0.15
            elif action == 2:  # Right
                move_vec[0] = speed
                energy_cost = 0.15
            elif action == 3:  # Down
                move_vec[1] = speed
                energy_cost = 0.15
            elif action == 4:  # Left
                move_vec[0] = -speed
                energy_cost = 0.15
            # action == 0 is stay (no movement, less energy cost)

            # Update position with boundary checks
            new_pos = self.positions[agent] + move_vec
            new_pos[0] = np.clip(new_pos[0], 0, self.width)
            new_pos[1] = np.clip(new_pos[1], 0, self.height)
            self.positions[agent] = new_pos

            # Energy and health dynamics
            self.energy[agent] = max(0, self.energy[agent] - energy_cost)

            # Health decay based on energy
            if self.energy[agent] < 20:
                self.health[agent] -= 0.2  # Faster health decay when low energy
            else:
                self.health[agent] -= 0.05  # Normal health decay

            # Survival reward based on level (herbivores get more for surviving)
            base_reward = {1: 0.01, 2: 0.03, 3: 0.05}[level]
            rewards[agent] += base_reward

            # Water source benefit
            for water in self.water_sources:
                if np.linalg.norm(self.positions[agent] - np.array(water)) < 30:
                    self.health[agent] = min(100, self.health[agent] + 0.3)
                    self.energy[agent] = min(100, self.energy[agent] + 0.2)
                    rewards[agent] += 0.1
                    break

            # Food sources for herbivores
            if level == 3:  # Herbivores can eat plants
                for food in self.food_sources:
                    if food['amount'] > 0 and np.linalg.norm(self.positions[agent] - np.array(food['pos'])) < 25:
                        eat_amount = min(food['amount'], 20)
                        food['amount'] -= eat_amount
                        self.energy[agent] = min(100, self.energy[agent] + eat_amount / 2)
                        self.health[agent] = min(100, self.health[agent] + eat_amount / 4)
                        rewards[agent] += 0.5
                        break

        # Check interactions (predation) with improved logic
        agents_to_remove = []
        for predator in self.agents:
            if predator in agents_to_remove:
                continue

            p_type = predator.split('_')[0]
            p_level = self.agent_types[p_type]["level"]
            p_config = self.agent_types[p_type]

            for prey in self.agents:
                if predator == prey or prey in agents_to_remove:
                    continue

                prey_type = prey.split('_')[0]
                prey_level = self.agent_types[prey_type]["level"]

                # Check if predator can hunt this prey
                if not self._can_hunt(p_level, prey_level):
                    continue

                # Check capture distance
                distance = np.linalg.norm(
                    self.positions[predator] - self.positions[prey]
                )
                capture_dist = 25

                if distance < capture_dist and self.energy[predator] > 20:  # Need energy to hunt
                    # Calculate reward based on food chain rules
                    hunt_reward = p_config["hunt_bonus"]
                    rewards[predator] += hunt_reward

                    # Penalize prey and mark for removal
                    rewards[prey] -= 15
                    agents_to_remove.append(prey)

                    # Predator gains health and energy
                    self.health[predator] = min(100, self.health[predator] + 40)
                    self.energy[predator] = min(100, self.energy[predator] + 30)

                    # Only one prey per predator per step
                    break

        # Remove captured prey
        for prey in agents_to_remove:
            if prey in self.agents:
                self.agents.remove(prey)
                terminations[prey] = True

        # Check health, energy, and step limits
        self.steps += 1
        for agent in self.agents[:]:  # Make a copy for safe removal
            if self.health[agent] <= 0 or self.energy[agent] <= 0:
                terminations[agent] = True
                self.agents.remove(agent)
                rewards[agent] -= 5  # Death penalty

            truncations[agent] = self.steps >= self.max_steps

            # Add health and energy to info
            infos[agent]["health"] = self.health[agent]
            infos[agent]["energy"] = self.energy[agent]

        # Regenerate food sources occasionally
        if self.steps % 100 == 0:
            for food in self.food_sources:
                food['amount'] = min(100, food['amount'] + 10)

        if self.render_mode == "human":
            self.render()

        return self._observe(), rewards, terminations, truncations, infos

    def _can_hunt(self, predator_level, prey_level):
        """Define hunting rules"""
        if predator_level == 1:  # Apex predators can hunt everyone
            return prey_level > predator_level
        elif predator_level == 2:  # Mid predators can only hunt herbivores
            return prey_level == 3
        else:  # Herbivores can't hunt
            return False

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Food Chain Simulation")
            self.clock = pygame.time.Clock()

        if self.screen is None:
            return

        # Fill background
        self.screen.fill((34, 139, 34))  # Forest green

        # Draw water sources
        for water in self.water_sources:
            pygame.draw.circle(self.screen, (64, 164, 223), water, 35)
            pygame.draw.circle(self.screen, (100, 200, 255), water, 30)

        # Draw food sources (for herbivores)
        for food in self.food_sources:
            if food['amount'] > 0:
                size = max(5, int(food['amount'] / 10))
                pygame.draw.circle(self.screen, (0, 255, 0), food['pos'], size)

        # Draw agents
        for agent in self.agents:
            animal = agent.split('_')[0]
            config = self.agent_types[animal]
            pos = self.positions[agent].astype(int)
            color = config["color"]
            size = config["size"]

            # Draw agent body
            pygame.draw.circle(self.screen, color, pos, size)

            # Draw health bar
            health_width = int(size * 2 * (self.health[agent] / 100))
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (pos[0] - size, pos[1] - size - 20, size * 2, 4))
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (pos[0] - size, pos[1] - size - 20, health_width, 4))

            # Draw energy bar
            energy_width = int(size * 2 * (self.energy[agent] / 100))
            pygame.draw.rect(self.screen, (128, 128, 128),
                             (pos[0] - size, pos[1] - size - 15, size * 2, 4))
            pygame.draw.rect(self.screen, (255, 255, 0),
                             (pos[0] - size, pos[1] - size - 15, energy_width, 4))

            # Draw level indicator
            font = pygame.font.SysFont(None, 20)
            level_text = font.render(str(config["level"]), True, (0, 0, 0))
            self.screen.blit(level_text, (pos[0] - 5, pos[1] - 8))

        # Draw legend
        font = pygame.font.SysFont(None, 24)
        legend_y = 10
        for animal, config in self.agent_types.items():
            text = font.render(f"Level {config['level']}: {animal}", True, config["color"])
            self.screen.blit(text, (10, legend_y))
            legend_y += 25

        # Draw statistics
        step_text = font.render(f"Step: {self.steps}/{self.max_steps}", True, (255, 255, 255))
        self.screen.blit(step_text, (self.width - 150, 10))

        alive_text = font.render(f"Alive: {len(self.agents)}", True, (255, 255, 255))
        self.screen.blit(alive_text, (self.width - 150, 35))

        pygame.event.pump()
        pygame.display.flip()

        if self.render_mode == "human":
            self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# Test the environment with random actions
if __name__ == "__main__":
    env = FoodChainEnv(render_mode="human")
    observations, _ = env.reset()

    running = True
    while running:
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()  # Fixed: Use action_spaces

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check for exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # End episode if all agents are done or too few remain
        if len(env.agents) < 3 or all(terminations.values()) or all(truncations.values()):
            observations, _ = env.reset()

    env.close()