import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE, SPEED
from model import Linear_QNet, QTrainer
from helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # controls exploration
        self.gamma = 0.9# discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # if we exceed this memory it will remove ele from left
        self.model = Linear_QNet(11, 256, 3)  # input size is 11, hidden layer size is 256, output size is 3 (left, right, straight)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self,game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state=[
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),  # Danger straight

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),  # Danger right

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or   
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),  # Danger left

            dir_l,  # Moving left
            dir_r,  # Moving right
            dir_u,  # Moving up
            dir_d,  # Moving down

            #Foodo
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y   # Food down


        ]
        return np.array(state, dtype=int)
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action, reward, next_state, done))#pop left when max memory is reached
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, action, rewards, next_states, done = zip(*mini_sample)
        self.trainer.train_step(states, action, rewards, next_states, done)
    

    

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state,action, reward, next_state, done)
    def get_action(self,state):
        #rand
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]  
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent=Agent()
    game= SnakeGameAI()
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old,final_move, reward, state_new, done)

        agent.remember(state_old,final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.gamma += 0.001
            if score > record:
                record = score
                agent.model.save()
                # torch.save(agent.model.state_dict(), 'model.pth')
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / (agent.n_games + 1)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Decay epsilon
            if agent.epsilon > 0.01:
                agent.epsilon *= 0.99

        

if __name__ == "__main__":
    train()

