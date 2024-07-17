
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import DQN, SnakeDQN
from helper import plot
import math


MAX_MEMORY = 100_000
BATCH_SIZE = 500
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # discount rate was 0.95
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = DQN(20*20, 256, 3) #middle nodes were 256
        self.trainer = SnakeDQN(self.model, lr=LR, gamma=self.gamma)


        '''
     def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)


        points = [
        Point(head.x - 20, head.y),   # point_l
        Point(head.x + 20, head.y),   # point_r
        Point(head.x, head.y - 20),   # point_u
        Point(head.x, head.y + 20),   # point_d
        
        ]

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
       

        return np.array(state, dtype=int)
         '''
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached    
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def len(self):
        return len(self.memory)
    
    def train_long_memory(self): # train agent from random things in deque
        if len(self.memory) > BATCH_SIZE:
            mini_sample = self.sample(BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        #states, actions, rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

   # def train_short_memory(self, state, action, reward, next_state, done): #commented this out for testing
       # self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # was 80
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: # reducing probalility of being random #was 200
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).view(1, -1)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = game.convertToState() # has been changed

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        if done: #necessary to keep 
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            
            #x = np.arange(len(plot_scores))
            #slope, intercept = np.polyfit(x, plot_scores, 1)
            #regression_line = slope * x + intercept
           # cleaplot_regression_line.append(regression_line)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
        else:
            state_new = game.convertToState()
        # train short memory
       # agent.train_short_memory(state_old, final_move, reward, state_new, done) #im hoping if I comment this out things will work
        # remember
            agent.remember(state_old, final_move, reward, state_new, done)

        


if __name__ == '__main__':
    train()