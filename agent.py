import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # discount rate was 0.95
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(38, 340, 3) #middle nodes were 256
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        


        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        point_ll = Point(head.x - 40, head.y)
        point_rr = Point(head.x + 40, head.y)
        point_uu = Point(head.x, head.y - 40)
        point_dd = Point(head.x, head.y + 40)

        point_lll = Point(head.x - 60, head.y)
        point_rrr = Point(head.x + 60, head.y)
        point_uuu = Point(head.x, head.y - 60)
        point_ddd = Point(head.x, head.y + 60)

        point_llll = Point(head.x - 80, head.y)
        point_rrrr = Point(head.x + 80, head.y)
        point_uuuu = Point(head.x, head.y - 80)
        point_dddd = Point(head.x, head.y + 80)

        point_lllll = Point(head.x - 100, head.y)
        point_rrrrr = Point(head.x + 100, head.y)
        point_uuuuu = Point(head.x, head.y - 100)
        point_ddddd = Point(head.x, head.y + 100)

        point_llllll = Point(head.x - 120, head.y)
        point_rrrrrr = Point(head.x + 120, head.y)
        point_uuuuuu = Point(head.x, head.y - 120)
        point_dddddd = Point(head.x, head.y + 120)

        point_ur = Point(head.x + 20, head.y - 20)
        point_ul = Point(head.x - 20, head.y - 20)
        point_dr = Point(head.x + 20, head.y + 20)
        point_dl = Point(head.x - 20, head.y + 20)

        point_urr = Point(head.x + 40, head.y - 20)
        point_ull = Point(head.x - 40, head.y - 20)
        point_drr = Point(head.x + 40, head.y + 20)
        point_dll = Point(head.x - 40, head.y + 20)

        point_uur = Point(head.x + 20, head.y - 40)
        point_uul = Point(head.x - 20, head.y - 40)
        point_ddr = Point(head.x + 20, head.y + 40)
        point_ddl = Point(head.x - 20, head.y + 40)

        

        point_urur = Point(head.x + 40, head.y - 40)
        point_ulul = Point(head.x - 40, head.y - 40)
        point_drdr = Point(head.x + 40, head.y + 40)
        point_dldl = Point(head.x - 40, head.y + 40)

        points = [
        Point(head.x - 20, head.y),   # point_l
        Point(head.x + 20, head.y),   # point_r
        Point(head.x, head.y - 20),   # point_u
        Point(head.x, head.y + 20),   # point_d
        Point(head.x - 40, head.y),   # point_ll
        Point(head.x + 40, head.y),   # point_rr
        Point(head.x, head.y - 40),   # point_uu
        Point(head.x, head.y + 40),   # point_dd
        Point(head.x - 60, head.y),   # point_lll
        Point(head.x + 60, head.y),   # point_rrr
        Point(head.x, head.y - 60),   # point_uuu
        Point(head.x, head.y + 60),   # point_ddd
        Point(head.x - 80, head.y),   # point_llll
        Point(head.x + 80, head.y),   # point_rrrr
        Point(head.x, head.y - 80),   # point_uuuu
        Point(head.x, head.y + 80),   # point_dddd
        Point(head.x - 100, head.y),  # point_lllll
        Point(head.x + 100, head.y),  # point_rrrrr
        Point(head.x, head.y - 100),  # point_uuuuu
        Point(head.x, head.y + 100),  # point_ddddd
        Point(head.x + 20, head.y - 20),  # point_ur
        Point(head.x - 20, head.y - 20),  # point_ul
        Point(head.x + 20, head.y + 20),  # point_dr
        Point(head.x - 20, head.y + 20),  # point_dl
        Point(head.x + 40, head.y - 40),  # point_urur
        Point(head.x - 40, head.y - 40),  # point_ulul
        Point(head.x + 40, head.y + 40),  # point_drdr
        Point(head.x - 40, head.y + 40),  # point_dldl
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

            # Danger straight 2
            (dir_r and game.is_collision(point_rr)) or 
            (dir_l and game.is_collision(point_ll)) or 
            (dir_u and game.is_collision(point_uu)) or 
            (dir_d and game.is_collision(point_dd)),

            # Danger right 2
            (dir_u and game.is_collision(point_rr)) or 
            (dir_d and game.is_collision(point_ll)) or 
            (dir_l and game.is_collision(point_uu)) or 
            (dir_r and game.is_collision(point_dd)),

            # Danger left 2
            (dir_d and game.is_collision(point_rr)) or 
            (dir_u and game.is_collision(point_ll)) or 
            (dir_r and game.is_collision(point_uu)) or 
            (dir_l and game.is_collision(point_dd)),

            # Danger straight 3
            (dir_r and game.is_collision(point_rrr)) or 
            (dir_l and game.is_collision(point_lll)) or 
            (dir_u and game.is_collision(point_uuu)) or 
            (dir_d and game.is_collision(point_ddd)),

            # Danger right 3
            (dir_u and game.is_collision(point_rrr)) or 
            (dir_d and game.is_collision(point_lll)) or 
            (dir_l and game.is_collision(point_uuu)) or 
            (dir_r and game.is_collision(point_ddd)),

            # Danger left 3
            (dir_d and game.is_collision(point_rrr)) or 
            (dir_u and game.is_collision(point_lll)) or 
            (dir_r and game.is_collision(point_uuu)) or 
            (dir_l and game.is_collision(point_ddd)),

            # Danger straight 4
            (dir_r and game.is_collision(point_rrrr)) or 
            (dir_l and game.is_collision(point_llll)) or 
            (dir_u and game.is_collision(point_uuuu)) or 
            (dir_d and game.is_collision(point_dddd)),

            # Danger right 4
            (dir_u and game.is_collision(point_rrrr)) or 
            (dir_d and game.is_collision(point_llll)) or 
            (dir_l and game.is_collision(point_uuuu)) or 
            (dir_r and game.is_collision(point_dddd)),

            # Danger left 4
            (dir_d and game.is_collision(point_rrrr)) or 
            (dir_u and game.is_collision(point_llll)) or 
            (dir_r and game.is_collision(point_uuuu)) or 
            (dir_l and game.is_collision(point_dddd)),

            # Danger straight 5
            (dir_r and game.is_collision(point_rrrrr)) or 
            (dir_l and game.is_collision(point_lllll)) or 
            (dir_u and game.is_collision(point_uuuuu)) or 
            (dir_d and game.is_collision(point_ddddd)),

            # Danger right 5
            (dir_u and game.is_collision(point_rrrrr)) or 
            (dir_d and game.is_collision(point_lllll)) or 
            (dir_l and game.is_collision(point_uuuuu)) or 
            (dir_r and game.is_collision(point_ddddd)),

            # Danger left 5
            (dir_d and game.is_collision(point_rrrrr)) or 
            (dir_u and game.is_collision(point_lllll)) or 
            (dir_r and game.is_collision(point_uuuuu)) or 
            (dir_l and game.is_collision(point_ddddd)),

             # Danger straight 6
            (dir_r and game.is_collision(point_rrrrrr)) or 
            (dir_l and game.is_collision(point_llllll)) or 
            (dir_u and game.is_collision(point_uuuuuu)) or 
            (dir_d and game.is_collision(point_dddddd)),

            # Danger right 6
            (dir_u and game.is_collision(point_rrrrrr)) or 
            (dir_d and game.is_collision(point_lllll)) or 
            (dir_l and game.is_collision(point_uuuuuu)) or 
            (dir_r and game.is_collision(point_dddddd)),

            # Danger left 6
            (dir_d and game.is_collision(point_rrrrrr)) or 
            (dir_u and game.is_collision(point_llllll)) or 
            (dir_r and game.is_collision(point_uuuuuu)) or 
            (dir_l and game.is_collision(point_dddddd)),

            # Danger up to the left
            (dir_u and game.is_collision(point_ul)) or 
            (dir_d and game.is_collision(point_dr)) or 
            (dir_r and game.is_collision(point_ur)) or 
            (dir_l and game.is_collision(point_dl)),

            # Danger up to the right
            (dir_u and game.is_collision(point_ur)) or 
            (dir_d and game.is_collision(point_dl)) or 
            (dir_l and game.is_collision(point_ul)) or 
            (dir_r and game.is_collision(point_dr)),

            # Danger up to the left up to the left
            (dir_u and game.is_collision(point_ulul)) or 
            (dir_d and game.is_collision(point_drdr)) or 
            (dir_r and game.is_collision(point_urur)) or 
            (dir_l and game.is_collision(point_dldl)),

            # Danger up to the right up to the right
            (dir_u and game.is_collision(point_urur)) or 
            (dir_d and game.is_collision(point_dldl)) or 
            (dir_l and game.is_collision(point_ulul)) or 
            (dir_r and game.is_collision(point_drdr)),

            # Danger urr
            (dir_u and game.is_collision(point_urr)) or 
            (dir_d and game.is_collision(point_dll)) or 
            (dir_l and game.is_collision(point_uul)) or 
            (dir_r and game.is_collision(point_ddr)),
            
            # Danger uur
            (dir_u and game.is_collision(point_uur)) or 
            (dir_d and game.is_collision(point_ddl)) or 
            (dir_l and game.is_collision(point_ull)) or 
            (dir_r and game.is_collision(point_drr)),
            
            # Danger uul
            (dir_u and game.is_collision(point_uul)) or 
            (dir_d and game.is_collision(point_ddr)) or 
            (dir_l and game.is_collision(point_dll)) or 
            (dir_r and game.is_collision(point_urr)),

            # Danger ull
            (dir_u and game.is_collision(point_ull)) or 
            (dir_d and game.is_collision(point_drr)) or 
            (dir_l and game.is_collision(point_ddl)) or 
            (dir_r and game.is_collision(point_uur)),

            # Danger dll
            (dir_u and game.is_collision(point_dll)) or 
            (dir_d and game.is_collision(point_urr)) or 
            (dir_l and game.is_collision(point_ddr)) or 
            (dir_r and game.is_collision(point_uul)),
            
            # Danger ddl
            (dir_u and game.is_collision(point_ddl)) or 
            (dir_d and game.is_collision(point_uur)) or 
            (dir_l and game.is_collision(point_drr)) or 
            (dir_r and game.is_collision(point_ull)),

            # Danger ddr
            (dir_u and game.is_collision(point_ddr)) or 
            (dir_d and game.is_collision(point_uul)) or 
            (dir_l and game.is_collision(point_urr)) or 
            (dir_r and game.is_collision(point_dll)),

            # Danger drr
            (dir_u and game.is_collision(point_drr)) or 
            (dir_d and game.is_collision(point_ull)) or 
            (dir_l and game.is_collision(point_uur)) or 
            (dir_r and game.is_collision(point_ddl)),


            #Distance to food
            #Distance to tail
       
            
            
            
            
            # Snake's body presence
            #*body_presence, #28 inputs

            

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # was 80
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: # making it low probalility of being random
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
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        
        if done:
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


if __name__ == '__main__':
    train()
