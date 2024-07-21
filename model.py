import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes) #first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)
         
    def forward(self, x):
        
        x=F.relu(self.fc1(x)) #relu activation function
        x=self.out(x) #calculate output
        return x
        
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class SnakeDQN: # needs changing... episodes as a param?
    def __init__(self, model, lr, gamma, target):
        self.lr = lr   # 0.001         # learning rate (alpha)
        self.gamma = gamma
        self.model = model
        self.target = target
        self.policyQList1 = []
        self.targetQList1 = []

    # Neural Network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  
        self.criterion = nn.MSELoss() # NN Loss function. MSE=Mean Squared Error can be swapped to something else?

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        policyQList = []
        targetQList = []

        #if len(state.shape) == 1:
                # (1, x)
            #state = torch.unsqueeze(state, 0)
            #next_state = torch.unsqueeze(next_state, 0)
            #action = torch.unsqueeze(action, 0)
            #reward = torch.unsqueeze(reward, 0)
            #done = (done, )

        if done:
            targetQ = torch.FloatTensor([reward])
        else: 
            with torch.no_grad():
                targetQ = torch.FloatTensor(
                        reward + self.gamma * self.target(next_state).max()
                    )

            # 1: predicted Q values with current state
        predPolicy = self.model(state) # policy net prediction of state
        policyQList.append(predPolicy)
        print(policyQList)

        AverageQ = torch.mean(predPolicy)
        
        predTarget = self.target(next_state) # target net prediction of s' (next state)
        predTarget[action] = targetQ
        targetQList.append(predTarget)
        

        #target = pred.clone()
       # for idx in range(len(done)):
           # Q_new = reward[idx]
            #if not done[idx]:
                #Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            #target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        loss = self.criterion(torch.stack(policyQList), torch.stack(targetQList))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return AverageQ
     


# I need a 
# init with lr, gamma, model, optimiser, MSELoss
# train step
'''
    class QTrainer:
        def __init__(self, model, lr, gamma):
            self.lr = lr
            self.gamma = gamma
            self.model = model
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()

        def train_step(self, state, action, reward, next_state, done):
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            # (n, x)

            if len(state.shape) == 1:
                # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )

            # 1: predicted Q values with current state
            pred = self.model(state)

            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new
        
            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()'''
