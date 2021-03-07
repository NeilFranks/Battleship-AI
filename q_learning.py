
import numpy as np
import time
import copy

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from bs_gym_agents import BattleshipAgent
from bs_gym_env import BattleshipEnv

BOARD_DIMENSION=10
HIDDEN_SIZE=512

Q_BATCH_SIZE=512

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class Q_function_FC(nn.Module):
    def __init__(self, input_size,output_size):
        super(Q_function_FC, self).__init__()
        self.fc_in = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_out = nn.Linear(HIDDEN_SIZE, output_size)
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc(x))
        x = self.fc_out(x);#breakpoint()
        x=self.sigmoid(x)
        #x=self.tanh(x)
        return x


class Q_learning_agent(BattleshipAgent):
    def __init__(self, q_function, params, delay=0):
        self.delay = delay
        self.q_function=q_function
        
        self.params=params
        self.params['gamma']=torch.tensor(self.params['gamma'],dtype=torch.float32)
        
        self.reset_experience()
        #self.experience=list() # element in list is [obs,action,reward,next obs]
        
        self.criterion=params['criterion']
        self.optim=params['optim'](self.q_function.parameters(),lr=0.001)
        self.loss_hist=list()
    def select_action(self, observation):
        
        mask=(env.observation==0)*1
        #breakpoint()
        if np.random.random()<self.params['epsilon']:
            
            choices=np.ones(observation.shape)
            valid_choice_indices=np.argwhere(choices*mask)
            random_valid_choice_ind=np.random.choice(len(valid_choice_indices))
            action=valid_choice_indices[random_valid_choice_ind]
            
            return action
        else:
            
            mask_tensor=torch.tensor(mask,dtype=torch.float32).to(device)
            observation_tensor=torch.tensor(observation,dtype=torch.float32).to(device)
            
            observation_tensor=observation_tensor.reshape(BOARD_DIMENSION**2).unsqueeze(0)
            
            q_values=self.q_function(observation_tensor)
            q_values=q_values.reshape(10,10)
            #breakpoint()
            q_values=q_values*mask_tensor+((mask_tensor==0)*1)*-1
            
            max_ind=q_values.argmax()
            
            x,y=max_ind//BOARD_DIMENSION,max_ind%BOARD_DIMENSION
            
            action=np.array([x,y])
        #time.sleep(self.delay)
        return action

    def reset(self):
        pass
    
    def add_experience(self, sars):
        assert len(sars)==4
        #breakpoint()
        state=torch.tensor(sars[0],dtype=torch.float32)
        action=torch.tensor(sars[1],dtype=torch.long)
        reward=torch.tensor(sars[2],dtype=torch.float32)
        state_next=torch.tensor(sars[3],dtype=torch.float32)
        
        #self.experience.append(sars)
        if len(self.experience['states'])>=self.params['replay_size']: #cut off first 500 experiences to make room for 500 more
            #breakpoint()
            self.experience['states']=self.experience['states'][500:]
            self.experience['actions']=self.experience['actions'][500:]
            self.experience['rewards']=self.experience['rewards'][500:]
            self.experience['states_next']=self.experience['states_next'][500:]
            
            
        self.experience['states'].append(state)
        self.experience['actions'].append(action)
        self.experience['rewards'].append(reward)
        self.experience['states_next'].append(state_next)
        
        
    def reset_experience(self):
        self.experience={'states':list(),'actions':list(),'rewards':list(),'states_next':list()}

    def update_target_network(self):
        self.target_network=copy.deepcopy(self.q_function)
        
    def load_q_data(self):
        #breakpoint()
        self.dataset=TensorDataset(torch.stack(self.experience['states']).reshape(-1,BOARD_DIMENSION**2),
                                   torch.stack(self.experience['actions']),
                                   torch.stack(self.experience['rewards']),
                                   torch.stack(self.experience['states_next']).reshape(-1,BOARD_DIMENSION**2))
        self.loader=DataLoader(self.dataset,batch_size=Q_BATCH_SIZE,shuffle=True)
        
        
    def train(self):
        self.load_q_data()
        self.update_target_network()
        self.target_network.eval()
        
        for epoch in range(self.params['num_of_Q_epochs']):
        
            for batch_idx, (s, a, r, s_next) in enumerate(self.loader):
                
                a_inds=(a*torch.tensor([BOARD_DIMENSION,1])).sum(-1)
                
                
                self.q_function.zero_grad()
                
                q_s=self.q_function(s)
                q_s_a=q_s[torch.arange(len(q_s)),a_inds]
                q_values_next_best=self.target_network(s_next).max(-1)[0]
                q_values_next_discounted=r+self.params['gamma']*q_values_next_best
                
                #TD=q_s_a-q_values_next_discounted
                
                loss=self.criterion(q_s_a,q_values_next_discounted)
                #print(loss)
                self.loss_hist.append(loss.detach())
                loss.backward()
                self.optim.step()
                
        #print(loss)

# =============================================================================
# class Q_data(Dataset):
#     def __init__(self,experience):
#         self.experience=experience
#         
#     def __len__(self):
#         return len(self.experience)
#     def __getitem__(self,index):
#         
#         return None
# =============================================================================

q_function=Q_function_FC(BOARD_DIMENSION**2,BOARD_DIMENSION**2)

env=BattleshipEnv()

params={'alpha':.01,'epsilon':1,'epsilon_decay':.99,'epsilon_length':1000,'replay_size':5000,'gamma':.99,
        'criterion':nn.MSELoss(),'optim': torch.optim.Adam,'num_of_Q_epochs':1}

agent=Q_learning_agent(q_function=q_function,params=params)

epsilon_func=lambda a: np.exp(-a/3000)+.1

num_training_episodes=50000
shots_fired_totals = list()
global_step=0
game_steps_list=list()
num_of_shots_history=list()
#plt.figure()
for episode_num in range(num_training_episodes):
    #breakpoint()
    agent.reset()
    obs = env.reset()
    done = False
    total_reward = 0
    agent.params['epsilon']=epsilon_func(episode_num)
    #breakpoint()
    
    game_steps=0
    while not done:
        old_obs=obs
        #breakpoint()
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        
        agent.add_experience([old_obs,action,reward,obs])
        total_reward += reward
        game_steps+=1
        global_step+=1
        if global_step%100000==0:
            breakpoint()
            pass
        if global_step%1000==0:
            #breakpoint()
            agent.train()
        
    num_of_shots_history.append(game_steps)
    if len(game_steps_list)>=100:
        game_steps_list.pop(0)
        game_steps_list.append(game_steps)
    else:
        game_steps_list.append(game_steps)
        
    average_shots=np.array(game_steps_list).mean()
    if episode_num %10==0:
        print('EPISODE NUM: {0}, average number of shots of last 100 games is {1} \t\t\t epsilon: {2}'.format(episode_num,average_shots.round(2),agent.params['epsilon'].round(3)))
        
        #plt.plot(np.arange(len(num_of_shots_history)),num_of_shots_history)
        #plt.pause(.001)
    shots_fired = -total_reward
    shots_fired_totals.append(shots_fired)
