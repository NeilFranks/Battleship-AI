
import numpy as np
import time
import copy

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms

import matplotlib.pyplot as plt

from bs_gym_agents import BattleshipAgent
from bs_gym_env import BattleshipEnv

BOARD_DIMENSION=10
HIDDEN_SIZE=1024
Q_BATCH_SIZE=512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class Q_function_CNN(nn.Module):
    def __init__(self):
        super(Q_function_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, BOARD_DIMENSION**2)
        
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x=self.softmax(x)
        return x

class Q_learning_agent(BattleshipAgent):
    def __init__(self, q_function, params, delay=0):
        self.delay = delay
        self.q_function=q_function
        
        self.params=params
        self.params['gamma']=torch.tensor(self.params['gamma'],dtype=torch.float32)
        
        self.reset_experience()

        
        self.criterion=params['criterion']
        self.optim=params['optim'](self.q_function.parameters(),lr=self.params['alpha'])
        self.loss_hist=list()
    def select_action(self, observation,test=False):
        
        mask=(env.observation==0)*1
        if np.random.random()<self.params['epsilon'] and not test:
            
            choices=np.ones(observation.shape)
            valid_choice_indices=np.argwhere(choices*mask)
            random_valid_choice_ind=np.random.choice(len(valid_choice_indices))
            action=valid_choice_indices[random_valid_choice_ind]
            
            return action
        else:
            self.q_function.eval()
            mask_tensor=torch.tensor(mask,dtype=torch.float32).to(device)
            observation_tensor=torch.tensor(observation,dtype=torch.float32).to(device)
            
            observation_tensor=observation_tensor.view(1,1,BOARD_DIMENSION,BOARD_DIMENSION)
            q_values=self.q_function(observation_tensor)
            q_values=q_values.reshape(BOARD_DIMENSION,BOARD_DIMENSION)

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
            
            self.load_q_data()
        self.experience['states'].append(state)
        self.experience['actions'].append(action)
        self.experience['rewards'].append(reward)
        self.experience['states_next'].append(state_next)
        
        if len(self.experience['states'])>=500 and len(self.experience['states'])<4500:
            if len(self.experience['states'])%500==0:
                self.load_q_data()
        
    def reset_experience(self):
        self.experience={'states':list(),'actions':list(),'rewards':list(),'states_next':list()}

    def update_target_network(self):
        self.target_network=copy.deepcopy(self.q_function)
        
    def load_q_data(self):
        self.dataset=TensorDataset(torch.stack(self.experience['states']).reshape(-1,1,BOARD_DIMENSION,BOARD_DIMENSION),
                                   torch.stack(self.experience['actions']),
                                   torch.stack(self.experience['rewards']),
                                   torch.stack(self.experience['states_next']).reshape(-1,1, BOARD_DIMENSION, BOARD_DIMENSION))
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
        #print('loss is {0}'.format(loss.item()))
        #print(loss)
        return loss.detach().item()
    
    def train_batch(self):
        #self.load_q_data()
        #self.update_target_network()
        #self.target_network.eval()
        self.q_function.train()
        
        s,a,r,s_next=next(iter(self.loader))
            
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
        #print('loss is {0}'.format(loss.item()))
        #print(loss)
        return 
    def evaluate(self,epsisode_num,average_shots,test_epsisodes):
        
        print('EPISODE NUM: {0}, avg # of shots (last 100 games) {1} \t\t\t epsilon: {2}'.format(episode_num,average_shots.round(2),self.params['epsilon'].round(3)))

        test_steps=list()
        for test_episode in range(test_epsisodes):

            agent.reset()
            obs = env.reset()
            game_steps=0
            done=False
            while not done:
                
                old_obs=obs
                action = agent.select_action(obs,test=True)
                obs, reward, done, info = env.step(action)
                game_steps+=1
            
            test_steps.append(game_steps)
        average_shots_test=np.array(test_steps).mean()
        print('TEST number of steps: {0}'.format(average_shots_test))
        
        return average_shots_test
        
    def load_agent(self,path):
        self.q_function.load_state_dict(torch.load(path))
        self.q_function.eval()
    def save_agent(self, save_path):
        torch.save(self.q_function.state_dict(), save_path)


q_function=Q_function_CNN()

env=BattleshipEnv()

opt=torch.optim.RMSprop

params={'alpha':.0001,'epsilon':1,'epsilon_decay':.99,'epsilon_length':1000,'replay_size':50000,'gamma':.99,
        'criterion':nn.MSELoss(),'optim': opt,'num_of_Q_epochs':2}

agent=Q_learning_agent(q_function=q_function,params=params)

epsilon_func=lambda a: np.exp(-a/30000)+.1

num_training_episodes=50000
global_step=0
game_steps_list=list()
num_of_shots_history=list()
num_updates=1
C=10000
agent.update_target_network()
test_history=list()
breakpoint()
for episode_num in range(1,num_training_episodes):

    agent.reset()
    obs = env.reset()
    
    done = False
    total_reward = 0
    agent.params['epsilon']=epsilon_func(episode_num)

    test=False
    game_steps=0

    while not done:
        
        old_obs=obs

        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        
        agent.add_experience([old_obs,action,reward,obs])
        if len(agent.experience['states'])>500:
            agent.train_batch()
            num_updates+=1
        if num_updates % C == 0:
            agent.update_target_network()
            

        total_reward += reward
        game_steps+=1
        global_step+=1

    if episode_num%50==0:
        
        test=True

    num_of_shots_history.append(game_steps)
    if len(game_steps_list)>=100:
        game_steps_list.pop(0)
        game_steps_list.append(game_steps)
    else:
        game_steps_list.append(game_steps)
        
    average_shots=np.array(game_steps_list).mean()
    if episode_num %10==0:
        print('EPISODE NUM: {0}, avg # of shots (last 100 games) {1} \t\t\t epsilon: {2}'.format(episode_num,average_shots.round(2),agent.params['epsilon'].round(3)))
        pass
    if test:
        breakpoint()
        avg_test_shots=agent.evaluate(episode_num,average_shots, 100)
        
        test_history.append(avg_test_shots)
        
        
