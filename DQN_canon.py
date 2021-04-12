
import numpy as np
import time
import copy
import json

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms

#import matplotlib.pyplot as plt

from bs_gym_agents import BattleshipAgent
from bs_gym_env import BattleshipEnv

from canon_obs import Canonicalizer
from utils import DEFAULT_SHIPS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


boardsize_cnn={10:6,9:5,8:4,7:3,6:2,5:1}



class Q_function_CNN(nn.Module):
    def __init__(self,board_dimension,hidden_size):
        super(Q_function_CNN, self).__init__()
        
        self.board_dimension=board_dimension
        self.extra_dim=boardsize_cnn[self.board_dimension]
        self.conv1 = nn.Conv2d(1, 32, 3,)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32 * self.extra_dim * self.extra_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.board_dimension**2)
        
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * self.extra_dim * self.extra_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #x=self.softmax(x)
        return x

class Q_learning_agent(BattleshipAgent):
    def __init__(self, q_function, params, board_dimension, delay=0):
        self.delay = delay
        self.BOARD_DIMENSION=board_dimension
        self.q_function=q_function
        
        self.params=params
        self.params['gamma']=torch.tensor(self.params['gamma'],dtype=torch.float32)
        
        self.reset_experience()

        self.criterion=params['criterion']
        self.optim=params['optim'](self.q_function.parameters(),lr=self.params['alpha'],momentum=.95,eps=0.1)
        self.loss_hist=list()

    def select_action(self, observation,test=False):
        mask=(observation==0)*1
        if np.random.random()<self.params['epsilon'] and not test:
            choices=np.ones(observation.shape)
            valid_choice_indices=np.argwhere(choices*mask)
            random_valid_choice_ind=np.random.choice(len(valid_choice_indices))
            action=valid_choice_indices[random_valid_choice_ind]
            
            return action
        else:
            self.q_function.eval()
            observation_tensor=torch.tensor(observation.copy(),dtype=torch.float32).to(device)
            
            observation_tensor=observation_tensor.view(1,1,self.BOARD_DIMENSION,self.BOARD_DIMENSION)
            ind_to_select_from=((observation_tensor.reshape(self.BOARD_DIMENSION**2)<1)*1).nonzero()
            
            q_values=self.q_function(observation_tensor)
            filtered_q_values=q_values[0][ind_to_select_from]
            max_ind=filtered_q_values.argmax()
            #q_values=q_values.reshape(self.BOARD_DIMENSION,self.BOARD_DIMENSION)

            max_ind=ind_to_select_from[max_ind]
            #max_ind=q_values.argmax()
            
            x,y=max_ind//self.BOARD_DIMENSION,max_ind%self.BOARD_DIMENSION
            
            action=np.array([x.detach().item(),y.detach().item()])
        return action

    def reset(self):
        pass
    
    def add_experience(self, sars):
        assert len(sars)==4

        state=torch.tensor(sars[0].copy(),dtype=torch.float32)
        action=torch.tensor(sars[1],dtype=torch.long)
        reward=torch.tensor(sars[2],dtype=torch.float32)
        state_next=torch.tensor(sars[3].copy(),dtype=torch.float32)
        
        if len(self.experience['states'])>=self.params['replay_size']: #cut off first 500 experiences to make room for 500 more

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
                try:
                    self.load_q_data()
                except:
                    breakpoint()
                    self.load_q_data()
        
    def reset_experience(self):
        self.experience={'states':list(),'actions':list(),'rewards':list(),'states_next':list()}

    def update_target_network(self):
        self.target_network=copy.deepcopy(self.q_function)
        
    def load_q_data(self):
        self.dataset=TensorDataset(torch.stack(self.experience['states']).reshape(-1,1,self.BOARD_DIMENSION,self.BOARD_DIMENSION),
                                   torch.stack(self.experience['actions']),
                                   torch.stack(self.experience['rewards']),
                                   torch.stack(self.experience['states_next']).reshape(-1,1, self.BOARD_DIMENSION, self.BOARD_DIMENSION))
        self.loader=DataLoader(self.dataset,batch_size=self.params['q_batch_size'],shuffle=True)
    
    def train_batch_base(self):

        self.q_function.train()
        
        s,a,r,s_next=next(iter(self.loader))
            
        a_inds=(a*torch.tensor([self.BOARD_DIMENSION,1])).sum(-1)
        
        self.q_function.zero_grad()
        
        inds_to_select_from=list()
        for state in s_next:
            ind_to_select_from=((state.reshape(self.BOARD_DIMENSION**2)<1)*1).nonzero()
            inds_to_select_from.append(ind_to_select_from)
        
        s=s.to(device)
        s_next=s_next.to(device)
        q_s=self.q_function(s)
        q_s_a=q_s[torch.arange(len(q_s)),a_inds]
        #breakpoint()
        q_values_next=self.target_network(s_next)
        q_values_next_best=list()
        for idx,inds in enumerate(inds_to_select_from):
            filtered_q_values=q_values_next[idx][inds]
            try:
                next_max_q=filtered_q_values.max()
            except:
                breakpoint()
            q_values_next_best.append(next_max_q)
        q_values_next_best=torch.stack(q_values_next_best)
                
        #breakpoint()
        q_values_next_discounted=r.to(device)+self.params['gamma']*q_values_next_best
        
        loss=self.criterion(q_s_a,q_values_next_discounted)
        #print(loss)
        self.loss_hist.append(loss.detach())
        loss.backward()
        self.optim.step()

        return
    
    def evaluate(self, env, test_epsisodes, canon=None):
        test_steps=list()
        test_reward=list()
        for test_episode in range(test_epsisodes):
            self.reset()
            obs = env.reset()
            game_steps=0
            reward_cumulative=0
            done=False
            while not done:
                if canon is not None:
                    c_obs, c_tuple = canon.canon_obs(obs, return_tuple=True)
                    c_action = self.select_action(c_obs, test=True)
                    action = canon.uncanon_action(c_action, c_tuple)
                else:
                    action = self.select_action(obs, test=True)
                obs, reward, done, info = env.step(action)
                reward_cumulative+=reward
                game_steps+=1
            
            test_steps.append(game_steps)
            test_reward.append(reward_cumulative)
        average_shots_test=np.array(test_steps).mean()
        average_reward=np.array(test_reward).mean()
        print('TEST number of steps: {0}, reward: {1}'.format(average_shots_test,average_reward))
        
        return average_shots_test,average_reward
        
    def load_agent(self,path):
        self.q_function.load_state_dict(torch.load(path))
        self.q_function.eval()

    def save_agent(self, save_path):
        torch.save(self.q_function.state_dict(), save_path)
        

# =============================================================================
# Parameters in Paper
# BOARD_DIMENSION=10
# HIDDEN_SIZE=512
# Q_BATCH_SIZE=32
# opt=torch.optim.RMSprop
# 
# DEFAULT_PARAMS={'alpha':.00025,'epsilon':1,'num_episodes':120,'epsilon_length':1000,'replay_size':50000,'gamma':.99,
#         'criterion':nn.MSELoss(),'optim': opt, 'C':10000,'num_of_Q_epochs':2, 'hidden_size':HIDDEN_SIZE,'q_batch_size': Q_BATCH_SIZE, 'test_episodes':100}
# =============================================================================



BOARD_DIMENSION=10
HIDDEN_SIZE=512
Q_BATCH_SIZE=32
opt=torch.optim.RMSprop

DEFAULT_PARAMS={'alpha':.00025,'epsilon':1,'num_episodes':50000,'epsilon_length':1000,'replay_size':50000,'gamma':.99,
        'criterion':nn.MSELoss(),'optim': opt, 'C':10000,'num_of_Q_epochs':2, 'hidden_size':HIDDEN_SIZE,'q_batch_size': Q_BATCH_SIZE, 'test_episodes':100}



def train_q_agent(board_dimension=BOARD_DIMENSION, hidden_size=HIDDEN_SIZE, params=DEFAULT_PARAMS, ships=DEFAULT_SHIPS, use_canon=False, save_name='test_save.pt'):
    q_function = Q_function_CNN(board_dimension=board_dimension,hidden_size=hidden_size)
    q_function.to(device)
    env = BattleshipEnv(board_dimension,board_dimension, ships=ships)

    agent = Q_learning_agent(q_function=q_function,params=params,board_dimension=board_dimension)
    epsilon_func = lambda a: np.float64(max(1-a*.90/200000,.1))
    
    num_training_episodes=params['num_episodes']
    global_step=0
    game_steps_list=list()
    num_of_shots_history=list()
    num_updates=1
    agent.update_target_network()
    test_history=list()
    performance_dict={'reward_hist':list(),'avg_shot_hist':list()}

    if use_canon:
        canon = Canonicalizer(board_dimension)
    else:
        canon = None

    for episode_num in range(1,num_training_episodes):
        agent.reset()
        obs = env.reset()
        if use_canon:
            c_obs = canon.canon_obs(obs)

        done = False
        total_reward = 0
        agent.params['epsilon']=epsilon_func(num_updates)
    
        test=False
        game_steps=0
    
        while not done:
            if use_canon:
                old_c_obs = c_obs
                c_obs, c_tuple = canon.canon_obs(obs, return_tuple=True)
                c_action = agent.select_action(c_obs)
                action = canon.uncanon_action(c_action, c_tuple)
            else:
                old_obs=obs
                action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            if done:
                game_steps+=1
                global_step+=1
                total_reward+=reward
                break
            if use_canon:
                agent.add_experience([old_c_obs,c_action,reward,c_obs])
            else:
                agent.add_experience([old_obs,action,reward,obs])

            if len(agent.experience['states'])>500:
                agent.train_batch_base()
                num_updates+=1
            if num_updates % agent.params['C'] == 0:
                agent.update_target_network()
                
            total_reward += reward
            game_steps+=1
            global_step+=1

        #breakpoint()
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
            print('EPISODE NUM: {0}, avg # of shots (last 100 games) {1} \t\t\t epsilon: {2}'.format(episode_num,average_shots.round(2),agent.params['epsilon'].round(4)))
            pass
        if test:

            avg_test_shots,avg_reward=agent.evaluate(env, agent.params['test_episodes'], canon=canon)
            performance_dict['reward_hist'].append(avg_reward)
            performance_dict['avg_shot_hist'].append(avg_test_shots)
            test_history.append(avg_test_shots)
            if use_canon:
                log_fname='{0}x{0}_hidden{1}_canon_log.json'.format(board_dimension,hidden_size)
            else:
                log_fname='{0}x{0}_hidden{1}_log.json'.format(board_dimension,hidden_size)
            
            with open(log_fname,"w") as f:
                json.dump(performance_dict, f)
            
    if use_canon:
        log_fname='{0}x{0}_hidden{1}_canon_log.json'.format(board_dimension,hidden_size)
    else:
        log_fname='{0}x{0}_hidden{1}_log.json'.format(board_dimension,hidden_size)
    
    with open(log_fname,"w") as f:
        json.dump(performance_dict, f)
    
    if use_canon:
        agent.save_agent('{0}x{0}_hidden{1}_episodes{2}_canon_model.pt'.format(board_dimension,hidden_size,num_training_episodes))
    else:
        agent.save_agent('{0}x{0}_hidden{1}_episodes{2}_model.pt'.format(board_dimension,hidden_size,num_training_episodes))



if __name__ == '__main__':
    params = DEFAULT_PARAMS.copy()
    params['num_episodes'] = 100

    train_q_agent(board_dimension=10, params=params, use_canon=False)