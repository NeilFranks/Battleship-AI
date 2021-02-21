import abc
from bs_gym_env import BattleshipEnv
import time
import utils
import pandas as pd
import statistics

class BattleshipAgent(abc.ABC):
    """takes game observation and returns an action"""
    @abc.abstractmethod
    def select_action(self, observation):
        """returns players next move based on the curent board"""
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class RandomBattleshipAgent(BattleshipAgent):
    def __init__(self, delay=0):
        self.delay = delay

    def select_action(self, observation):
        time.sleep(self.delay)
        return utils.pick_random_valid_action(observation)

    def reset(self):
        pass


def basic_example():
    delay = .1
    env = BattleshipEnv()
    obs = env.reset()
    agent = RandomBattleshipAgent(delay=delay)
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    shots_fired = -total_reward
    print('{} shots fired'.format(shots_fired))

AGENT_DICT =    {'random':RandomBattleshipAgent,
                }

def evaluate_agents(episodes=100):
    env = BattleshipEnv()
    results = pd.DataFrame(columns=['min', 'median', 'max', 'mean', 'std', 'avg_time', 'episodes'])
    for agent_name in AGENT_DICT:
        agent = AGENT_DICT[agent_name]()
        shots_fired_totals = []
        start = time.time()
        for _ in range(episodes):
            agent.reset()
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            shots_fired = -total_reward
            shots_fired_totals.append(shots_fired)
        #compute statistics
        avg_time = (time.time() - start) / episodes
        min_sf = min(shots_fired_totals)
        max_sf = max(shots_fired_totals)
        median_sf = statistics.median(shots_fired_totals)
        mean_sf = statistics.mean(shots_fired_totals)
        std_sf = statistics.stdev(shots_fired_totals)
        agent_results = pd.Series(data={'min':min_sf, 'median':median_sf, 
                                        'max':max_sf, 'mean':mean_sf, 
                                        'std':std_sf, 'avg_time':avg_time,
                                        'episodes':episodes},
                                        name=agent_name)
        results = results.append(agent_results)
    print(results)                                                       

if __name__ == '__main__':
    evaluate_agents()
    #basic_example()