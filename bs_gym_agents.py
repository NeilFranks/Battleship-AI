import abc
from bs_gym_env import BattleshipEnv
import time
import utils

class BattleshipPlayer(abc.ABC):
    """takes game observation and returns an action"""
    @abc.abstractmethod
    def select_action(self, observation):
        """returns players next move based on the curent board"""
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class RandomBattleshipPlayer(BattleshipPlayer):
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
    player = RandomBattleshipPlayer(delay=delay)
    done = False
    total_reward = 0
    while not done:
        action = player.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    shots_fired = -total_reward
    print('{} shots fired'.format(shots_fired))


if __name__ == '__main__':
    basic_example()