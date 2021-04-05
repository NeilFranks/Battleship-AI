import numpy as np
import random

from bs_gym_env import BattleshipEnv
from bs_gym_agents import RandomBattleshipAgent
from utils import DEFAULT_SHIPS, UNKNOWN, MISS, HIT, SUNK, ENCODING, generate_particle, get_adjacent_squares

def test_generate_particle_WHEN_board_is_valid(n=1000):
    for check in range(n):
        if check%10 == 0:
            print(check)

        # Play games up to turn_limit until you get one that isnt done
        done = True
        while done:
            # We're gonna make a real game and run it for some number of turns, take the observation and check a particle
            env = BattleshipEnv()
            agent = RandomBattleshipAgent(delay=0)
            obs = env.reset()

            turn = 0
            turn_limit = random.randrange(0, 100)

            done = False
            while not done and turn <= turn_limit:
                action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                turn += 1

        # IMPORTANT: particle will have 0s where there is no ship, and some positive number where there is a ship
        particle = generate_particle(obs, agent.unsunk_ships)

        for i in range(len(obs)):
            for j in range(len(obs[i])):
                cell = obs[i][j]
                if (cell == HIT or cell == SUNK) and particle[i][j] == 0:
                    raise Exception('The particle is missing a ship on (%s, %s)' % (i, j))

                if cell == MISS and particle[i][j] != 0:
                    raise Exception('The particle should not have a ship on (%s, %s)' % (i, j))

    print("Found %s particles successfully!" % n)


test_generate_particle_WHEN_board_is_valid()