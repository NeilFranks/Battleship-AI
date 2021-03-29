import os
import numpy as np
from gym import core, spaces
import random
from utils import DEFAULT_GRID_SIZE, DEFAULT_SHIPS, UNKNOWN, MISS, HIT, SUNK, ENCODING


class BattleshipEnv(core.Env):
    def __init__(self, width=None, height=None, ships=None):
        low = min([UNKNOWN, MISS, HIT, SUNK])
        high = max([UNKNOWN, MISS, HIT, SUNK])
        self.cols = width or DEFAULT_GRID_SIZE
        self.rows = height or DEFAULT_GRID_SIZE
        self.observation_space = spaces.Box(low=low, high=high, shape=(self.rows, self.cols), 
                                            dtype='int32')
        action_low = np.zeros(2, dtype=np.int32)
        action_high = np.array([self.rows-1, self.cols-1], dtype=np.int32)
        # an action is a 1-D numpy array of the form array([row, col], dtype=np.int32)
        # so row = action[0] and col = action[1]
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype='int32')
        self.ships = ships or DEFAULT_SHIPS
        self.ship_lengths_by_id = dict()
        
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        zeros out the state and randomly placed ships in valid locations.
        returns a blank observation (all locations are UNKNOWN)
        """
        #debugging, only one board 
        #np.random.seed(0)
        #random.seed(0)
        #breakpoint()
        self.shots_fired = 0
        self.state = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.place_ships()
        self.observation = np.ones((self.rows, self.cols), dtype=np.int32) * UNKNOWN
        return self.observation.copy()

    def step(self, action):
        """Runs one timestep and returns the next observation.

        This means that the agents action is evaluated to see if it is a hit or a miss.
        self.observation is updated accordingly, and a copy of it is returned
        to the agent.
        """
        self.shots_fired += 1

        row, col = action[0], action[1]
        assert row >= 0 and row < self.rows, ('Error: action row value (action[0]'
                                              ') must satisfy 0 <= row <= {} but '
                                              'instead was {}').format(self.rows - 1, row)
        assert col >= 0 and col < self.cols, ('Error: action col value (action[1]'
                                             ') must satisfy 0 <= col <= {} but i'
                                             'nstead was {}').format(self.cols - 1, col)
        #assert self.observation[row, col] == UNKNOWN, ('Error: row {} col {} has '
        #                                              'already been fired at!').format(row, col)
        

        hit_unknown = True if self.observation[row,col]==UNKNOWN else False #true if the shot is to an unknown spot
        sunk_ship = False # keeps track if the shot is part of a sunk ship
        #update observation
        if self.state[row, col] == 0: #if its a miss, mark it as such
            self.observation[row, col] = MISS
        else: #since it's a hit, mark it as such and check if the ship is sunk
            self.observation[row, col] = HIT
            ship_id = self.state[row, col]
            ship_coords = np.where(self.state == ship_id)
            if all(self.observation[ship_coords] == HIT):
                self.observation[ship_coords] = SUNK #ship sunk
                sunk_ship= True
                #del self.unsunk_ship_lengths_by_id[ship_id]


        if not hit_unknown: #if shot already marked spot, punish (not sure if this ever happens)
            breakpoint()
            reward = -2 
        else:
            #breakpoint()
            if self.state[row,col] !=0:
                reward = 1
                if sunk_ship: #if hitting a new spot (hit_unknown == True) AND sunk_ship == True, then we know the action caused a ship to sink
                    reward = 3
                
            else:
                reward=-.1




        #check if the game is over (all ships sunk)
        ship_locations = np.where(self.state > 0)
        is_game_over = all(self.observation[ship_locations] == SUNK)

        # reward is set to -1, because we are trying to win in least steps possible
        #reward = -1

        return self.observation.copy(), reward, is_game_over, None

    def render(self):
        """Update display based most recent observation"""
        os.system('clear||cls') # clear terminal
        print('Shots Fired: {}'.format(self.shots_fired))
        line = '    ' + '  '.join('{:2}'.format(k) for k in range(self.cols))
        print(line)
        horizontal_line = '-' * (4 * self.cols - 1)
        print('   +' + horizontal_line + '+')
        for i in range(self.rows):
            row = self.observation[i]
            line = '{:2} |'.format(i)
            for j in row:
                line += ' ' + ENCODING[j] + ' |'
            print(line)
            if i < self.rows -1:
                print('   |' + horizontal_line + '|')
        print('   +' + horizontal_line + '+')

    def close(self):
        pass

    def place_ships(self, strategy=None):
        col_max = self.cols - 1
        row_max = self.rows - 1
        ship_id = 1
        for ship_len in self.ships:
            for _ in range(self.ships[ship_len]):
                placed = False
                while not placed:
                    row, col = random.randint(0, row_max), random.randint(0, col_max)
                    if self.state[row, col]: #is there already a ship here?
                        continue
                    direction = random.randint(0,3)
                    if direction == 0: #right
                        if col + ship_len > col_max + 1: #ship out of bounds
                            continue
                        if sum(self.state[row, col:col+ship_len]) > 0: #ship already present
                            continue
                        self.state[row, col:col+ship_len] = ship_id
                    elif direction == 1: #down
                        if row + ship_len > row_max + 1: #ship out of bounds
                            continue
                        if sum(self.state[row:row+ship_len, col]) > 0: #ship already present
                            continue
                        self.state[row:row+ship_len, col] = ship_id
                    elif direction == 2: #left
                        if col - ship_len + 1 < 0: #ship out of bounds
                            continue
                        if sum(self.state[row, col-ship_len+1:col+1]) > 0: #ship already present
                            continue
                        self.state[row, col-ship_len+1:col+1] = ship_id
                    else: #up
                        if row - ship_len + 1 < 0: #ship out of bounds
                            continue
                        if sum(self.state[row-ship_len+1:row+1, col]) > 0: #ship already present
                            continue
                        self.state[row-ship_len+1:row+1, col] = ship_id
                    placed = True
                    self.ship_lengths_by_id[ship_id] = ship_len
                ship_id += 1

if __name__ == '__main__':
    env = BattleshipEnv(width=7, height=6)
    env.reset()
    env.step(np.array([3,2], dtype=np.int32))
    env.render()
