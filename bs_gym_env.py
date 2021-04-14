import os
import numpy as np
from gym import core, spaces
import random
from utils import DEFAULT_GRID_SIZE, DEFAULT_SHIPS, UNKNOWN, MISS, HIT, SUNK, ENCODING
from utils import show_ships


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
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype='int32')
        self.ships = ships or DEFAULT_SHIPS
        self.ship_lengths_by_id = dict()

    def reset(self, favor_top=0, favor_bottom=0, favor_left=0, favor_right=0, gradient_coef=lambda x: x, vert_probability=0.5):
        """Resets the environment to an initial state and returns an initial
        observation.

        zeros out the state and randomly placed ships in valid locations.
        returns a blank observation (all locations are UNKNOWN)
        """
        # debugging, only one board
        # np.random.seed(0)
        # random.seed(0)
        # breakpoint()
        self.shots_fired = 0
        self.state = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.place_ships(favor_top=favor_top, favor_bottom=favor_bottom, favor_left=favor_left,
                         favor_right=favor_right, gradient_coef=gradient_coef, vert_probability=vert_probability)
        self.observation = np.ones(
            (self.rows, self.cols), dtype=np.int32) * UNKNOWN
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
        # assert self.observation[row, col] == UNKNOWN, ('Error: row {} col {} has '
        #                                              'already been fired at!').format(row, col)

        # true if the shot is to an unknown spot
        hit_unknown = True if self.observation[row, col] == UNKNOWN else False
        sunk_ship = False  # keeps track if the shot is part of a sunk ship
        # update observation
        if self.state[row, col] == 0:  # if its a miss, mark it as such
            self.observation[row, col] = MISS
        else:  # since it's a hit, mark it as such and check if the ship is sunk
            self.observation[row, col] = HIT
            ship_id = self.state[row, col]
            ship_coords = np.where(self.state == ship_id)
            if all(self.observation[ship_coords] == HIT):
                self.observation[ship_coords] = SUNK  # ship sunk
                sunk_ship = True
                #del self.unsunk_ship_lengths_by_id[ship_id]

        assert hit_unknown, 'You shold not reach this point. Shot at already known area.'

        if self.state[row, col] != 0:
            reward = 1
            # if hitting a new spot (hit_unknown == True) AND sunk_ship == True, then we know the action caused a ship to sink
            if sunk_ship:
                reward = 3

        else:
            reward = -.1

        # check if the game is over (all ships sunk)
        ship_locations = np.where(self.state > 0)
        is_game_over = all(self.observation[ship_locations] == SUNK)

        # reward is set to -1, because we are trying to win in least steps possible
        #reward = -1

        return self.observation.copy(), reward, is_game_over, None

    def render(self):
        """Update display based most recent observation"""
        os.system('clear||cls')  # clear terminal
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
            if i < self.rows - 1:
                print('   |' + horizontal_line + '|')
        print('   +' + horizontal_line + '+')

    def close(self):
        pass

    def place_ships(self, favor_top=0, favor_bottom=0, favor_left=0, favor_right=0, gradient_coef=lambda x: x, vert_probability=0.5):
        """
        `favor_top`, `favor_bottom`, `favor_left`, and `favor_right` determine the likelihood a boat is placed in a certain position. 
        For example, if the value of `favor_top` is a large positive number, the odds of a boat being placed toward the top is high. 
        If the value is instead a large negative number, the odds of a boat being placed toward the top is low. 
        A value of zero gives no preference one way or another.

        `gradient_coef`: This lambda function determines how each striation of the gradient should be calculated

        `vert_probability` should be from 0 to 1. Represents probability of placing a ship vertically vs horizontally. 
        """
        if vert_probability < 0 or vert_probability > 1:
            raise ValueError("vert_probability must be between 0 and 1")

        # Initialize a grid of preference values;
        # this will determine how likely a boat is to be placed on each square
        # grid of ones means every cell is equally likely
        preferences = np.ones((self.rows, self.cols), dtype=np.float32)

        # we will use these gradients to apply our `favor` arguments in multiple ways
        vert_gradient = np.zeros((self.rows, self.cols), dtype=np.float32)
        hor_gradient = np.zeros((self.rows, self.cols), dtype=np.float32)
        for i in range(len(vert_gradient)//2, len(vert_gradient)):
            vert_gradient[i] = gradient_coef(i-len(vert_gradient)//2)

        for i in range(len(hor_gradient)):
            for j in range(len(hor_gradient[i])//2, len(hor_gradient[i])):
                hor_gradient[i][j] = gradient_coef(j-len(hor_gradient[i])//2)

        preferences += favor_top*np.flipud(vert_gradient)
        preferences += favor_bottom*vert_gradient
        preferences += favor_left*np.fliplr(hor_gradient)
        preferences += favor_right*hor_gradient

        # we should refactor preferences so the lowest value is 1
        preferences -= (preferences.min()-1) * \
            np.ones((self.rows, self.cols), dtype=np.float32)

        # we will need a flattened vector of board coordinates for our call to `random.choices()`
        flattened_board_coords = []
        for i in range(self.rows):
            for j in range(self.cols):
                flattened_board_coords.append((i, j))

        ship_id = 1
        for ship_len in self.ships:
            for _ in range(self.ships[ship_len]):
                placed = False
                while not placed:
                    row, col = random.choices(
                        flattened_board_coords, weights=list(preferences.flatten()))[0]
                    if self.state[row, col]:  # is there already a ship here?
                        continue

                    # let's see if we should do vertical or horizontal
                    if random.choices(['v', 'h'], weights=[vert_probability, 1-vert_probability])[0] == 'v':
                        # You will be placing vertically
                        # if out-of-bounds, or ship already present
                        if row+ship_len > self.rows or sum(self.state[row:row+ship_len, col]) > 0:
                            # cannot place downwards. Check upwards
                            # if out-of-bounds, or ship already present
                            if row-ship_len+1 < 0 or sum(self.state[row-ship_len+1:row+1, col]) > 0:
                                # cannot place upwards either. Out of luck on this one
                                continue
                            else:
                                # can place upwards, so do it!
                                self.state[row-ship_len+1:row+1, col] = ship_id
                        else:
                            # can place downwards! Check if you can place upwards, too
                            # if out-of-bounds, or ship already present
                            if row-ship_len+1 < 0 or sum(self.state[row-ship_len+1:row+1, col]) > 0:
                                # cannot place upwards. So do downwards!
                                self.state[row:row+ship_len, col] = ship_id
                            else:
                                # can place upwards as well as downwards. Pick one
                                if preferences[row-1][col] > preferences[row+1][col]:
                                    # upwards is more preferable
                                    self.state[row-ship_len +
                                               1:row+1, col] = ship_id
                                else:
                                    # downwards is more preferable
                                    self.state[row:row+ship_len, col] = ship_id
                    else:
                        # You will be placing horizontally
                        # if out-of-bounds, or ship already present
                        if col+ship_len > self.cols or sum(self.state[row, col:col+ship_len]) > 0:
                            # cannot place rightwards. Check leftwards
                            # if out-of-bounds, or ship already present
                            if col-ship_len+1 < 0 or sum(self.state[row, col-ship_len+1:col+1]) > 0:
                                # cannot place leftwards either. Out of luck on this one
                                continue
                            else:
                                # can place leftwards, so do it!
                                self.state[row, col-ship_len+1:col+1] = ship_id
                        else:
                            # can place rightwards! Check if you can place leftwards, too
                            # if out-of-bounds, or ship already present
                            if col-ship_len+1 < 0 or sum(self.state[row, col-ship_len+1:col+1]) > 0:
                                # cannot place leftwards. So do rightwards!
                                self.state[row, col:col+ship_len] = ship_id
                            else:
                                # can place leftwards as well as rightwards. Pick one
                                if preferences[row][col-1] > preferences[row][col+1]:
                                    # leftwards is more preferable
                                    self.state[row, col -
                                               ship_len+1:col+1] = ship_id
                                else:
                                    # rightwards is more preferable
                                    self.state[row, col:col+ship_len] = ship_id
                    placed = True
                    self.ship_lengths_by_id[ship_id] = ship_len
                ship_id += 1


if __name__ == '__main__':
    env = BattleshipEnv(width=10, height=10)
    env.reset()
    env.step(np.array([3, 2], dtype=np.int32))
    env.render()
    # show_ships(env.state)
