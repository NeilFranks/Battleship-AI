from collections import namedtuple
import random
import numpy as np

DEFAULT_GRID_SIZE = 10
DEFAULT_SHIPS = {5:1, 4:1, 3:2, 2:1}

#observation encoding
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3

ENCODING = {UNKNOWN: ' ', MISS: 'o', HIT: 'x', SUNK: '#'}


def pick_random_valid_adjacent_action(observation, row, col):
    """Returns a 2-D numpy array where each for contains the coords of a square
    adjacent to observation[row, col] and whose value is UNKNOWN.
    """
    rows, cols = observation.shape
    adj_squares = [(r,c) for r,c in [(row,col+1), (row,col-1), (row+1,col), (row-1,col)]]
    adj_squares = [(r, c) for r,c in adj_squares if r >= 0 and r < rows and c >= 0 and c < cols]
    valid_adj = np.array([(r, c) for r,c in adj_squares if observation[r,c] == UNKNOWN], dtype=np.int32)
    assert len(valid_adj) > 0, 'Error: no valid actions adjacent to row:{} col:{}'.format(row, col)
    return random.choice(valid_adj)


def possible_hit_count(obervation, ships, row, col):
    """Counts the number of ways one of the remaining ships could fit in the square.
    """
    rows, cols = obervation.shape
    possible_hits = 0
    for ship_len, ship_count in ships.items():
        count = 0
        for offset in range(1, ship_len+1):
            #check if horizontally in bounds
            left = col - ship_len + offset
            right = col + offset - 1
            if left >= 0 and right < cols:
                # check location is valid ship location (no misses or sunk ships)
                possible_ship = obervation[row, left:right+1]
                if all(possible_ship != MISS) and all(possible_ship != SUNK):
                    count += 1
            #check if vertically in bounds
            down = row - ship_len + offset
            up = row + offset - 1
            if up >= 0 and down < rows:
                possible_ship = obervation[down:up+1, col]
                if all(possible_ship != MISS) and all(possible_ship != SUNK):
                    count += 1
        count *= ship_count
        possible_hits += count
    return possible_hits


if __name__ == '__main__':
    obs = np.zeros((10,10), dtype=np.int32)

    ships = DEFAULT_SHIPS
    print(possible_hit_count(obs, ships, 5, 5))