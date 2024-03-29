import abc
from bs_gym_env import BattleshipEnv
import time
import utils
from utils import UNKNOWN, MISS, HIT, SUNK, DEFAULT_SHIPS, possible_hit_count
from utils import check_particles_are_valid, generate_particles_from_scratch
from utils import best_action_from_particles, generate_particles_from_invalid_particles, generate_particles_from_valid_particles
import pandas as pd
import statistics
import numpy as np
import random
import itertools
import tqdm  # progress bar


class BattleshipAgent(abc.ABC):
    """takes game observation and returns an action"""
    @abc.abstractmethod
    def select_action(self, observation):
        """returns players next move based on the curent board"""
        pass

    @abc.abstractmethod
    def reset(self):
        pass


def pick_randomly_strategy(observation):
    """Returns a random action which is valid given the observation.
    """
    rows, cols = observation.shape
    moves = [(r, c) for r, c in itertools.product(range(rows), range(cols))]
    valid_moves = [(r, c) for r, c in moves if observation[r, c] == UNKNOWN]
    move = random.choice(valid_moves)
    return np.array(move, dtype=np.int32)


class RandomBattleshipAgent(BattleshipAgent):
    def __init__(self, delay=0, ships=None):
        self.ships = ships or DEFAULT_SHIPS  # keep track of ships remaining
        self.delay = delay
        self.reset()

    def select_action(self, observation):
        time.sleep(self.delay)

        # update self.unsunk_ships, if anything sunk
        if self.previous_action is not None:  # first turn
            # if previous shot sunk a ship, update unsunk_ships
            row, col = self.previous_action
            if observation[row, col] == SUNK:
                sunk_coords = [tuple(c) for c in np.transpose(
                    np.where(observation == SUNK))]
                new_coords = [
                    c for c in sunk_coords if c not in self.previously_sunk_ship_coords]
                self.previously_sunk_ship_coords = sunk_coords
                recently_sunk_ship_length = len(new_coords)
                self.unsunk_ships[recently_sunk_ship_length] -= 1

        action = pick_randomly_strategy(observation)
        self.previous_action = action
        return self.previous_action

    def reset(self):
        self.unsunk_ships = self.ships.copy()
        self.previously_sunk_ship_coords = []
        self.previous_action = None


def search_left(observation, c, row):
    while c >= 0 and observation[row, c] == HIT:
        c -= 1
    if c >= 0 and observation[row, c] == UNKNOWN:
        return np.array((row, c), dtype=np.int32)
    else:
        return None


def search_right(observation, c, row, num_cols):
    while c < num_cols and observation[row, c] == HIT:
        c += 1
    if c < num_cols and observation[row, c] == UNKNOWN:
        return np.array((row, c), dtype=np.int32)
    else:
        return None


def search_horizontal(observation, row, col, num_cols):
    if (col > 0 and observation[row, col-1] == HIT) or \
            (col < num_cols-1 and observation[row, col+1] == HIT):
        if random.choice(['l', 'r']) == 'l':
            # continue search left until non-hit encountered or edge of board
            result = search_left(observation, col, row)
            if type(result) != type(None):
                return result

            # continue search right until non-hit encountered or edge of board
            result = search_right(observation, col, row, num_cols)
            if type(result) != type(None):
                return result
        else:
            # continue search right until non-hit encountered or edge of board
            result = search_right(observation, col, row, num_cols)
            if type(result) != type(None):
                return result

            # continue search left until non-hit encountered or edge of board
            result = search_left(observation, col, row)
            if type(result) != type(None):
                return result
    return None


def search_up(observation, r, col):
    while r >= 0 and observation[r, col] == HIT:
        r -= 1
    if r >= 0 and observation[r, col] == UNKNOWN:
        return np.array((r, col), dtype=np.int32)
    else:
        return None


def search_down(observation, r, col, num_rows):
    while r < num_rows and observation[r, col] == HIT:
        r += 1
    if r < num_rows and observation[r, col] == UNKNOWN:
        return np.array((r, col), dtype=np.int32)
    else:
        return None


def search_vertical(observation, row, col, num_rows):
    if (row > 0 and observation[row-1, col] == HIT) or \
            (row < num_rows-1 and observation[row+1, col] == HIT):
        if random.choice(['u', 'd']) == 'u':
            # continue search up until non-hit encountered or edge of board
            result = search_up(observation, row, col)
            if type(result) != type(None):
                return result

            # continue search down until non-hit encountered or edge of board
            result = search_down(observation, row, col, num_rows)
            if type(result) != type(None):
                return result
        else:
            # continue search down until non-hit encountered or edge of board
            result = search_down(observation, row, col, num_rows)
            if type(result) != type(None):
                return result

            # continue search up until non-hit encountered or edge of board
            result = search_up(observation, row, col)
            if type(result) != type(None):
                return result

    return None


def look_adjacent_strategy(observation, known_hit):
    """Pick best square nearby a known hit to shoot at next.
    """
    num_rows, num_cols = observation.shape
    row, col = known_hit

    if random.choice(['h', 'v']) == 'h':
        # look horizontally for adjacent hits
        result = search_horizontal(observation, row, col, num_cols)
        if type(result) != type(None):
            return result

        # look vertically for adjacent hits
        result = search_vertical(observation, row, col, num_rows)
        if type(result) != type(None):
            return result
    else:
        # look vertically for adjacent hits
        result = search_vertical(observation, row, col, num_rows)
        if type(result) != type(None):
            return result

        # look horizontally for adjacent hits
        result = search_horizontal(observation, row, col, num_cols)
        if type(result) != type(None):
            return result

    # if reached, no adjacent hits -> pick random valid adjacent square
    row, col = known_hit
    return utils.pick_random_valid_adjacent_action(observation, row, col)


class RandomWithPreferredActionsAgent(BattleshipAgent):
    """If there is a known hit, shoot around it until it is sunk, otherwise, shoot randomly."""

    def __init__(self, delay=0):
        self.delay = delay
        self.reset()

    def reset(self):
        pass

    def select_action(self, observation):
        time.sleep(self.delay)
        hits = np.transpose(np.where(observation == HIT))
        # if there are no known hits, shoot randomly
        if len(hits) == 0:
            action = pick_randomly_strategy(observation)
        else:
            action = look_adjacent_strategy(observation, hits[0])
        return action


def pick_most_probable_strategy(observation, ships):
    possible_actions = np.transpose(np.where(observation == UNKNOWN))
    assert len(possible_actions) > 0, 'Error: no possible actions available.'
    greatest_possible_hits = 0
    best_action = possible_actions[0]
    for action in possible_actions:
        row, col = action
        possible_hits = utils.possible_hit_count(observation, ships, row, col)
        if possible_hits > greatest_possible_hits:
            greatest_possible_hits = possible_hits
            best_action = action
    return best_action


class ProbabilisticAgent(BattleshipAgent):
    """
    Shoot the square for which there is greatest number of ways a ship could occupy it,
    given the current observation.
    """

    def __init__(self, delay=0, ships=None):
        self.ships = ships or DEFAULT_SHIPS  # keep track of ships remaining
        self.delay = delay
        self.reset()

    def reset(self):
        self.unsunk_ships = self.ships.copy()
        self.previously_sunk_ship_coords = []
        self.previous_action = None

    def select_action(self, observation):
        time.sleep(self.delay)

        # update self.unsunk_ships, if anything sunk
        if self.previous_action is not None:  # first turn
            # if previous shot sunk a ship, update unsunk_ships
            row, col = self.previous_action
            if observation[row, col] == SUNK:
                sunk_coords = [tuple(c) for c in np.transpose(
                    np.where(observation == SUNK))]
                new_coords = [
                    c for c in sunk_coords if c not in self.previously_sunk_ship_coords]
                self.previously_sunk_ship_coords = sunk_coords
                recently_sunk_ship_length = len(new_coords)
                self.unsunk_ships[recently_sunk_ship_length] -= 1

        # if there are no known hits, shoot probabilistically
        hits = np.transpose(np.where(observation == HIT))
        if len(hits) == 0:
            action = pick_most_probable_strategy(
                observation, self.unsunk_ships.copy())
        else:  # follow adjacent strategy
            action = look_adjacent_strategy(observation, hits[0])
        self.previous_action = action
        return self.previous_action


class ParticleFilterAgent(BattleshipAgent):
    """
    Maintain a belief state built from particles, and infer your best shot from the particles.
    `k` refers to the number of particles to maintain.
    """

    def __init__(self, delay=0, ships=None, k=180, board_width=10, board_height=10):
        self.ships = ships or DEFAULT_SHIPS  # keep track of ships remaining
        self.delay = delay
        self.k = k
        self.EMPTY_BOARD = np.zeros(
            (board_height, board_width), dtype=np.int32)
        self.reset()

    def reset(self):
        self.unsunk_ships = self.ships.copy()
        self.previously_sunk_ship_coords = []
        self.previous_action = None
        self.particles = generate_particles_from_scratch(
            self.k, self.EMPTY_BOARD, self.unsunk_ships
        )

    def select_action(self, observation):
        time.sleep(self.delay)

        # update self.unsunk_ships, if anything sunk
        if self.previous_action is not None:  # first turn
            # if previous shot sunk a ship, update unsunk_ships
            row, col = self.previous_action
            if observation[row, col] == SUNK:
                sunk_coords = [tuple(c) for c in np.transpose(
                    np.where(observation == SUNK))]
                new_coords = [
                    c for c in sunk_coords if c not in self.previously_sunk_ship_coords]
                self.previously_sunk_ship_coords = sunk_coords
                recently_sunk_ship_length = len(new_coords)
                self.unsunk_ships[recently_sunk_ship_length] -= 1

        # update particles
        valid_particles, invalid_particles = check_particles_are_valid(
            self.particles, observation, self.previous_action)

        # particle reinvigoration
        self.particles = valid_particles
        self.particles.extend(
            generate_particles_from_scratch(
                self.k-len(valid_particles), observation, self.unsunk_ships
            )
        )

        # find best action from particles
        action = best_action_from_particles(observation, self.particles)

        self.previous_action = action
        return self.previous_action


def basic_example():
    delay = .1
    env = BattleshipEnv()
    obs = env.reset()
    # agent = RandomWithPreferredActionsAgent(delay=delay)
    agent = ProbabilisticAgent(delay=delay)

    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    shots_fired = -total_reward
    print('{} shots fired'.format(shots_fired))


AGENT_DICT = {
    'Random Agent': RandomBattleshipAgent,
    'Random w/ Preferred Actions Agent': RandomWithPreferredActionsAgent,
    'Probabilistic Agent': ProbabilisticAgent,
    'Particle Filter Agent': ParticleFilterAgent,
}


def evaluate_agents(episodes=100):
    env = BattleshipEnv()
    results = pd.DataFrame(
        columns=['min', 'median', 'max', 'mean', 'std', 'avg_time', 'episodes'])
    print('Agents to be evaluated: {}'.format(list(AGENT_DICT.keys())))
    for agent_name in AGENT_DICT:
        agent = AGENT_DICT[agent_name]()
        shots_fired_totals = []
        start = time.time()
        for _ in tqdm.tqdm(range(episodes), desc='Evaluating {} agent'.format(agent_name, )):
            agent.reset()
            obs = env.reset()
            done = False
            total_reward = 0
            shots_fired = 0
            while not done:
                action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                shots_fired += 1
            shots_fired_totals.append(shots_fired)
        # compute statistics
        avg_time = (time.time() - start) / episodes
        min_sf = min(shots_fired_totals)
        max_sf = max(shots_fired_totals)
        median_sf = statistics.median(shots_fired_totals)
        mean_sf = statistics.mean(shots_fired_totals)
        std_sf = statistics.stdev(shots_fired_totals)
        agent_results = pd.Series(data={'min': min_sf, 'median': median_sf,
                                        'max': max_sf, 'mean': mean_sf,
                                        'std': std_sf, 'avg_time': avg_time,
                                        'episodes': episodes},
                                  name=agent_name)
        results = results.append(agent_results)
    print(results)


if __name__ == '__main__':
    # breakpoint()
    evaluate_agents()
    # basic_example()
