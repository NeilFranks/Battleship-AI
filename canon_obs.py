""" Tool for reducing the observation space of battleship by roughly a factor
of eight.

idea: The optimal action for a given observation is the same as the respective
action for a rotation or reflection of that observation. We denote the eight
observations that can be created by rotating and flipping an observation as
the observation's "symmetry set". canon_obs provides a function which maps
all members of a given symmetry set to one particular member of that set which
we call the canonical observation for that set.

The point is that during training and evaluation of various ML battleship agents,
by training on only the canonical observations we can effectively reduce the
number of possible states by roughly a factor of eight and thus speed up training.

method: we treat each observation as a base 4 number (since possible cell values
are 0, 1, 2, and 3). canon-obs stretchs each observation in the symmetry set of
the given observation into a vector and evaluates its value as a base 4 number. 
Each observation in the set evaluates to a different number unless the they're
identical (e.g. obs comprised of all 0's is the same even when rotated or 
reflected), in which case canonicalization does not matter anyway. We canon-obs
returns the observation from the set with the greatest base 4 value.
"""

import numpy as np
import timeit

import bs_gym_env
from utils import DEFAULT_GRID_SIZE

QUARTERNARY_POWERS = np.array([4**p for p in range(DEFAULT_GRID_SIZE ** 2)])

class Canonicalizer:
    def __init__(self, grid_size=DEFAULT_GRID_SIZE):
        self.grid_size = grid_size
        self.quarternary_powers = np.array([4**p for p in range(self.grid_size ** 2)])

    def _obs_to_int(self, obs):
        """stretch obs into vector, evaluate as base 4 number
        """
        obs_vec = obs.flatten()
        return np.dot(obs_vec, self.quarternary_powers)

    def canon_obs(self, obs, return_tuple=False):
        """ Given an observation, calculates the canonical observation from its
        symmetry set. Returns the canonical observation and (conditionally) a 
        tuple describing the transformations performed to get it.

        tuple of the form (mirror, num_rotations):
            mirror: boolean specifying if the observation should be flipped along
                axis 0
            num_rotations: int in range [0, 4] specifying the number of times to
                rotate the observation counterclockwise (after mirroring it, if 
                mirror=True)
        """
        obs = obs.copy()
        mirror_obs = np.flip(obs, axis=0)
        obs_values = []
        # tuple describing the transformation performed to each obs
        canon_tuples = dict() 
        for mirror in [True, False]:
            obs_to_use = mirror_obs if mirror else obs
            for num_rot in range(4): # rotate 90 either 0, 1, 2, or 3 times
                rot_obs = np.rot90(obs_to_use, k=num_rot)
                obs_value = self._obs_to_int(rot_obs)
                canon_tuples[obs_value] = (mirror, num_rot)
                obs_values.append((obs_value, rot_obs))
        c_obs_value, c_obs = max(obs_values, key=lambda x: x[0])
        if not return_tuple:
            return c_obs
        c_tuple = canon_tuples[c_obs_value]
        return c_obs, c_tuple

    def uncanon_action(self, action, canon_tuple):
        """ Performs the opposite transformation used to canonicalize an observation
        to the given action.
        """
        row, col = action
        for _ in range(canon_tuple[1]):
            row, col = col, self.grid_size - 1 - row
        if canon_tuple[0]: # was the observation mirrored?
            row = self.grid_size - 1 - row
        return np.array((row, col))

#
# Functions to evaluate the Canonicalizer
#

def _passes_test(obs, canon):
    """ Verify whether canon_obs can correctly canonicalize the given obervation.

    Meaning, will canon_obs return the same observation when fed all eight rotations
    and flipped rotations of the given observation? If so, returns True, else returns
    False.
    """
    canon_obs = canon.canon_obs
    c_obs = canon_obs(obs)
    try:
        assert np.array_equal(c_obs, canon_obs(np.rot90(obs, k=1)))
        assert np.array_equal(c_obs, canon_obs(np.rot90(obs, k=2)))
        assert np.array_equal(c_obs, canon_obs(np.rot90(obs, k=3)))
        mirror_obs = np.flip(obs, axis=0)
        assert np.array_equal(c_obs, canon_obs(mirror_obs))
        assert np.array_equal(c_obs, canon_obs(np.rot90(mirror_obs, k=1)))
        assert np.array_equal(c_obs, canon_obs(np.rot90(mirror_obs, k=2)))
        assert np.array_equal(c_obs, canon_obs(np.rot90(mirror_obs, k=3)))
        return True
    except AssertionError:
        return False

def _uncanon_passes_test(obs, action, canon):
    c_obs, c_tuple = canon.canon_obs(obs, return_tuple=True)
    c_obs[action[0], action[1]] = -1
    unc_action = canon.uncanon_action(action, c_tuple)
    obs[unc_action[0], unc_action[1]] = -1
    c_obs = np.rot90(c_obs, k=-c_tuple[1])
    if c_tuple[0]:
        c_obs = np.flip(c_obs, axis=0)
    return np.array_equal(obs, c_obs)

def _evaluate_uncanon_action():
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    action_space = env.action_space
    canon = Canonicalizer()
    num_trials = 0
    num_failures = 0
    latest_obs_action_tuple_that_fails = None
    try:
        while(True):
            obs = obs_space.sample()
            action = action_space.sample()
            if not _uncanon_passes_test(obs, action, canon):
                num_failures += 1
                if latest_obs_action_tuple_that_fails is None:
                    print('First observation action tuple failed after {} trials. Tuple:'.format(num_trials + 1))
                    print(repr(obs))
                    print(repr(action))
                latest_obs_action_tuple_that_fails = (obs, action)
            num_trials += 1
    except KeyboardInterrupt:
        print('\nPercentage of actions uncanonicalized out of {}:'.format(num_trials))
        print('{:0.5f}%'.format((1 - (num_failures / num_trials)) * 100))

def _evaluate_pass_rate():
    """ Continuously test canon_obs on random observations and record the fail rate.

    This function will continue to run until a keyboard interupt is given (Ctrl-c).
    """
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    num_failures = 0
    num_trials = 0
    latest_obs_that_fails = None
    canon = Canonicalizer()
    try:
        while(True):
            obs = obs_space.sample()
            if not _passes_test(obs, canon):
                num_failures += 1
                if latest_obs_that_fails is None:
                    print('First observation failed after {} trials. Observation:'.format(num_trials + 1))
                    print(repr(obs))
                latest_obs_that_fails = obs
            num_trials += 1
    except KeyboardInterrupt:
        print('\nPercentage of observation space symmetry sets canonicalized out of {}:'.format(num_trials))
        print('{:0.5f}%'.format((1 - (num_failures / num_trials)) * 100))

def _evaluate_execution_time(num_trials=10000):
    """ Calculate the average time taken to run canon_obs.
    """
    print('Total and average execution time for {} executions:'.format(num_trials))
    SETUP = ("from __main__ import Canonicalizer\n"
             "from bs_gym_env import BattleshipEnv\n"
             "canon = Canonicalizer()\n"
             "env = BattleshipEnv()\n"
             "obs = env.observation_space.sample()\n"
             "action = env.action_space.sample()\n"
             "_, c_tuple = canon.canon_obs(obs, True)\n")
    C_STMT = "canon.canon_obs(obs)"
    UC_STMT = "canon.uncanon_action(action, c_tuple)"
    for stmt, name in zip([C_STMT, UC_STMT], ['canonicalize', 'uncanonicalize']):
        total_time = timeit.timeit(stmt, SETUP, number=num_trials)
        avg_time = total_time / num_trials
        print('{:14}: Total: {:.2f}sec Average: {:.7f}sec'.format(name, total_time, avg_time))

if __name__ == '__main__':
    _evaluate_pass_rate()
    #_evaluate_execution_time()
    #_evaluate_uncanon_action()

    # Latest Results
    '''
    Total and average execution time for 10000 executions:
    canonicalize  : Total: 3.20sec Average: 0.0003199sec
    uncanonicalize: Total: 0.06sec Average: 0.0000065sec
    '''