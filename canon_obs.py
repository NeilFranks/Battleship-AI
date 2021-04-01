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

def obs_to_int(obs):
    """stretch obs into vector, evaluate as base 4 number
    """
    obs_vec = obs.flatten()
    return np.dot(obs_vec, QUARTERNARY_POWERS)

def canon_obs(obs):
    """ Supposed to return the canonical observation when fed any observation in
    the symmetry set.
    """
    mirror_obs = np.flip(obs, axis=0)
    symmetry_set = [obs, mirror_obs]
    for num_rot in range(1,4): # rotate 90 3 times
        symmetry_set.append(np.rot90(obs, k=num_rot))
        symmetry_set.append(np.rot90(mirror_obs, k=num_rot))
    obs_scores = []
    for obs_trans in symmetry_set:
        obs_scores.append((obs_to_int(obs_trans), obs_trans))
    return max(obs_scores, key=lambda x: x[0])[1]

def passes_test(obs):
    """ Verify whether canon_obs can correctly canonicalize the given obervation.

    Meaning, will canon_obs return the same observation when fed all eight rotations
    and flipped rotations of the given observation? If so, returns True, else returns
    False.
    """
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

def _evaluate_pass_rate():
    """ Continuously test canon_obs on random observations and record the fail rate.

    This function will continue to run until a keyboard interupt is given (Ctrl-c).
    """
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    num_failures = 0
    num_trials = 0
    latest_obs_that_fails = None
    try:
        while(True):
            obs = obs_space.sample()
            if not passes_test(obs):
                num_failures += 1
                if latest_obs_that_fails is not None:
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
    SETUP = ("from __main__ import canon_obs\n"
             "from __main__ import canon_obs\n"
             "from bs_gym_env import BattleshipEnv\n"
             "obs = BattleshipEnv().observation_space.sample()\n")
    STMT = "canon_obs(obs)"
    total_time = timeit.timeit(STMT, SETUP, number=num_trials)
    avg_time = total_time / num_trials
    print('Total: {:.2f}sec Average: {:.7f}sec'.format(total_time, avg_time))


if __name__ == '__main__':
    #_evaluate_pass_rate()
    _evaluate_execution_time()

    # Latest Results
    '''
    Percentage of observation space symmetry sets canonicalized out of 7866807:
    84.63713% quad_sum
    98.03271% quad_sum_var
    99.97064% quad_original_kernel
    99.99917% quad_prime_kernel

    Total and average execution time for 20000 executions:
    name | total | avg
    quad_sum | 0.96sec | 0.00005sec
    quad_sum_var | 3.51sec | 0.00018sec
    quad_original_kernel | 3.85sec | 0.00019sec
    quad_prime_kernel | 3.72sec | 0.00019sec

    new method:

    Percentage of observation space symmetry sets canonicalized out of 4415288:
    99.99916% canon_obs
    100.00000% canon_obs2

    Total and average execution time for 10000 executions:
    Total: 1.71sec Average: 0.0001714sec
    '''