import numpy as np
from time import time

import bs_gym_env
from utils import DEFAULT_GRID_SIZE
##
## Quadrant symmetry group identifier methods
##

def quad_sum(matrix):
    assert matrix.shape[0] == matrix.shape[1], 'matrix must be square'
    q_size = matrix.shape[0] // 2
    q_stats = np.zeros(4)
    q_stats[0] = matrix[:q_size, :q_size].sum() #top left
    q_stats[1] = matrix[:q_size, q_size:].sum() #top right
    q_stats[2] = matrix[q_size:, q_size:].sum() #bottom right
    q_stats[3] = matrix[q_size:, :q_size].sum() #bottom left
    return q_stats


def quad_sum_var(matrix):
    assert matrix.shape[0] == matrix.shape[1], 'matrix must be square'
    q_size = matrix.shape[0] // 2
    q_stats = np.zeros(4)
    q_stats[0] = matrix[:q_size, :q_size].sum() * matrix[:q_size, :q_size].var() #top left
    q_stats[1] = matrix[:q_size, q_size:].sum() * matrix[:q_size, q_size:].var() #top right
    q_stats[2] = matrix[q_size:, q_size:].sum() * matrix[q_size:, q_size:].var() #bottom right
    q_stats[3] = matrix[q_size:, :q_size].sum() * matrix[q_size:, :q_size].var() #bottom left
    return q_stats


ORIGINAL_KERKEL = np.array([[1, 2, 3, 2, 1],
                            [2, 4, 5, 4, 2],
                            [3, 5, 6, 5, 3],
                            [2, 4, 5, 4, 2],
                            [1, 2, 3, 2, 1]])

PRIME_KERNEL = np.array([[1,  7,  11, 7,  1],
                         [7,  13, 17, 13, 7],
                         [11, 17, 19, 17, 11],
                         [7,  13, 17, 13, 7],
                         [1,  7,  11, 7,  1]])

DEFAULT_KERNEL = PRIME_KERNEL

def quad_kernel(matrix, kernel_str):
    assert matrix.shape[0] == matrix.shape[1], 'matrix must be square'
    q_size = matrix.shape[0] // 2
    
    q_stats = np.zeros(4)

    if kernel_str == 'original':
        kernel = ORIGINAL_KERKEL
    elif kernel_str == 'prime':
        kernel = PRIME_KERNEL
    else:
        kernel = DEFAULT_KERNEL

    quad = np.multiply(matrix[:q_size, :q_size], kernel)
    q_stats[0] = quad.sum() * quad.var() #top left

    quad = np.multiply(matrix[:q_size, q_size:], kernel)
    q_stats[1] = quad.sum() * quad.var() #top right

    quad = np.multiply(matrix[q_size:, q_size:], kernel)
    q_stats[2] = quad.sum() * quad.var() #bottom right

    quad = np.multiply(matrix[q_size:, :q_size], kernel)
    q_stats[3] = quad.sum() * quad.var() #bottom left

    return q_stats


def quad_prime_kernel(matrix):
    return quad_kernel(matrix, kernel_str='prime')


def quad_original_kernel(matrix):
    return quad_kernel(matrix, kernel_str='original')

##
## Alternate approach: stretch obs into vector, treat as base 4 number
##      evaluate for all members of symmetry group, choose tranformation
##      which results in largest number
##

QUARTERNARY_POWERS = np.array([4**p for p in range(DEFAULT_GRID_SIZE ** 2)])

def obs_to_int(obs):
    """stretch obs into vector, treat as base 4 number
    """
    obs_vec = obs.flatten()
    return np.dot(obs_vec, QUARTERNARY_POWERS)

def canonicalize_observation2(obs):
    mirror_obs = np.flip(obs, axis=0)
    symmetry_set = [obs, mirror_obs]
    for num_rot in range(1,4): # rotate 90 3 times
        symmetry_set.append(np.rot90(obs, k=num_rot))
        symmetry_set.append(np.rot90(mirror_obs, k=num_rot))
    obs_scores = []
    for obs_trans in symmetry_set:
        obs_scores.append((obs_to_int(obs_trans), obs_trans))
    return max(obs_scores, key=lambda x: x[0])[1]
    

##
## Observation Canonicalizer function
##

def canonicalize_observation(obs, iden_method):
    """ transform observation into its canonical form. Meaning, that if you
    pass in an observation and a rotation or flip of
    """
    # index in q_stats corresponds to number of 90 degree counter clockwise
    # rotations needed to move the quadrant with the highest sum to the top left
    q_stats = iden_method(obs)
    max_q_idx = q_stats.argmax()
    # flip vertically if q_stat after max_q if less than q_stat before it (we want)
    # the greater of the two quadrants adjacent to the max to be in the upper right
    if q_stats[(max_q_idx + 1) % 4] < q_stats[max_q_idx - 1]:
        obs = np.flip(obs, axis=0)
        q_stats = np.flip(q_stats)    # q_stats order gets reversed
        max_q_idx = q_stats.argmax() # update index of max q_stat
    # rotate obs counter clockwise max_q_idx times
    obs = np.rot90(obs, k=max_q_idx)
    return obs

##
## Tests
##

def passes_test(obs, method):
    canon_obs = canonicalize_observation(obs, method)
    try:
        assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(obs, k=1), method))
        assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(obs, k=2), method))
        assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(obs, k=3), method))
        mirror_obs = np.flip(obs, axis=0)
        assert np.array_equal(canon_obs, canonicalize_observation(mirror_obs, method))
        assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(mirror_obs, k=1), method))
        assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(mirror_obs, k=2), method))
        assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(mirror_obs, k=3), method))
        return True
    except AssertionError:
        return False


def passes_test2(obs):
    canon_obs = canonicalize_observation2(obs)
    try:
        assert np.array_equal(canon_obs, canonicalize_observation2(np.rot90(obs, k=1)))
        assert np.array_equal(canon_obs, canonicalize_observation2(np.rot90(obs, k=2)))
        assert np.array_equal(canon_obs, canonicalize_observation2(np.rot90(obs, k=3)))
        mirror_obs = np.flip(obs, axis=0)
        assert np.array_equal(canon_obs, canonicalize_observation2(mirror_obs))
        assert np.array_equal(canon_obs, canonicalize_observation2(np.rot90(mirror_obs, k=1)))
        assert np.array_equal(canon_obs, canonicalize_observation2(np.rot90(mirror_obs, k=2)))
        assert np.array_equal(canon_obs, canonicalize_observation2(np.rot90(mirror_obs, k=3)))
        return True
    except AssertionError:
        return False


def find_symmetry_set_which_fails(method):
    """ verify that canonicalize_observation maps all 8 transformations of a
    given observation to the same observation.
    """
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    num_obs_tried = 0
    print('looking for symmetry set which fails with method: {}'.format(method.__name__))
    while(True):
        obs = obs_space.sample()
        num_obs_tried += 1
        if not passes_test(obs, method):
            break
    print('found after {} observations tried'.format(num_obs_tried))
    print('observation', repr(obs))
    print('quad identifier method output for obs ', method(obs))
    return obs


IDENTIFICATION_METHODS = [quad_sum, quad_sum_var, quad_original_kernel, quad_prime_kernel]

def compare_identification_method_performance(num_trials=1000):
    """ Compare identification methods by seeing what percentage of observations
    can be correctly canonicallized. aka the output of canon(obs) is the same as
    the output of canon when given any of the 8 transformations of obs.
    """
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    num_failures = dict()
    for method in IDENTIFICATION_METHODS:
        num_failures[method.__name__] = 0
    try:
        num_trials = 0
        while(True):
            obs = obs_space.sample()
            for method in IDENTIFICATION_METHODS:
                if not passes_test(obs, method):
                    num_failures[method.__name__] += 1
            num_trials += 1
    except:
        print('\nPercentage of observation space symmetry sets canonicalized out of {}:'.format(num_trials))
        for method, pass_rate in num_failures.items():
            print('{:0.5f}% {}'.format((1 - (pass_rate / num_trials)) * 100, method))


def compare_identification_method_time(num_trials=20000):
    print('Total and average execution time for {} executions:'.format(num_trials))
    print('name | total | avg')
    for method in IDENTIFICATION_METHODS:
        start = time()
        env = bs_gym_env.BattleshipEnv()
        obs = env.observation_space.sample()
        for _ in range(num_trials):
            canonicalize_observation(obs, method)
        total_time = time() - start
        avg_time = total_time / num_trials
        print('{} | {:.2f}sec | {:.5f}sec'.format(method.__name__, total_time, avg_time))


def compare_canon_functions():
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    num_failures = dict()
    num_failures[canonicalize_observation.__name__] = 0
    num_failures[canonicalize_observation2.__name__] = 0
    method = quad_prime_kernel
    try:
        num_trials = 0
        while(True):
            obs = obs_space.sample()
            if not passes_test(obs, method):
                num_failures[canonicalize_observation.__name__] += 1
            if not passes_test2(obs):
                num_failures[canonicalize_observation2.__name__] += 1
            num_trials += 1
    except KeyboardInterrupt:
        print('\nPercentage of observation space symmetry sets canonicalized out of {}:'.format(num_trials))
        for function, pass_rate in num_failures.items():
            print('{:0.5f}% {}'.format((1 - (pass_rate / num_trials)) * 100, function))


def compare_canon_function_execution_time(num_trials=10000):
    print('Total and average execution time for {} executions:'.format(num_trials))
    print('name | total | avg')
    env = bs_gym_env.BattleshipEnv()
    obs = env.observation_space.sample()
    method = quad_prime_kernel

    start = time()
    for _ in range(num_trials):
        canonicalize_observation(obs, method)
    total_time = time() - start
    avg_time = total_time / num_trials
    print('{} | {:.2f}sec | {:.5f}sec'.format(canonicalize_observation.__name__, total_time, avg_time))

    start = time()
    for _ in range(num_trials):
        canonicalize_observation2(obs)
    total_time = time() - start
    avg_time = total_time / num_trials
    print('{} | {:.2f}sec | {:.5f}sec'.format(canonicalize_observation2.__name__, total_time, avg_time))


if __name__ == '__main__':
    #test_canonicalize_observation(quadrant_sums)
    #compare_identification_method_performance()
    #compare_identification_method_time()
    #find_symmetry_set_which_fails(quad_prime_kernel)
    compare_canon_functions()
    #compare_canon_function_execution_time()

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
    99.99916% canonicalize_observation
    100.00000% canonicalize_observation2

    Total and average execution time for 10000 executions:
    name | total | avg
    canonicalize_observation | 1.83sec | 0.00018sec
    canonicalize_observation2 | 2.50sec | 0.00025sec
    '''