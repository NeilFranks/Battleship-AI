import numpy as np

import bs_gym_env

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

def test_canonicalize_observation(method, num_tests=1000):
    """ verify that canonicalize_observation maps all 8 transformations of a
    given observation to the same observation.
    """
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    for test_no in range(num_tests):
        obs = obs_space.sample()
        if not passes_test(obs, method):
            print('failed on test #{}'.format(test_no))
            print('observation', repr(obs))
            print('quad identifier method output for obs ', method(obs))
            print('quad identifier method output for canon(obs) ', method(canonicalize_observation(obs, method)))
            exit()
    print('passed {} trials'.format(num_tests))

def compare_identification_methods(num_trials=1000):
    """ Compare identification methods by seeing what percentage of observations
    can be correctly canonicallized. aka the output of canon(obs) is the same as
    the output of canon when given any of the 8 transformations of obs.
    """
    identification_methods = [quad_sum, quad_sum_var, quad_original_kernel, quad_prime_kernel]
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    num_failures = dict()
    for method in identification_methods:
        num_failures[method.__name__] = 0
    try:
        num_trials = 0
        while(True):
            obs = obs_space.sample()
            for method in identification_methods:
                if not passes_test(obs, method):
                    num_failures[method.__name__] += 1
            num_trials += 1
    except:
        print('\nPercentage of observation space symmetry sets canonicalized out of {}:'.format(num_trials))
        for method, pass_rate in num_failures.items():
            print('{:0.5f}% {}'.format((1 - (pass_rate / num_trials)) * 100, method))



if __name__ == '__main__':
    #test_canonicalize_observation(quadrant_sums)
    compare_identification_methods()
    
    