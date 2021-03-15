import numpy as np

import bs_gym_env

def quadrant_sums(matrix):
    assert matrix.shape[0] == matrix.shape[1], 'matrix must be square'
    q_size = matrix.shape[0] // 2
    q_sums = np.zeros(4)
    q_sums[0] = matrix[:q_size, :q_size].sum() #top left
    q_sums[1] = matrix[:q_size, q_size:].sum() #top right
    q_sums[2] = matrix[q_size:, q_size:].sum() #bottom right
    q_sums[3] = matrix[q_size:, :q_size].sum() #bottom left
    return q_sums

def canonicalize_observation(obs):
    """ transform observation into its canonical form. Meaning, that if you
    pass in an observation and a rotation or flip of
    """
    # index in q_sums corresponds to number of 90 degree counter clockwise
    # rotations needed to move the quadrant with the highest sum to the top left
    q_sums = quadrant_sums(obs)
    max_q_idx = q_sums.argmax()
    # flip vertically if q_sum after max_q if less than q_sum before it (we want)
    # the greater of the two quadrants adjacent to the max to be in the upper right
    if q_sums[(max_q_idx + 1) % 4] < q_sums[max_q_idx - 1]:
        obs = np.flip(obs, axis=0)
        q_sums = np.flip(q_sums)    # q_sums order gets reversed
        max_q_idx = q_sums.argmax() # update index of max q_sum
    # rotate obs counter clockwise max_q_idx times
    obs = np.rot90(obs, k=max_q_idx)
    return obs

def test_canonicalize_observation(num_tests=1000):
    """ verify that canonicalize_observation maps all 8 transformations of a
    given observation to the same observation.
    """
    env = bs_gym_env.BattleshipEnv()
    obs_space = env.observation_space
    for test_no in range(num_tests):
        obs = obs_space.sample()
        canon_obs = canonicalize_observation(obs)
        try:
            assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(obs, k=1)))
            assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(obs, k=2)))
            assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(obs, k=3)))
            mirror_obs = np.flip(obs, axis=0)
            assert np.array_equal(canon_obs, canonicalize_observation(mirror_obs))
            assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(mirror_obs, k=1)))
            assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(mirror_obs, k=2)))
            assert np.array_equal(canon_obs, canonicalize_observation(np.rot90(mirror_obs, k=3)))
        except:
            print('failed on test #{}'.format(test_no))
            print(repr(obs))
            raise
    print('passed {} trials'.format(num_tests))


if __name__ == '__main__':
    test_canonicalize_observation()
    
    #below is a test case which fails because 3 quadrants all have the same sum
    # how should we handle such cases?

    # we need a better way to identify quadrants
    
    obs = np.array([[1, 1, 2, 0, 1, 1, 3, 3, 0, 2],
                    [0, 0, 1, 1, 3, 2, 2, 1, 2, 0],
                    [2, 1, 2, 3, 2, 0, 0, 1, 2, 0],
                    [1, 1, 0, 1, 2, 1, 0, 1, 0, 0],
                    [3, 2, 0, 2, 3, 3, 2, 3, 2, 2],
                    [2, 0, 3, 2, 1, 0, 0, 3, 0, 3],
                    [0, 3, 1, 2, 0, 3, 2, 3, 1, 2],
                    [2, 1, 3, 0, 1, 0, 3, 1, 0, 1],
                    [2, 1, 1, 3, 0, 2, 1, 0, 2, 0],
                    [1, 0, 2, 1, 3, 1, 3, 1, 1, 2]], dtype=np.int32)
    print(quadrant_sums(obs))
    print(quadrant_sums(canonicalize_observation(obs)))
