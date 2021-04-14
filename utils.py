from collections import namedtuple
import random
import numpy as np

DEFAULT_GRID_SIZE = 10
DEFAULT_SHIPS = {5: 1, 4: 1, 3: 2, 2: 1}

# observation encoding
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3

ENCODING = {UNKNOWN: ' ', MISS: 'o', HIT: 'x', SUNK: '#'}

ship_details = namedtuple(
    'ship_details', ['ship_length', 'coords']
)
particle = namedtuple('particle', ['board', 'ship_details_list'])


def pick_random_valid_adjacent_action(observation, row, col):
    """Returns a 2-D numpy array where each for contains the coords of a square
    adjacent to observation[row, col] and whose value is UNKNOWN.
    """
    rows, cols = observation.shape
    adj_squares = get_adjacent_squares(row, col, rows, cols)
    valid_adj = np.array(
        [(r, c) for r, c in adj_squares if observation[r, c] == UNKNOWN], dtype=np.int32)
    assert len(
        valid_adj) > 0, 'Error: no valid actions adjacent to row:{} col:{}'.format(row, col)
    return random.choice(valid_adj)


def get_adjacent_squares(row, col, rows, cols):
    adj_squares = [(r, c) for r, c in [(row, col+1),
                                       (row, col-1), (row+1, col), (row-1, col)]]
    return [(r, c) for r, c in adj_squares if r >= 0 and r < rows and c >= 0 and c < cols]


def possible_hit_count(observation, ships, row, col):
    """Counts the number of ways one of the remaining ships could fit in the square.
    """
    rows, cols = observation.shape
    possible_hits = 0
    for ship_len, ship_count in ships.items():
        count = 0
        for offset in range(1, ship_len+1):
            # check if horizontally in bounds
            left = col - ship_len + offset
            right = col + offset - 1
            if left >= 0 and right < cols:
                # check location is valid ship location (no misses or sunk ships)
                possible_ship = observation[row, left:right+1]
                if MISS not in possible_ship and SUNK not in possible_ship:
                    count += 1
            # check if vertically in bounds
            up = row - ship_len + offset
            down = row + offset - 1
            if up >= 0 and down < rows:
                possible_ship = observation[up:down+1, col]
                if MISS not in possible_ship and SUNK not in possible_ship:
                    count += 1
        count *= ship_count
        possible_hits += count
    return possible_hits


def find_UNKNOWN_sequences_where_ship_can_lie(obs, ship_length):
    rows = len(obs)
    cols = len(obs[0])

    sequences = []

    # find horizontal sequences
    for i in range(rows):
        for j in range(cols-ship_length+1):
            if all(val in [UNKNOWN] for val in obs[i, j:j+ship_length]):
                sequences.append([(i, cell_j)
                                 for cell_j in range(j, j+ship_length)])

    # find vertical sequences
    for j in range(cols):
        for i in range(rows-ship_length+1):
            if all(val in [UNKNOWN] for val in obs[i:i+ship_length, j]):
                sequences.append([(cell_i, j)
                                 for cell_i in range(i, i+ship_length)])

    return sequences


def find_sequences_where_ship_covers_hit(obs, ship_length, hit_coords):
    row = hit_coords[0]
    col = hit_coords[1]
    rows = len(obs)
    cols = len(obs[0])

    sequences = []

    # find horizontal sequences
    i = row
    for j in range(col+1-ship_length, col+1):
        if j >= 0 and j+ship_length <= cols:
            if all(val in [UNKNOWN, HIT] for val in obs[i, j:j+ship_length]):
                sequences.append([(i, cell_j)
                                 for cell_j in range(j, j+ship_length)])

    # find vertical sequences
    j = col
    for i in range(row+1-ship_length, row+1):
        if i >= 0 and i+ship_length <= rows:
            if all(val in [UNKNOWN, HIT] for val in obs[i:i+ship_length, j]):
                sequences.append([(cell_i, j)
                                 for cell_i in range(i, i+ship_length)])

    return sequences


def make_board_from_ship_details(shape, ship_details_list):
    board = np.zeros(shape, dtype=np.int32)
    for ship_details in ship_details_list:
        for coord in ship_details.coords:
            board[coord] = SUNK
    return board


def generate_particle_from_scratch_recursive(obs, ships: dict, depth=0):
    ship_details_list = []

    """
    SETUP
    """
    # put ships in random order
    ship_list = []
    for ship in ships.keys():
        for _ in range(ships[ship]):
            ship_list.append(ship)
    random.shuffle(ship_list)

    # find all hits; put them in random order
    hit_list = []
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            if obs[i][j] == HIT:
                hit_list.append((i, j))
    random.shuffle(hit_list)

    """
    Place a ship and recurse until either it works, or you determine it's impossible
    """
    ship_index = 0
    hit_index = 0
    sub_particle = None  # This is a particle which will be generated by a recursive call
    temp_obs = obs.copy()  # keep a temp observation
    while type(sub_particle) == type(None) or HIT in temp_obs:
        # If we've checked every ship and still no luck, this must be impossible; there is no particle to return
        if ship_index >= len(ship_list):
            return None

        # select a ship
        ship_to_place = ship_list[ship_index]

        if hit_list:
            # select a hit
            hit_to_cover = hit_list[hit_index]

            # find all sequences of cells which the ship could lie on which would cover the hit
            sequences = find_sequences_where_ship_covers_hit(
                obs=obs, ship_length=ship_to_place, hit_coords=hit_to_cover)
        else:
            # find all sequences of cells which the ship could lie on which are all UNKNOWN cells
            sequences = find_UNKNOWN_sequences_where_ship_can_lie(
                obs=obs, ship_length=ship_to_place)

        # only continue if there is at least one sequence
        if sequences:
            random.shuffle(sequences)

            sequence_index = 0
            while sequence_index < len(sequences) and type(sub_particle) == type(None):

                # keep track of your particle
                particle_board = np.zeros(obs.shape, dtype=np.int32)
                temp_obs = obs.copy()  # keep a temp observation
                temp_ships = ships.copy()  # keep temp copy of ships

                # place the ship on a selected sequence and see if it works
                selected_sequence = sequences[sequence_index]
                ship_coords = []
                for cell in selected_sequence:
                    # update our observation
                    temp_obs[cell] = SUNK

                    # update our particle_board
                    particle_board[cell] = SUNK

                    # update ship coords
                    ship_coords.append(cell)

                # update ship details
                ship_details_list.append(ship_details(
                    ship_length=ship_to_place, coords=ship_coords))

                # we have placed this ship!
                temp_ships[ship_to_place] -= 1

                if all(val == 0 for val in temp_ships.values()) and HIT not in temp_obs:
                    # No ships left, you're good!
                    sub_particle = particle(
                        board=np.zeros(obs.shape, dtype=np.int32),
                        ship_details_list=None
                    )
                else:
                    # recurse; place the remaining ships!
                    sub_particle = generate_particle_from_scratch_recursive(
                        temp_obs, temp_ships, depth=depth+1)

                # If the subparticle is not None, congrats, it worked!
                sequence_index += 1

        # If we havent solved it, we'll need to check the next hit
        hit_index += 1

        # If true, we checked every possible hit for this boat; This boat must not lie on a hit. Try the next boat!
        if hit_index >= len(hit_list):
            hit_index = 0
            ship_index += 1

        if type(sub_particle) != type(None):
            # update temp_obs to reflect the other ships
            for i in range(len(sub_particle.board)):
                for j in range(len(sub_particle.board[i])):
                    if sub_particle.board[i][j] != 0:
                        temp_obs[i][j] = 3

            particle_board = particle_board + sub_particle.board
            if sub_particle.ship_details_list:
                ship_details_list.extend(sub_particle.ship_details_list)

    return particle(board=particle_board, ship_details_list=ship_details_list)


def generate_particle_from_scratch(obs, ships: dict):
    '''
    observation: What the agent knows about the state
    ships: a dict; key is the length of a ship, the value is how many ships there are of that length
    '''
    # board_with_only_SUNKS = np.zeros(obs.shape, dtype=np.int32)
    # for i in range(len(obs)):
    #     for j in range(len(obs[i])):
    #         if obs[i][j] == SUNK:
    #             board_with_only_SUNKS[i][j] = SUNK

    # return board_with_only_SUNKS+generate_particle_from_scratch_recursive(obs, ships)
    return generate_particle_from_scratch_recursive(obs, ships)


def generate_particles_from_scratch(k, obs, ships):
    particles = []
    while len(particles) < k:
        particles.append(generate_particle_from_scratch(obs, ships))
    return particles


def generate_particle_from_valid_particle(valid_particle, obs):
    """
    Pick a random ship. If it covers hits, pick another ship which it could swap with.
    If it doesn't cover hits, move it to an UNKNOWN sequence
    """
    ships_which_dont_cover_hits = []
    for ship in valid_particle.ship_details_list:
        if HIT not in ship.coords and SUNK not in ship.coords:
            ships_which_dont_cover_hits.append(ship)
    random.shuffle(ships_which_dont_cover_hits)

    for ship_to_move in ships_which_dont_cover_hits:
        # just move this ship to an unoccupied UNKNOWN sequence and that's it!
        ships_to_stay_put = valid_particle.ship_details_list[:]
        ships_to_stay_put.remove(ship_to_move)

        occupied_coords = [
            c for ship in ships_to_stay_put for c in ship.coords
        ]

        UNKNOWN_sequences = find_UNKNOWN_sequences_where_ship_can_lie(
            obs, ship_to_move.ship_length)
        unoccupied_sequences = []
        for sequence in UNKNOWN_sequences:
            occupied = False
            for coord in sequence:
                if coord in occupied_coords:
                    occupied = True

            if not occupied:
                unoccupied_sequences.append(sequence)

        valid_coords = random.choice(unoccupied_sequences)
        ships_to_stay_put.append(
            ship_details(ship_length=ship_to_move.ship_length,
                         coords=valid_coords)
        )

        return particle(board=make_board_from_ship_details(obs.shape, ships_to_stay_put), ship_details_list=ships_to_stay_put)

    return None


def generate_particles_from_valid_particles(valid_particles, k, obs):
    length_of_valid = len(valid_particles)
    particles = []
    i = 0
    while len(particles) < k:
        particles.append(
            generate_particle_from_valid_particle(valid_particles[i], obs))
        i = (i+1) % length_of_valid
    return particles


def generate_particle_from_invalid_particle(invalid_particle, obs, previous_action):
    """
    Will need to find which ship is invalid. Should only be one ship,
    but we'll check every ship just in case implementation changes.

    Move any invalid ship to a valid location. If there is no such location, return None
    """
    conflict = (previous_action[0], previous_action[1])
    if obs[conflict] == MISS:
        invalid_ships = []
        valid_ships = []
        for ship in invalid_particle.ship_details_list:
            if conflict in ship.coords:
                invalid_ships.append(ship)
            else:
                valid_ships.append(ship)

        valid_ship_coords = [
            c for ship in valid_ships for c in ship.coords
        ]

        for ship in invalid_ships:
            UNKNOWN_sequences = find_UNKNOWN_sequences_where_ship_can_lie(
                obs, ship.ship_length)
            unoccupied_sequences = []
            for sequence in UNKNOWN_sequences:
                occupied = False
                for coord in sequence:
                    if coord in valid_ship_coords:
                        occupied = True

                if not occupied:
                    unoccupied_sequences.append(sequence)

            valid_coords = random.choice(unoccupied_sequences)
            valid_ship_coords.extend(valid_coords)
            valid_ships.append(
                ship_details(ship_length=ship.ship_length,
                             coords=valid_coords)
            )
    elif obs[conflict] == HIT:
        # find a ship which will cover the hit
        ships = invalid_particle.ship_details_list
        random.shuffle(ships)
        hit_covered = False
        i = 0
        while not hit_covered:
            if i >= len(ships):
                # we checked every ship; it is impossible to cover this hit.
                return None
            ship_to_place = ships[i]
            sequences = find_sequences_where_ship_covers_hit(
                obs, ship_to_place.ship_length, conflict)
            if sequences:
                selected_sequence = random.choice(sequences)
                hit_covered = True
            i += 1

        # Place this ship to cover the hit
        ships.pop(i-1)
        valid_ships = [ship_details(ship_length=ship_to_place.ship_length,
                                    coords=selected_sequence)]
        valid_ship_coords = [selected_sequence]

        # place the rest of the ships
        for ship in ships:
            UNKNOWN_sequences = find_UNKNOWN_sequences_where_ship_can_lie(
                obs, ship.ship_length)
            unoccupied_sequences = []
            for sequence in UNKNOWN_sequences:
                occupied = False
                for coord in sequence:
                    if coord in valid_ship_coords:
                        occupied = True

                if not occupied:
                    unoccupied_sequences.append(sequence)

            valid_coords = random.choice(unoccupied_sequences)
            valid_ship_coords.extend(valid_coords)
            valid_ships.append(
                ship_details(ship_length=ship.ship_length,
                             coords=valid_coords)
            )
    elif obs[conflict] == SUNK:
        pass
    else:
        raise("this is not supposed to happen; bad")

    return particle(board=make_board_from_ship_details(obs.shape, valid_ships), ship_details_list=valid_ships)


def generate_particles_from_invalid_particles(invalid_particles, k, obs, previous_action):
    length_of_inv = len(invalid_particles)
    particles = []
    i = 0
    while len(particles) < k:
        particles.append(
            generate_particle_from_invalid_particle(invalid_particles[i], obs, previous_action))
        i = (i+1) % length_of_inv
    return particles


def check_particles_are_valid(particles, obs, previous_action):
    """
    Since all these particles were valid PRIOR to the previous action,
    we only need to check if they're still valid WITH the previous action
    """
    if type(previous_action) == type(None):
        # first turn; all particles are valid
        return particles, []

    coord = (previous_action[0], previous_action[1])
    valid_particles = []
    invalid_particles = []
    for particle in particles:
        board = particle.board
        if board[coord] > 0 and obs[coord] == MISS:
            # Particle has a boat here, but that's impossible
            invalid_particles.append(particle)
        elif board[coord] == 0 and obs[coord] in [HIT, SUNK]:
            # There is a boat here, but particle doesn't have one here!
            invalid_particles.append(particle)
        else:
            valid_particles.append(particle)

    return valid_particles, invalid_particles


def best_action_from_particles(obs, particles):
    shape = particles[0].board.shape
    master_board = np.zeros(shape, dtype=np.int32)
    for particle in particles:
        master_board += particle.board

    # remember, you can't shoot where you've already shot
    mask = [cell == UNKNOWN for cell in obs]
    master_board *= mask

    flat_argmax = np.argmax(master_board)
    return np.array((flat_argmax//shape[1], flat_argmax % shape[1]), dtype=np.int32)


def show_ships(state):
    rows = len(state)
    cols = len(state[0])

    max_digits = len(str(state.max()))

    print('\nShip placement:')
    line = '    ' + ((' '*max_digits) +
                     ' ').join('{:2}'.format(k) for k in range(cols))
    print(line)
    horizontal_line = '-' * ((3+max_digits) * cols - 1)
    print('   +' + horizontal_line + '+')
    for i in range(rows):
        line = '{:2} |'.format(i)
        for j in range(cols):
            v = str(state[i][j])
            line += ' ' + ' '*(max_digits-len(v)) + v + ' |'
        print(line)
        if i < rows - 1:
            print('   |' + horizontal_line + '|')
    print('   +' + horizontal_line + '+')


if __name__ == '__main__':
    obs = np.zeros((10, 10), dtype=np.int32)

    ships = DEFAULT_SHIPS
    print(possible_hit_count(obs, ships, 5, 5))
