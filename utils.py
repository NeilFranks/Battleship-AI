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
    adj_squares = get_adjacent_squares(row, col, rows, cols)
    valid_adj = np.array([(r, c) for r,c in adj_squares if observation[r,c] == UNKNOWN], dtype=np.int32)
    assert len(valid_adj) > 0, 'Error: no valid actions adjacent to row:{} col:{}'.format(row, col)
    return random.choice(valid_adj)

def get_adjacent_squares(row, col, rows, cols):
    adj_squares = [(r,c) for r,c in [(row,col+1), (row,col-1), (row+1,col), (row-1,col)]]
    return [(r, c) for r,c in adj_squares if r >= 0 and r < rows and c >= 0 and c < cols]

def possible_hit_count(observation, ships, row, col):
    """Counts the number of ways one of the remaining ships could fit in the square.
    """
    rows, cols = observation.shape
    possible_hits = 0
    for ship_len, ship_count in ships.items():
        count = 0
        for offset in range(1, ship_len+1):
            #check if horizontally in bounds
            left = col - ship_len + offset
            right = col + offset - 1
            if left >= 0 and right < cols:
                # check location is valid ship location (no misses or sunk ships)
                possible_ship = observation[row, left:right+1]
                if MISS not in possible_ship and SUNK not in possible_ship:
                    count += 1
            #check if vertically in bounds
            up = row - ship_len + offset
            down = row + offset - 1
            if up >= 0 and down < rows:
                possible_ship = observation[up:down+1, col]
                if MISS not in possible_ship and SUNK not in possible_ship:
                    count += 1
        count *= ship_count
        possible_hits += count
    return possible_hits

def horizontal_spaces(observation, values=[UNKNOWN, HIT]):
    '''
    given an observation, return the horizontal spaces which a boat could lie in
    '''
    hor_spaces = {}
    for i in range(len(observation)):
        begin_coord = None  # the first coord the ship covers
        end_coord = None  # the last coord the ship covers
        for j in range(len(observation[i])):
            cell = observation[i][j]
            if not begin_coord:
                if cell in values:
                    begin_coord = (i, j)
            if begin_coord and not end_coord:
                if cell not in values:
                    end_coord = (i, j-1)
                elif j == len(observation[i])-1:
                    end_coord = (i, j)
            if end_coord:
                if end_coord[1]-begin_coord[1]+1 not in hor_spaces:
                    hor_spaces[end_coord[1]-begin_coord[1]+1] = []
                hor_spaces[end_coord[1]-begin_coord[1]+1].append((begin_coord, end_coord))
                
                # reset coords
                begin_coord = None
                end_coord = None
    
    return hor_spaces       

def vertical_spaces(observation, values=[UNKNOWN, HIT]):
    '''
    given an observation, return the vertical spaces which a boat could lie in
    '''
    vert_spaces = {}
    for j in range(len(observation[0])):
        begin_coord = None  # the first coord the ship covers
        end_coord = None  # the last coord the ship covers
        for i in range(len(observation)):
            cell = observation[i][j]
            if not begin_coord:
                if cell in values:
                    begin_coord = (i, j)
            if begin_coord and not end_coord:
                if cell not in values:
                    end_coord = (i-1, j)
                elif i == len(observation)-1:
                    end_coord = (i, j)
            if end_coord:
                if end_coord[0]-begin_coord[0]+1 not in vert_spaces:
                    vert_spaces[end_coord[0]-begin_coord[0]+1] = []
                vert_spaces[end_coord[0]-begin_coord[0]+1].append((begin_coord, end_coord))
                
                # reset coords
                begin_coord = None
                end_coord = None
    
    return vert_spaces     

def find_placement_to_explain_horizontal_hits(ship_to_place, observation, begin_HITs, end_HITs):
    # We need to cover all HITS, but may have to overlap cells which are UNKNOWN if there is overhang
    i = begin_HITs[0]

    # Will ship overhang? 
    if ship_to_place > end_HITs[1]+1-begin_HITs[1]:
        # pick among valid placements
        valid_begin_columns = []

        for j in range(end_HITs[1]+1-ship_to_place, begin_HITs[1]+1):
            if j >= 0 and j+ship_to_place <= len(observation[i]):
                if all(val in [UNKNOWN, HIT] for val in observation[i, j:j+ship_to_place]):
                    valid_begin_columns.append(j)
        
        begin_placement = (i, random.choice(valid_begin_columns))
        end_placement = (i, begin_placement[1]-1+ship_to_place)
    else:
        # The ship can be placed right on the HITS with no issue
        begin_placement = begin_HITs
        end_placement = end_HITs
        
    return begin_placement, end_placement

def find_placement_to_explain_vertical_hits(ship_to_place, observation, begin_HITs, end_HITs):
    # We need to cover all HITS, but may have to overlap cells which are UNKNOWN if there is overhang
    j = begin_HITs[1]

    # Will ship overhang? 
    if ship_to_place > end_HITs[0]+1-begin_HITs[0]:
        # pick among valid placements
        valid_begin_rows = []

        for i in range(end_HITs[0]+1-ship_to_place, begin_HITs[0]+1):
            if i >= 0 and i+ship_to_place <= len(observation):
                if all(val in [UNKNOWN, HIT] for val in observation[i:i+ship_to_place, j]):
                    valid_begin_rows.append(i)
        
        begin_placement = (random.choice(valid_begin_rows), j)
        end_placement = (begin_placement[0]-1+ship_to_place, j)
    else:
        # The ship can be placed right on the HITS with no issue
        begin_placement = begin_HITs
        end_placement = end_HITs
        
    return begin_placement, end_placement

def generate_particle(observation, ships: dict, max_attempts=100):
    '''
    observation: What the agent knows about the state
    ships: a dict; key is the length of a ship, the value is how many ships there are of that length
    max_attempts: how many times the algorithm is allowed to fail before deciding there are no valid particles
                TODO: Probably could find a mathematical solution to determine if no valid particle exists.
    '''
    complete = False
    attempt = 0
    while not complete and attempt < max_attempts:
        attempt += 1

        temp_ships = ships.copy()
        temp_observation = observation.copy()
        valid_state = np.zeros(temp_observation.shape, dtype=np.int32)
        
        failed_attempt = False
        while temp_ships and not failed_attempt:
            # we shall pick a random ship to place
            ship_to_place = random.choice(list(temp_ships))
            
            try:
                # prioritize covering HITs
                hor_HIT_spaces = horizontal_spaces(temp_observation, values=[HIT])
                vert_HIT_spaces = vertical_spaces(temp_observation, values=[HIT])
                if list(hor_HIT_spaces):
                    if list(vert_HIT_spaces):
                        if random.choice(['H', 'V']) == 'H':
                            begin_coord, end_coord = random.choice(
                                [coord_tuple 
                                    for k in list(hor_HIT_spaces.keys()) 
                                    for coord_tuple in hor_HIT_spaces[k] 
                                    if k < ship_to_place
                                    ])
                            
                            begin_coord, _ = find_placement_to_explain_horizontal_hits(ship_to_place, temp_observation, begin_coord, end_coord)

                            # now place the ship
                            i = begin_coord[0]
                            start_of_ship = begin_coord[1]
                            for j in range(start_of_ship, start_of_ship+ship_to_place):
                                # update valid state
                                valid_state[i, j] = ship_to_place
                                
                                # update temp_observation so we don't try to place a ship here again
                                temp_observation[i, j] = SUNK
                            
                            # update dict of remaining ships
                            temp_ships[ship_to_place] -= 1
                            if temp_ships[ship_to_place] <= 0:
                                temp_ships.pop(ship_to_place)
                                
                        else:
                            begin_coord, end_coord = random.choice(
                                [coord_tuple 
                                    for k in list(vert_HIT_spaces.keys()) 
                                    for coord_tuple in vert_HIT_spaces[k] 
                                    if k < ship_to_place
                                    ])

                            begin_coord, _ = find_placement_to_explain_vertical_hits(ship_to_place, temp_observation, begin_coord, end_coord)
                            
                            # now place the ship
                            j = begin_coord[1]
                            start_of_ship = begin_coord[0]
                            for i in range(start_of_ship, start_of_ship+ship_to_place):
                                # update valid state
                                valid_state[i, j] = ship_to_place
                                
                                # update temp_observation so we don't try to place a ship here again
                                temp_observation[i, j] = SUNK
                            
                            # update dict of remaining ships
                            temp_ships[ship_to_place] -= 1
                            if temp_ships[ship_to_place] <= 0:
                                temp_ships.pop(ship_to_place)

                    else:
                        # Can't do vertical, so go horizontal
                        begin_coord, end_coord = random.choice(
                            [coord_tuple 
                                for k in list(hor_HIT_spaces.keys()) 
                                for coord_tuple in hor_HIT_spaces[k] 
                                if k <= ship_to_place
                                ])
                            
                        begin_coord, _ = find_placement_to_explain_horizontal_hits(ship_to_place, temp_observation, begin_coord, end_coord)
                    
                        # now place the ship
                        i = begin_coord[0]
                        start_of_ship = begin_coord[1]
                        for j in range(start_of_ship, start_of_ship+ship_to_place):
                            # update valid state
                            valid_state[i, j] = ship_to_place
                            
                            # update temp_observation so we don't try to place a ship here again
                            temp_observation[i, j] = SUNK
                        
                        # update dict of remaining ships
                        temp_ships[ship_to_place] -= 1
                        if temp_ships[ship_to_place] <= 0:
                            temp_ships.pop(ship_to_place)

                elif list(vert_HIT_spaces):
                    # Can't do horizontal, so go vertical
                    begin_coord, end_coord = random.choice(
                        [coord_tuple 
                            for k in list(vert_HIT_spaces.keys()) 
                            for coord_tuple in vert_HIT_spaces[k] 
                            if k <= ship_to_place
                            ])
                    begin_coord, _ = find_placement_to_explain_vertical_hits(ship_to_place, temp_observation, begin_coord, end_coord)
                    
                    # now place the ship
                    j = begin_coord[1]
                    start_of_ship = begin_coord[0]
                    for i in range(start_of_ship, start_of_ship+ship_to_place):
                        # update valid state
                        valid_state[i, j] = ship_to_place
                        
                        # update temp_observation so we don't try to place a ship here again
                        temp_observation[i, j] = SUNK
                    
                    # update dict of remaining ships
                    temp_ships[ship_to_place] -= 1
                    if temp_ships[ship_to_place] <= 0:
                        temp_ships.pop(ship_to_place)

                else:
                    hor_spaces = horizontal_spaces(temp_observation, values=[UNKNOWN])
                    vert_spaces = vertical_spaces(temp_observation, values=[UNKNOWN])
                    if random.choice(['H', 'V']) == 'H':
                        begin_coord, end_coord = random.choice(
                            [coord_tuple 
                                for k in list(hor_spaces.keys()) 
                                for coord_tuple in hor_spaces[k] 
                                if k >= ship_to_place
                                ])
                        start_of_ship = random.choice([row for row in range(begin_coord[1], end_coord[1]+1-ship_to_place+1)])

                        # now place the ship
                        i = begin_coord[0]
                        for j in range(start_of_ship, start_of_ship+ship_to_place):
                            # update valid state
                            valid_state[i, j] = ship_to_place
                            
                            # update temp_observation so we don't try to place a ship here again
                            temp_observation[i, j] = SUNK
                        
                        # update dict of remaining ships
                        temp_ships[ship_to_place] -= 1
                        if temp_ships[ship_to_place] <= 0:
                            temp_ships.pop(ship_to_place)
                    else:
                        # find a vertical 
                        begin_coord, end_coord = random.choice(
                            [coord_tuple 
                                for k in list(vert_spaces.keys()) 
                                for coord_tuple in vert_spaces[k] 
                                if k >= ship_to_place
                                ])

                        # find a row where the ship can start
                        start_of_ship = random.choice([row for row in range(begin_coord[0], end_coord[0]+1-ship_to_place+1)])

                        # now place the ship
                        j = begin_coord[1]
                        for i in range(start_of_ship, start_of_ship+ship_to_place):
                            # update valid state
                            valid_state[i, j] = ship_to_place
                            
                            # update temp_observation so we don't try to place a ship here again
                            temp_observation[i, j] = SUNK
                        
                        # update dict of remaining ships
                        temp_ships[ship_to_place] -= 1
                        if temp_ships[ship_to_place] <= 0:
                            temp_ships.pop(ship_to_place)

            except Exception as e:
                # will have raised an exception if we tried to choose from empty array (ship doesnt fit anywhere)
                failed_attempt = True

            if not temp_ships:
                complete = True
        
    return valid_state


def show_ships(state):
    rows = len(state)
    cols = len(state[0])

    max_digits = len(str(state.max()))

    print('\nShip placement:')
    line = '    ' + ((' '*max_digits)+' ').join('{:2}'.format(k) for k in range(cols))
    print(line)
    horizontal_line = '-' * ((3+max_digits) * cols - 1)
    print('   +' + horizontal_line + '+')
    for i in range(rows):
        line = '{:2} |'.format(i)
        for j in range(cols):
            v = str(state[i][j])
            line += ' ' + ' '*(max_digits-len(v)) + v + ' |'
        print(line)
        if i < rows -1:
            print('   |' + horizontal_line + '|')
    print('   +' + horizontal_line + '+')


if __name__ == '__main__':
    obs = np.zeros((10,10), dtype=np.int32)

    ships = DEFAULT_SHIPS
    print(possible_hit_count(obs, ships, 5, 5))
