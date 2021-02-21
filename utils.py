from collections import namedtuple

#observation encoding
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3

PossibleSpace = namedtuple('PossibleSpace', 'left_space, right_space, up_space, down_space')
def getPossibleSpace(board, row, col, longest_unsunk_ship_length):
    """
    Look left, right, up, and down. 
    Count the longest span of squares that a ship could feasibly lie in 
    while on the square specified by `board[row][col]`.
    """

    # look left
    obs = board.observation[row][col]
    look_col = col
    while abs(col-look_col)+1 <= longest_unsunk_ship_length and \
        (obs == HIT or obs == UNKNOWN):
        look_col -= 1
        if look_col < 0:
            obs = -1  # invalid square
        else:
            obs = board.observation[row][look_col]  # get observation from board
    left_space = abs(col-look_col)
        
    # look right
    obs = board.observation[row][col]
    look_col = col
    while abs(col-look_col)+1 <= longest_unsunk_ship_length and \
        (obs == HIT or obs == UNKNOWN):
        look_col += 1
        if look_col >= board._width:
            obs = -1  # invalid square
        else:
            obs = board.observation[row][look_col]  # get observation from board
    right_space = abs(col-look_col)

    # look up
    obs = board.observation[row][col]
    look_row = row
    while abs(row-look_row)+1 <= longest_unsunk_ship_length and \
        (obs == HIT or obs == UNKNOWN):
        look_row -= 1
        if look_row < 0:
            obs = -1  # invalid square
        else:
            obs = board.observation[look_row][col]  # get observation from board
    up_space = abs(row-look_row)

    # look down
    obs = board.observation[row][col]
    look_row = row
    while abs(row-look_row)+1 <= longest_unsunk_ship_length and \
        (obs == HIT or obs == UNKNOWN):
        look_row += 1
        if look_row >= board._height:
            obs = -1  # invalid square
        else:
            obs = board.observation[look_row][col]  # get observation from board
    down_space = abs(row-look_row)

    return PossibleSpace(left_space=left_space, right_space=right_space, up_space=up_space, down_space=down_space)

def pick_most_probable_move(board):
    """
    Iterate through each square and return the one that the most boats could lie on
    """
    longest_unsunk_ship_length = max(board.unsunk_ship_lengths_by_id.values())
    most_likely_square = None
    most_possible_hits = 0

    for row in range(board._height):
        for col in range(board._width):
            if board.observation[row][col] == UNKNOWN:
                possible_hits_for_current_square = 0

                # how far can we look in any direction without seeing a SUNK or MISS?
                possible_space = getPossibleSpace(board, row, col, longest_unsunk_ship_length)
                
                # for each unsunk ship, how many times might they be overlapping this square?
                for ship_length in board.unsunk_ship_lengths_by_id.values():
                    # Horizontal
                    if possible_space.left_space >= ship_length:
                        if possible_space.right_space >= ship_length:
                            possible_hits_for_current_square += ship_length
                        else:
                            possible_hits_for_current_square += possible_space.right_space
                    elif possible_space.right_space >= ship_length:
                        possible_hits_for_current_square += possible_space.left_space
                    else:  # neither side can fit the ship alone, but maybe it can fit across both sides?
                        hor_space = possible_space.left_space+possible_space.right_space-1
                        if hor_space >= ship_length:
                            possible_hits_for_current_square += hor_space-ship_length+1

                    # Vertical
                    if possible_space.up_space >= ship_length:
                        if possible_space.down_space >= ship_length:
                            possible_hits_for_current_square += ship_length
                        else:
                            possible_hits_for_current_square += possible_space.down_space
                    elif possible_space.down_space >= ship_length:
                        possible_hits_for_current_square += possible_space.up_space
                    else:  # neither side can fit the ship alone, but maybe it can fit across both sides?
                        vert_space = possible_space.up_space+possible_space.down_space-1
                        if vert_space >= ship_length:
                            possible_hits_for_current_square += vert_space-ship_length+1

                if possible_hits_for_current_square > most_possible_hits:
                    most_possible_hits = possible_hits_for_current_square
                    most_likely_square = (row, col)

    return most_likely_square