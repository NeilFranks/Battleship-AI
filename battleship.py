import game
import numpy as np
import os
import random
import itertools
import time

#observation encoding
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3


class SinglePlayerBattleship(game.Game):
    """single player version of battleship where ships are randomly placed"""
    def __init__(self, player=None, display=None, ships=None, height=10, width=10):
        """sets up the game.
        
        Parameters:
            - player (BattleshipPlayer): default is HumanBattleshipPlayer
            - display (game.Display or -1): default is BattleshipCLI, -1 means no display
            - ships (dict): a dictionary describing the type and number of
                ships to be placed on the board.
                ex. {5:1, 4:1, 3:2, 2:1} is the default and it specifies
                that there is to be 1 ship of length 5, 1 ship of length 4,
                2 ships of length 3 and 1 ship of length 2.
        """
        self.board = BattleshipBoard(height=height, width=width)
        self.player = player or HumanBattleshipPlayer()
        assert isinstance(self.player, game.Player), 'player must be instance of game.Player or subclass'
        if display == -1:
            self.display = -1
        else:
            self.display = display or BattleshipCLI()

        #assert isinstance(self.display, game.Display) or display is None, 'display must be insatnce of game.Display or subclass'
        self.ships = ships or {5:1, 4:1, 3:2, 2:1}
        assert isinstance(self.ships, dict), 'ships must be dictionary mapping ship length to number of ships of that length'

    def reset(self):
        self.board.reset()
        self.player.reset()
        self.place_ships()

    def place_ships(self):
        self.unsunk_ship_lengths_by_id = {}
        col_max = self.board._width - 1
        row_max = self.board._height - 1
        ship_id = 1
        for ship_len in self.ships:
            for _ in range(self.ships[ship_len]):
                placed = False
                while not placed:
                    row, col = random.randint(0, row_max), random.randint(0, col_max)
                    if self.board.state[row, col]: #is there already a ship here?
                        continue
                    direction = random.randint(0,3)
                    if direction == 0: #right
                        if col + ship_len > col_max + 1: #ship out of bounds
                            continue
                        if sum(self.board.state[row, col:col+ship_len]) > 0: #ship already present
                            continue
                        self.board.state[row, col:col+ship_len] = ship_id
                    elif direction == 1: #down
                        if row + ship_len > row_max + 1: #ship out of bounds
                            continue
                        if sum(self.board.state[row:row+ship_len, col]) > 0: #ship already present
                            continue
                        self.board.state[row:row+ship_len, col] = ship_id
                    elif direction == 2: #left
                        if col - ship_len + 1 < 0: #ship out of bounds
                            continue
                        if sum(self.board.state[row, col-ship_len+1:col+1]) > 0: #ship already present
                            continue
                        self.board.state[row, col-ship_len+1:col+1] = ship_id
                    else: #up
                        if row - ship_len + 1 < 0: #ship out of bounds
                            continue
                        if sum(self.board.state[row-ship_len+1:row+1, col]) > 0: #ship already present
                            continue
                        self.board.state[row-ship_len+1:row+1, col] = ship_id
                    placed = True
                    self.unsunk_ship_lengths_by_id[ship_id] = ship_len
                ship_id += 1

    def play(self):
        """lets the player play a new game and returns the number of shots required to finish"""
        self.reset()
        while not self.board.is_game_over():
            row, col = self.player.take_turn(self.board)
            sunk_id = self.board.process_shot(row, col)  # if sunk_id is 0, no ship was sunk.
            if sunk_id > 0:  # if sunk_id is a number, the ship with that id was sunk
                del self.unsunk_ship_lengths_by_id[sunk_id]
            if self.display != -1:
                self.display.update(self.board)
        return self.board.shots_fired

class HumanBattleshipPlayer(game.Player):
    def __init__(self):
        pass

    def reset(self):
        pass

class RandomBattleshipPlayer(game.Player):
    def __init__(self, delay=0):
        self.delay = delay

    def take_turn(self, board):
        time.sleep(self.delay)
        return pick_random_valid_move(board)

    def reset(self):
        pass


class HardCodedBattleshipPlayer(game.Player):
    """shoots randomly until a ship is hit, then it explores the area around it"""
    def __init__(self, delay=0):
        self.delay = delay
        self.reset()
    
    def take_turn(self, board):
        time.sleep(self.delay)
        if self.previous_shot is None: #first shot
            self.previous_shot = pick_random_valid_move(board)
            return self.previous_shot

        #remove sunk ships from hits
        bf = len(self.hits)
        sunk = []
        for row, col in self.hits:
            if board.observation[row, col] == SUNK:
                sunk.append((row, col))
        for coords in sunk:
            self.hits.remove(coords)                

        #check if previous shot was a hit
        if board.observation[self.previous_shot] == HIT:
            self.hits.append(self.previous_shot)

        #if there are no known hits, shoot randomly
        if len(self.hits) == 0:
            self.previous_shot = pick_random_valid_move(board)
            return self.previous_shot

        #look vertically for adjacent hits
        row, col = hit = self.hits[0]
        if (row > 0 and board.observation[row-1, col] == HIT) or \
           (row < board._height-1 and board.observation[row+1, col] == HIT):
            #continue search up until non-hit encountered or edge of board
            r = row
            while r >= 0 and board.observation[r, col] == HIT:
                r -= 1
            if r >= 0 and board.observation[r, col] == UNKNOWN:
                self.previous_shot = (r, col)
                return self.previous_shot
            #continue search down until non-hit encountered or edge of board
            r = row
            while r < board._height and board.observation[r, col] == HIT:
                r += 1
            if r < board._height and board.observation[r, col] == UNKNOWN:
                self.previous_shot = (r, col)
                return self.previous_shot

        #look horizontally for adjacent hits
        if (col > 0 and board.observation[row, col-1] == HIT) or \
            (col < board._width-1 and board.observation[row, col+1] == HIT):
            #continue search left until non-hit encountered or edge of board
            c = col
            while c >= 0 and board.observation[row, c] == HIT:
                c -= 1
            if c >= 0 and board.observation[row, c] == UNKNOWN:
                self.previous_shot = (row, c)
                return self.previous_shot
            #continue search right until non-hit encountered or edge of board
            c = col
            while c < board._width and board.observation[row, c] == HIT:
                c += 1
            if c < board._width and board.observation[row, c] == UNKNOWN:
                self.previous_shot = (row, c)
                return self.previous_shot

        #if reached, no adjacent hits -> shoot random valid adjacent square
        r, c = hit
        valid_adj = valid_adjacent_squares(board, r, c)
        try:
            assert len(valid_adj) > 0, 'error: no valid adjacent squares to {}'.format(hit)
        except(AssertionError) as e:
            print(board.state)
            print(board.observation)
            print(self.hits)
            raise e
        self.previous_shot = random.choice(valid_adj)
        #print('random')
        return self.previous_shot

    def reset(self):
        self.previous_shot = None #tuple
        self.hits = []  #list of known hits (not including sunk ships)


class ProbabilisticPlayer(game.Player):
    """
    Shoot the square that is a hit the maximum number of times across
    the entire statespace
    """
    def __init__(self, delay=0):
        self.delay = delay
        self.reset()
    
    def take_turn(self, board):
        time.sleep(self.delay)

    def reset(self):
        self.previous_shot = None #tuple
        self.hits = []  #list of known hits (not including sunk ships)


def pick_random_valid_move(board):
    moves = [(r,c) for r,c in itertools.product(range(board._height), range(board._width))]
    valid_moves = [(r,c) for r,c in moves if board.observation[r,c] == UNKNOWN]
    return random.choice(valid_moves)

def valid_adjacent_squares(board, row, col):
    adj_squares = [(r,c) for r,c in [(row,col+1), (row,col-1), (row+1,col), (row-1,col)]]
    adj_squares = [(r, c) for r,c in adj_squares if r >= 0 and r < board._height and c >= 0 and c < board._width]
    valid_adj = [(r, c) for r,c in adj_squares if board.observation[r,c] == UNKNOWN]
    return valid_adj


class BattleshipCLI(game.Display):
    """displays game state to the terminal"""
    def __init__(self):
        #maps board values to characters
        self.encoding = {UNKNOWN: ' ', MISS: 'o', HIT: 'x', SUNK: '#'}

    def update(self, board):
        """update display based on board observation"""
        assert isinstance(board, BattleshipBoard), 'board must be instance of BattleshipBoard'
        os.system('clear||cls') # clear terminal
        print('Shots Fired: {}'.format(board.shots_fired))
        height = board._height
        width = board._width
        line = '    ' + '  '.join('{:2}'.format(k) for k in range(width))
        print(line)
        horizontal_line = '-' * (4*width - 1)
        print('   +' + horizontal_line + '+')
        for i in range(height):
            row = board.observation[i]
            line = '{:2} |'.format(i)
            for j in row:
                line += ' ' + self.encoding[j] + ' |'
            print(line)
            if i < height -1:
                print('   |' + horizontal_line + '|')
        print('   +' + horizontal_line + '+')


class BattleshipBoard(game.Board):
    """contains the state of the battleship game and the logic to update it.

        Attributes:
            - observation (numpy.array): a height x width array representing
              what the player knowns about the board
                - value encodings:
                        0: unknown
                        1: miss
                        2: hit
                        3: sunk
            - state (numpy.array): a height x width array representing ship locations
                - value encodings:
                        0: no ship present
                        positive number: ship present (number is ship id)
            #- shots_fired (int): number of turns that have been taken
    """
    def __init__(self, height=10, width=10):
        """
        Parameters:
            - height (int): height of game grid
            - width (int): width of game grid
        """
        self._height = height
        self._width = width
        self.reset()

    def reset(self):
        """zeros out observation and state"""
        self.observation = np.zeros((self._height, self._width), dtype='int32')
        self.state = np.zeros((self._height, self._width), dtype='int32')
        self.shots_fired = 0

    def process_shot(self, row, col):
        """determine if a shot is a mit or a miss and update the observation accordingly.
        
        does nothing if grid square has already been shot at."""
        sunk_id = 0  # assume no ship has been sunk
        self.shots_fired += 1
        if self.observation[row, col]: #square already been fired at
            return
        #if its a miss, mark it as such
        if self.state[row, col] == 0:
            self.observation[row, col] = MISS
        else:
            #since it's a hit, mark it as such and check if the ship is sunk
            self.observation[row, col] = HIT
            ship_id = self.state[row, col]
            ship_coords = np.where(self.state == ship_id)
            if all(self.observation[ship_coords] == HIT):
                self.observation[ship_coords] = SUNK #ship sunk
                sunk_id = ship_id
        
        return sunk_id

    def is_game_over(self):
        """checks if all ships are sunk."""
        ship_locations = np.where(self.state > 0)
        return all(self.observation[ship_locations] == SUNK)


if __name__ == '__main__':
    #bs_game = SinglePlayerBattleship(player=RandomBattleshipPlayer(delay=.5))
    bs_game = SinglePlayerBattleship(player=HardCodedBattleshipPlayer(delay=.5))
    bs_game.play()  

'''
class TKIInterface(game.Display):
    """displays game state in a tkinter text box"""
    pass
'''