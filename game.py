import abc
import itertools
import random


class Game(abc.ABC):
    """contains game rules and logic"""
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """resets board and players. call before play"""
        pass

    @abc.abstractmethod
    def play(self):
        """main game loop. returns result of last game"""
        pass


class Display(abc.ABC):
    """responsible for outputting game state to player"""
    @abc.abstractmethod
    def update(self, board):
        """update the display based on new game state (board)"""
        pass


class Player(abc.ABC):
    """takes game state in form of Board instance and returns next move"""
    @abc.abstractmethod
    def take_turn(self, board):
        """returns players next move based on the curent board"""
        pass

    @abc.abstractmethod
    def reset(self):
        pass

class Board(abc.ABC):
    """contains game state"""
    @abc.abstractmethod
    def reset(self):
        """resets board to initial state"""
        pass


'''
class SinglePlayerGame(Game):
    def __init__(self):
        super().__init__()


class TwoPlayerGame(Game):
    def next_player(self):
        """returns the next player in order"""
        pass
'''