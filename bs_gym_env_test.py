import numpy as np
from bs_gym_env import BattleshipEnv
from utils import show_ships

def test_PLACE_SHIPS_looks_reasonable_NO_FAVORS(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset()
        cumulative_ships += env.state

    print("\nNO FAVORS")
    show_ships(cumulative_ships)


def test_PLACE_SHIPS_looks_reasonable_TOP(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_top=100)
        cumulative_ships += env.state

    print("\nTOP")
    show_ships(cumulative_ships)

    
def test_PLACE_SHIPS_looks_reasonable_LEFT(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_left=100)
        cumulative_ships += env.state

    print("\nLEFT")
    show_ships(cumulative_ships)

    
def test_PLACE_SHIPS_looks_reasonable_BOTTOM(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_bottom=100)
        cumulative_ships += env.state

    print("\nBOTTOM")
    show_ships(cumulative_ships)

    
def test_PLACE_SHIPS_looks_reasonable_RIGHT(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_right=100)
        cumulative_ships += env.state

    print("\nRIGHT")
    show_ships(cumulative_ships)

    
def test_PLACE_SHIPS_looks_reasonable_TOP_RIGHT(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_top=100, favor_right=100)
        cumulative_ships += env.state

    print("\nTOP RIGHT")
    show_ships(cumulative_ships)

    
def test_PLACE_SHIPS_looks_reasonable_TOP_RIGHT_with_STEEP_GRADIENT(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_top=100, favor_right=100, gradient_coef=lambda x: x**10)
        cumulative_ships += env.state

    print("\nTOP RIGHT with STEEP GRADIENT")
    show_ships(cumulative_ships)

    
def test_PLACE_SHIPS_looks_reasonable_LEFT_with_STEEP_GRADIENT(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_left=10000, gradient_coef=lambda x: x**10)
        cumulative_ships += env.state

    print("\nLEFT with STEEP GRADIENT")
    show_ships(cumulative_ships)


def test_PLACE_SHIPS_looks_reasonable_LEFT_with_UNSTEEP_GRADIENT(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_left=10, gradient_coef=lambda x: x**(1/2))
        cumulative_ships += env.state

    print("\nLEFT with UNSTEEP GRADIENT")
    show_ships(cumulative_ships)


def test_PLACE_SHIPS_looks_reasonable_CENTER(n=1000):
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_top=-10000, favor_right=-10000, favor_left=-10000, favor_bottom=-10000)
        cumulative_ships += env.state

    print("\nCENTER")
    show_ships(cumulative_ships)


def test_PLACE_SHIPS_looks_reasonable_CENTER_with_STEEP_GRADIENT(n=1000):
    """
    Note: To get a really defined center, the gradient coeff needs to be something that grows slowly like lambda x: x**(1/10)
    """
    w=10
    h=10
    cumulative_ships = np.zeros((h, w), dtype=np.int32)
    env = BattleshipEnv(width=w, height=h)

    for _ in range(n):
        env.reset(favor_top=-100000, favor_right=-100000, favor_left=-100000, favor_bottom=-100000, gradient_coef=lambda x: x**(1/1000))
        cumulative_ships += env.state

    print("\nCENTER with STEEP GRADIENT")
    show_ships(cumulative_ships)


def test_PLACE_SHIPS_looks_reasonable_HORIZONTAL(n=1000):
    w=10
    h=10
    env = BattleshipEnv(width=w, height=h)
    
    for _ in range(n):
        env.reset(vert_probability=0)
        seen_in_previous_rows = []
        for row in env.state:
            newly_seen = []
            for element in row:
                if element != 0:
                    if element in seen_in_previous_rows:
                        print("\nBAD!!! Saw a %s in multiple rows. Showing the offending board:" % element)
                        show_ships(env.state)
                        raise Exception("BAD!!! Saw a %s in multiple rows." % element)
                    else:
                        newly_seen.append(element)
            seen_in_previous_rows += newly_seen

    print("\nHORIZONTAL")
    show_ships(env.state)


def test_PLACE_SHIPS_looks_reasonable_VERTICAL(n=1000):
    w=10
    h=10
    env = BattleshipEnv(width=w, height=h)
    
    for _ in range(n):
        env.reset(vert_probability=1)
        seen_in_previous_rows = []
        for col in range(w):
            newly_seen = []
            for row in range(h):
                element = env.state[row][col]
                if element != 0:
                    if element in seen_in_previous_rows:
                        print("\nBAD!!! Saw a %s in multiple cols. Showing the offending board:" % element)
                        show_ships(env.state)
                        raise Exception("BAD!!! Saw a %s in multiple col." % element)
                    else:
                        newly_seen.append(element)
            seen_in_previous_rows += newly_seen

    print("\nVERTICAL")
    show_ships(env.state)


test_PLACE_SHIPS_looks_reasonable_NO_FAVORS()
test_PLACE_SHIPS_looks_reasonable_TOP_RIGHT_with_STEEP_GRADIENT()
test_PLACE_SHIPS_looks_reasonable_CENTER_with_STEEP_GRADIENT()
# test_PLACE_SHIPS_looks_reasonable_HORIZONTAL()
test_PLACE_SHIPS_looks_reasonable_VERTICAL()