from enum import Enum

import copy
from os import close 
import numpy as np
import random
from gym import Env
from gym.spaces import Discrete

from gym_pomdp.envs.coord import Grid, Coord
from gym_pomdp.envs.gui import ShipGui


# TODO fix state
class Compass(Enum):
    North = Coord(0, 1)
    East = Coord(1, 0)
    South = Coord(0, -1)
    West = Coord(-1, 0)
    Null = Coord(0, 0)
    NorthEast = Coord(1, 1)
    SouthEast = Coord(1, -1)
    SouthWest = Coord(-1, -1)
    NorthWest = Coord(-1, 1)

    @staticmethod
    def get_coord(idx):
        return list(Compass)[idx].value


class Obs(Enum):
    NULL = 0
    HIT = 1


class Ship(object):
    def __init__(self, coord, length):
        self.pos = coord
        self.direction = np.random.randint(4)
        self.length = length


class ShipState(object):
    def __init__(self):
        self.ships = []
        self.total_remaining = 0


class Cell(object):
    def __init__(self):
        self.occupied = False
        self.visited = False
        self.diagonal = False


class BattleGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)

    def build_board(self, value=0):
        self.board = []
        for idx in range(self.n_tiles):
            self.board.append(Cell())
        self.board = np.asarray(self.board).reshape((self.x_size, self.y_size))


class BattleShipEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=(5, 5), max_len=3):
        self.grid = BattleGrid(board_size)
        self.action_space = Discrete(self.grid.n_tiles)
        self.observation_space = Discrete(len(Obs))
        self.num_obs = 2
        self._reward_range = self.action_space.n / 4.
        self._discount = 1.
        self.total_remaining = max_len - 1
        self.max_len = max_len + 1

    def seed(self, seed=None):
        np.random.seed(seed)

    def _compute_prob(self, action, next_state, ob):

        action_pos = self.grid.get_coord(action)
        cell = self.grid[action_pos]
        if ob == Obs.NULL.value and cell.visited:
            return 1
        elif ob == Obs.HIT.value and cell.occupied:
            return 1
        else:
            return int(ob == Obs.NULL.value)

    def step(self, action):

        assert self.done is False
        assert self.action_space.contains(action)
        assert self.total_remaining > 0
        self.last_action = action
        self.t += 1
        action_pos = self.grid.get_coord(action)
        # cell = self.grid.get_value(action_pos)
        cell = self.grid[action_pos]
        reward = 0
        if cell.visited:
            reward -= 10
            obs = Obs.NULL.value
        else:
            if cell.occupied:
                reward -= 1
                obs = 1
                self.state.total_remaining -= 1

                for d in range(4, 8):
                    if self.grid[action_pos + Compass.get_coord(d)]:
                        self.grid[action_pos + Compass.get_coord(d)].diagonal = False
            else:
                reward -= 1
                obs = Obs.NULL.value
            cell.visited = True
        if self.state.total_remaining == 0:
            reward += self.grid.n_tiles
            self.done = True
        self.tot_rw += reward
        return obs, reward, self.done, {"state": self.state}

    def _set_state(self, state):
        self.done = False
        self.tot_rw = 0
        self.t = 0
        self.last_action = -1
        self.state = self._reset_to_state(state)
        return Obs.NULL.value

    def _reset_to_state(self,state):
        bsstate = ShipState()
        self.grid.build_board()
        for ship in state.ships:
            self.mark_ship(ship, self.grid, bsstate)
            bsstate.ships.append(ship)
        return bsstate


    def close(self):
        return

    def reset(self):
        self.done = False
        self.tot_rw = 0
        self.t = 0
        self.last_action = -1
        self.state = self._get_init_state()
        return Obs.NULL.value

    def render(self, gui_reset = False, mode='human', close = False):
        if close:
            return
        if mode == 'human':
            if not hasattr(self, "gui") or gui_reset:
                obj_pos = []
                for ship in self.state.ships:
                    pos = ship.pos
                    obj_pos.append(self.grid.get_index(pos))
                    for i in range(ship.length):
                        pos += Compass.get_coord(ship.direction)
                        obj_pos.append(self.grid.get_index(pos))
                self.gui = ShipGui(board_size=self.grid.get_size, obj_pos=obj_pos)
            if self.t > 0:
                msg = "A: " + str(self.grid.get_coord(self.last_action)) + "T: " + str(self.t) + "Rw :" + str(
                    self.tot_rw)
                self.gui.render(state=self.last_action, msg=msg)

    def _generate_legal(self):
        # assert self.state.total_remaining > 0
        actions = []
        for action in range(self.action_space.n):
            action_pos = self.grid.get_coord(action)
            if not self.grid[action_pos].visited:
                actions.append(action)
        # assert len(actions) > 0
        return actions

    def _get_init_state(self):
        bsstate = ShipState()
        self.grid.build_board()

        for length in reversed(range(1, self.max_len)):
            # num_ships = 1
            # for idx in range(num_ships):
            while True:  # add one ship of each kind
                ship = Ship(coord=self.grid.sample(), length=length)
                if not self.collision(ship, self.grid, bsstate):
                    break
            self.mark_ship(ship, self.grid, bsstate)
            bsstate.ships.append(ship)
        return bsstate

    @staticmethod
    def mark_ship(ship, grid, state):

        pos = ship.pos  # .copy()

        for i in range(ship.length + 1):
            cell = grid[pos]
            assert not cell.occupied
            cell.occupied = True
            if not cell.visited:
                state.total_remaining += 1
            pos += Compass.get_coord(ship.direction)

    @staticmethod
    def collision(ship, grid, state):

        pos = ship.pos
        for i in range(ship.length + 1):
            if not grid.is_inside(pos + Compass.get_coord(ship.direction)):
                return True
            # cell = grid.get_value(pos)
            cell = grid[pos]
            if cell.occupied:
                return True
            for adj in range(8):
                coord = pos + Compass.get_coord(adj)
                if grid.is_inside(coord) and grid[coord].occupied:
                    return True
            pos += Compass.get_coord(ship.direction)
        return False

class po_rollout(object):
    def __init__(self, env, num_particle = 10000, num_sim = 10000):

        self.num_particle = num_particle   # number of particles
        self.num_sim = num_sim             # MC simulation size
        self.sim_env = env                 # simulation env
        self.belief = []                   # belief states
        self.legal_actions = []            # legal actions
        self.applied_actions = []         #save applied actions
        self.visual_close = True          # viaualize the rollouts
        #self.belief = self.scenario_gen(env, num_particle) #initialize belief states (random sampling over inital states)
        

    def _init_solver(self):
        scenarios = []
        for _ in range(self.num_particle):
            state = self.sim_env._get_init_state()   #randomly inial state sample
            scenarios.append(state)

        self.belief = scenarios   #initialize belief set

    def _MC_sim(self):
        num_action = len(self.legal_actions)                                                   # number of actions
        num_sim_actions = int(np.floor(self.num_sim/num_action)+0.01)           
        avg_rw = np.zeros((num_action))

        if len(self.belief)!=0:
            for a in range(num_action):
                init_action = self.legal_actions.pop(0)                                                 #remove the selected action from the set of legal actions
                for _ in range(num_sim_actions):
                    self._reset_sim()                                                              # reset the simulator with new samples state from the belief set
                    action_set = random.sample(self.legal_actions, len(self.legal_actions))        # generate a random action sequence from the set of available legal actions 
                    action_set.insert(0, init_action)
                    tot_rw = self._rollout(action_set)                                                      # rollout for the sampled action set
                    avg_rw[a] += tot_rw
                avg_rw[a] = avg_rw[a]/num_sim_actions
                self.legal_actions.append(init_action)                                                  # push back the action to the end of legal list of actions
                #print(a)

            max_action_idx = np.argmax(avg_rw)
            best_action = self.legal_actions[max_action_idx]
            self.applied_actions.append(best_action)
        else:
            max_action_idx = random.randint(0,len(self.legal_actions)-1)
            best_action = self.legal_actions[max_action_idx]
            self.applied_actions.append(best_action)

        return max_action_idx, avg_rw

    def _reset_sim(self):
        state = random.sample(self.belief,1)                      # belief sampling
        self.sim_env._set_state(state[0])
        for action in self.applied_actions:
            self._generative_model(action)
        return state[0]

    def _reset_sim_state(self, state):
        self.sim_env._set_state(state)
        return state
        
    
    def _rollout(self, action_set):       
        done = False
        tot_rw = 0
        self.sim_env.render(gui_reset = True, close = self.visual_close)                     # reset gui
        for action in action_set:
            _ , rw , done = self._generative_model(action)
            self.sim_env.render(close = self.visual_close)
            tot_rw += rw
            if done:
                return tot_rw
        return tot_rw

    def _generative_model(self, action):
        ob, rw, done, info = self.sim_env.step(action)
        return ob, rw, done
    

    def _belief_update(self, action, ob):
        updated_belief = []
        K = 0           #keep track of new particles
        no_belief = False
        
        #first iterate over all particles 
        for particle in self.belief:
            state = self._reset_sim_state(particle)
            new_ob, _, _ = self._generative_model(action)
            if new_ob==ob:
                K+=1
                updated_belief.append(state)

        #check if we still need particles
        if K==self.num_particle:  # particles full
            self.belief = updated_belief
            return no_belief
        else:
            if len(updated_belief) == 0:  # no particle at all!
                no_belief = True
                self.belief = updated_belief
                return no_belief
            else:                               # particle not full
                while K != self.num_particle:
                    temp = random.sample(updated_belief, 1)
                    state = temp[0]
                    state = self._reset_sim_state(state)
                    new_ob, _, _ = self._generative_model(action)
                    if new_ob==ob:
                        K+=1
                        updated_belief.append(state)
                self.belief = updated_belief
                return no_belief

    
#    def _MC_sim():
#        k = 1

#    def _belief_sample():
#        k = 1

    
            


if __name__ == "__main__":
    board_size = (5, 5)  # grid_size 5*5
    max_ship_len = 3    # ship lengths 1*2 and  1*3 
    env = BattleShipEnv(board_size = board_size, max_len = max_ship_len-1)
    ob = env.reset()

    solver = po_rollout(copy.deepcopy(env))  
    solver._init_solver()   # initialize the POMDP solver (initialize belief set)

    remaining_action = env._generate_legal()  # list of all remaining actions
    env.render()
    done = False
    t = 0
    while not done:
        #action = env.action_space.sample()
        #action = remaining_action.pop(random.randint(0,len(remaining_action)))
        solver.legal_actions = remaining_action         #update the action set of solver with the updated legal actions
        action_idx, avg_rwd = solver._MC_sim()
        action = remaining_action.pop(action_idx)
        ob, rw, done, info = env.step(action)
        env.render()
        # update the belief set
        no_belief = solver._belief_update(action, ob)
        if no_belief == True:
            print("no belief particle survived ==> random sampling!")

        t += 1
        if t%5==0:
            print(env.state.total_remaining)
    env.close()

    print("total_rw {}, rw {}, t{}".format(env.tot_rw, rw, t))
