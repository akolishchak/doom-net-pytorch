#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from vizdoom import *
import numpy as np
from doom_instance import DoomInstance


class DoomInstanceD3(DoomInstance):
    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, args=None, config_wad=None, map_id=None):
        super().__init__(config, wad, skiprate, visible, mode, actions, id, args)

    def step(self, action):
        reset_variables = False
        if self.use_action_set:
            action = self.actions[action]

        if self.game.is_player_dead():
            self.game.respawn_player()
            reset_variables = True

        reward = self.game.make_action(action, self.skiprate)

        episode_finished = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        finished = episode_finished or dead
        if finished:
            self.episode_return = self.variables[2]

        if finished:
            self.new_episode()
            reset_variables = True

        state = self.get_state()

        if reset_variables and state.game_variables is not None:
            self.variables = state.game_variables

        return state, reward, finished, dead


    def step_normalized(self, action):
        state, reward, finished, dead = self.step(action)
        state = self.normalize(state)

        if state.variables is not None:
            diff = state.variables - self.variables
            diff[2] *= 100
            reward += diff.sum()
            self.variables = state.variables.copy()

        return state, reward, finished
