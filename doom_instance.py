#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from vizdoom import *
import numpy as np

class NormalizedState:
    def __init__(self, screen, depth, labels, variables):
        self.screen = screen
        self.depth = depth
        self.labels = labels
        self.variables = variables


class DoomInstance:
    def __init__(self, config, wad, skiprate, id=None, visible=False):
        self.game = DoomGame()
        self.game.set_doom_game_path(wad)
        self.game.load_config(config)
        if visible:
            self.game.set_window_visible(True)
            self.game.set_sound_enabled(True)
            self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()

        self.game.new_episode()
        self.button_num = len(self.game.get_available_buttons())
        self.actions = np.eye(self.button_num, dtype=int).tolist()
        self.episode_return = 0
        self.skiprate = skiprate
        self.id = id

    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.skiprate)
        finished = self.game.is_episode_finished()
        if finished:
            self.episode_return = self.game.get_total_reward()
            self.game.new_episode()

        if self.game.is_player_dead():
            self.game.respawn_player()

        state = self.game.get_state()
        return state, reward, finished

    def step_normalized(self, action):
        state, reward, finished = self.step(action)
        return self.normalize(state), reward, finished

    def normalize(self, state):
        screen = (state.screen_buffer.astype(np.float32) - 127) / 255

        if state.depth_buffer is not None:
            depth = state.depth_buffer.astype(np.float32) / 255
        else:
            depth = None

        if state.game_variables is not None:
            variables = state.game_variables.astype(np.float32)
        else:
            variables = None

        return NormalizedState(screen=screen, depth=depth, labels=None, variables=variables)

    def get_state(self):
        state = self.game.get_state()
        return state

    def get_state_normalized(self):
        state = self.get_state()
        return self.normalize(state)

    def release(self):
        self.game.close()

    def get_button_num(self):
        return self.button_num

    def get_episode_return(self):
        return self.episode_return

    def get_id(self):
        return self.id













