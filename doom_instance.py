#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from vizdoom import *
import numpy as np
import datetime


class NormalizedState:
    def __init__(self, screen, variables=None, depth=None, labels=None, automap=None):
        self.screen = screen
        self.depth = depth
        self.labels = labels
        self.variables = variables
        self.labels = labels
        self.automap = automap


class DoomInstance:
    def __init__(self, config, wad, skiprate, id=None, visible=False, actions=None):
        self.game = DoomGame()
        self.game.set_doom_game_path(wad)
        self.game.load_config(config)
        self.visible = visible
        if self.visible:
            self.game.set_window_visible(True)
            self.game.set_sound_enabled(True)
            self.game.set_mode(Mode.ASYNC_PLAYER)

        self.variables = None
        self.game.init()
        self.new_episode()

        if actions is None:
            self.actions = np.eye(len(self.game.get_available_buttons()), dtype=int).tolist()
        else:
            self.actions = actions
        self.button_num = len(self.actions)

        self.episode_return = 0
        self.skiprate = skiprate
        self.id = id

    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.skiprate)
        finished = self.game.is_episode_finished()
        if finished:
            self.episode_return = self.game.get_total_reward()
            self.new_episode()

        if self.game.is_player_dead():
            self.new_episode(is_finished=False)

        state = self.game.get_state()
        return state, reward, finished

    def advance(self):
        self.game.advance_action()
        action = self.game.get_last_action()
        reward = self.game.get_last_reward()
        finished = self.game.is_episode_finished()
        return action, reward, finished

    def step_normalized(self, action):
        state, reward, finished = self.step(action)
        state = self.normalize(state)
        # comment this for basic and rocket configs
        if state.variables is not None:
            diff = state.variables - self.variables
            #diff[2] *= 20
            reward += diff.sum()
            self.variables = state.variables

        return state, reward, finished

    @staticmethod
    def normalize(state):
        screen = state.screen_buffer.astype(np.float32) / 127.5 - 1.
        #screen = state.labels_buffer / 127.5 - 1.
        screen = screen[None, :, :]

        if state.game_variables is not None:
            variables = state.game_variables
        else:
            variables = None

        if state.depth_buffer is not None:
            depth = state.depth_buffer / 127.5 - 1.
        else:
            depth = None

        if state.labels_buffer is not None:
            labels = state.labels_buffer
        else:
            labels = None

        if state.automap_buffer is not None:
            automap = state.automap_buffer
        else:
            automap = None

        return NormalizedState(screen=screen, variables=variables, depth=depth, labels=labels, automap=automap)

    def get_state(self):
        state = self.game.get_state()
        return state

    def get_state_normalized(self):
        state = self.get_state()
        if state is not None:
            return self.normalize(state)
        else:
            return None

    def is_finished(self):
        self.game.is_episode_finished()

    def new_episode(self, is_finished=True):
        if is_finished:
            if self.visible:
                file_name = '{:%Y-%m-%d_%H-%M-%S}_rec.lmp'.format(datetime.datetime.now())
                self.game.new_episode(file_name)
            else:
                self.game.new_episode()
        else:
            self.game.respawn_player()
        state = self.game.get_state()
        if state.game_variables is not None:
            self.variables = state.game_variables

    def release(self):
        self.game.close()

    def get_button_num(self):
        return self.button_num

    def get_episode_return(self):
        return self.episode_return

    def get_id(self):
        return self.id













