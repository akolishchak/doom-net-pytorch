#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from vizdoom import *
import numpy as np


class DoomInstance:
    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, args=None, config_wad=None, map_id=None):
        self.game = DoomGame()
        self.game.set_doom_game_path(wad)
        self.game.load_config(config)
        self.game.set_mode(mode)
        if config_wad is not None:
            self.game.set_doom_scenario_path(config_wad)
        if map_id is not None:
            self.game.set_doom_map('map{:02d}'.format(map_id+1))
        self.visible = visible
        self.episode_return = 0
        self.skiprate = skiprate
        self.id = id

        if self.visible:
            self.game.set_window_visible(True)
            self.game.set_sound_enabled(True)
            self.game.set_render_all_frames(True)

        #self.game.set_window_visible(True)

        if args is not None:
            self.game.add_game_args(args)

        self.game.init()
        self.new_episode()

        if actions is None:
            self.actions = np.eye(len(self.game.get_available_buttons()), dtype=int).tolist()
        elif len(actions) != 0:
            self.actions = actions
        else:
            self.actions = None

        if self.actions is not None:
            self.button_num = len(self.actions)
            self.use_action_set = True
        else:
            self.button_num = len(self.game.get_available_buttons())
            self.use_action_set = False

        self.variables = None
        state = self.get_state()
        if state.game_variables is not None:
            self.variables = state.game_variables

    def step(self, action):
        reset_variables = False
        if self.use_action_set:
            action = self.actions[action]

        if self.game.is_player_dead():
            self.game.respawn_player()
            reset_variables = True

        if self.visible is False:
            reward = self.game.make_action(action, self.skiprate)
        else:
            self.game.set_action(action)
            for i in range(self.skiprate):
                self.game.advance_action(1, True)
            reward = self.game.get_last_reward()

        episode_finished = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        finished = episode_finished or dead
        if finished:
            self.episode_return = self.game.get_total_reward()

        if finished:
            self.new_episode()
            reset_variables = True

        state = self.get_state()

        if reset_variables and state.game_variables is not None:
            self.variables = state.game_variables

        return state, reward, finished, dead

    def advance(self):
        self.game.advance_action(self.skiprate)
        action = self.game.get_last_action()
        reward = self.game.get_last_reward()
        finished = self.game.is_episode_finished()
        return action, reward, finished

    def step_normalized(self, action):
        state, reward, finished, dead = self.step(action)
        state = self.normalize(state)

        #if reward != 4:
        #    print(reward)
        #reward = 0
        #if state.variables is not None:
        #    diff = state.variables - self.variables
        #    reward = diff.sum()
        #    self.variables = state.variables.copy()

        return state, reward, finished

    class NormalizedState:
        def __init__(self, screen, variables=None, depth=None, labels=None, automap=None):
            self.screen = screen
            self.depth = depth
            self.labels = labels
            self.variables = variables
            self.labels = labels
            self.automap = automap

    def normalize(self, state):
        if state.screen_buffer is not None:
            screen = state.screen_buffer.astype(np.float32) / 127.5 - 1.
        else:
            screen = None

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

        return self.NormalizedState(screen=screen, variables=variables, depth=depth, labels=labels, automap=automap)

    def get_state(self):
        state = self.game.get_state()
        while state is None:
            if self.game.is_player_dead():
                self.game.respawn_player()
                self.game.advance_action(1)
            elif self.game.is_episode_finished():
                self.new_episode()
            else:
                self.game.advance_action(1)
            state = self.game.get_state()

        return state

    def get_state_normalized(self):
        state = self.get_state()
        return self.normalize(state)

    def is_finished(self):
        self.game.is_episode_finished()

    def new_episode(self):
        if self.visible:
            #file_name = '{:%Y-%m-%d_%H-%M-%S}_rec.lmp'.format(datetime.datetime.now())
            #self.game.new_episode(file_name)
            self.game.new_episode()
        else:
            self.game.new_episode()

    def release(self):
        self.game.close()

    def get_button_num(self):
        return self.button_num

    def get_episode_return(self):
        return self.episode_return

    def get_id(self):
        return self.id
