#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import datetime
import time
from subprocess import Popen
import numpy as np
from vizdoom import *
from doom_server import start_server


class NormalizedState:
    def __init__(self, screen, variables=None, depth=None, labels=None, automap=None):
        self.screen = screen
        self.depth = depth
        self.labels = labels
        self.variables = variables
        self.labels = labels
        self.automap = automap


class DoomInstance:
    def __init__(self, config, wad, skiprate, id=None, visible=False, actions=None, bot_cmd=None, color=0):
        self.game = DoomGame()
        self.game.set_doom_game_path(wad)
        self.game.load_config(config)
        self.game.set_mode(Mode.PLAYER)
        self.visible = visible
        self.is_server = bot_cmd is not None
        self.episode_return = 0
        self.skiprate = skiprate
        self.id = id
        self.port = 55255 + id if id is not None else 0
        self.bots_num = 0
        self.bot_cmd = bot_cmd

        # game args https://zdoom.org/wiki/Command_line_parameters
        self.game.add_game_args("+name DoomNet +colorset {}".format(color))
        if self.visible:
            print("visible set to true")
            self.game.set_window_visible(True)
            self.game.set_sound_enabled(True)
            self.game.set_mode(Mode.ASYNC_PLAYER)
        self.cig = 'cig' in config
        if self.cig:
            if self.is_server:
                self.bots_num = 2
                for i in range(self.bots_num):
                    print(id, "==START BOT==", i)
                    Popen('{} --color {} --port {}'.format(self.bot_cmd, 2, self.port), shell=True)

                self.game.add_game_args("-port {}".format(self.port))

            self.game.add_game_args("-host {}".format(self.bots_num+1))
            self.game.add_game_args("-deathmatch")
            self.game.add_game_args("+sv_forcerespawn 0")
            self.game.add_game_args("+sv_noautoaim 1")
            self.game.add_game_args("+sv_respawnprotect 1")
            self.game.add_game_args("+sv_spawnfarthest 1")
            self.game.add_game_args("+sv_nocrouch 1")
            self.game.add_game_args("+viz_respawn_delay 0")
            self.game.add_game_args("+viz_nocheat 1")
            self.game.add_game_args("+viz_debug 0")
            self.game.add_game_args("+timelimit 10.0")

        self.variables = None
        self.game.init()
        if not self.is_server:
            self.new_episode()

        if actions is None:
            self.actions = np.eye(len(self.game.get_available_buttons()), dtype=int).tolist()
        else:
            self.actions = actions
        self.button_num = len(self.actions)

        state = self.get_state()
        if state.game_variables is not None:
            self.variables = state.game_variables

    def step(self, action):
        reset_variables = False

        if self.game.is_player_dead():
            self.game.respawn_player()
            reset_variables = True

        if self.visible is False:
            reward = self.game.make_action(self.actions[action], self.skiprate)
        else:
            self.game.set_action(self.actions[action])
            for i in range(self.skiprate):
                self.game.advance_action(1, True)
            reward = self.game.get_last_reward()

        episode_finished = self.game.is_episode_finished()
        finished = episode_finished or self.game.is_player_dead()
        if finished:
            if self.cig:
                self.episode_return = self.variables[2]
            else:
                self.episode_return = self.game.get_total_reward()

        if episode_finished:
            self.new_episode()
            reset_variables = True

        state = self.get_state()

        if reset_variables and state.game_variables is not None:
            self.variables = state.game_variables

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
            if self.cig:
                if diff[1] < -100:
                    diff[1] = 0
                diff = np.multiply(diff, [100 * 0.5 * (0.2 if diff[0] > 0 else 0.1), 100 * 0.5 * 0.01, 100 * 1 * 1])
                if diff[2] > 0:
                    print('HIT!!!', self.id)
                # penalize shots with zero ammo
                if self.variables[0] == 0 and self.actions[action][2] == 1:
                    diff[0] -= 10
                reward += diff.sum() - 3

            self.variables = state.variables.copy()
            if self.cig:
                state.variables[2] = 0

        return state, reward, finished

    @staticmethod
    def normalize(state):
        screen = state.screen_buffer.astype(np.float32) / 127.5 - 1.
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
        reset_variables = False
        while state is None:
            if self.game.is_player_dead():
                self.game.respawn_player()
                self.game.advance_action(1)
            elif self.game.is_episode_finished():
                self.new_episode()
                reset_variables = True
            else:
                self.game.advance_action(1)
            state = self.game.get_state()

        if reset_variables and state.game_variables is not None:
            self.variables = state.game_variables

        return state

    def get_state_normalized(self):
        state = self.get_state()
        return self.normalize(state)

    def is_finished(self):
        self.game.is_episode_finished()

    def new_episode(self):
        if self.is_server:
            pass
        if self.visible:
            file_name = '{:%Y-%m-%d_%H-%M-%S}_rec.lmp'.format(datetime.datetime.now())
            self.game.new_episode(file_name)
        else:
            self.game.new_episode()
        if self.cig:
            self.game.send_game_command("removebots")
            if self.id is not None:
                for i in range(5):
                    self.game.send_game_command("addbot")
            else:
                for i in range(10):
                    self.game.send_game_command("addbot")

    def release(self):
        self.game.close()

    def get_button_num(self):
        return self.button_num

    def get_episode_return(self):
        return self.episode_return

    def get_id(self):
        return self.id
