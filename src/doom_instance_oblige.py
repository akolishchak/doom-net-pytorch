#
# doom_instance_cig.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import glob
from vizdoom import *
from doom_instance import DoomInstance
import numpy as np
from wad import Wad
from doom_object import DoomObject


class DoomInstanceOblige(DoomInstance):
    exits = None

    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, wad_file=None, map_id=0, max_steps=150, eval_mode=False):
        args = (""
                #" +viz_nocheat 1"
                #" +viz_debug 0"
        )

        if wad_file is None:
            dir = os.path.dirname(config)
            wad_file = glob.glob(os.path.join(dir, '*.wad'))[0]

        super().__init__(config, wad, skiprate, visible, mode, actions, id, args, config_wad=wad_file, map_id=map_id)
        self.distance = 1000
        self.level_map = None

        if not eval_mode:
            wad = Wad(wad_file)
            if wad.levels:
                self.level_map = wad.levels[map_id].get_map()
                self.distance = self.get_distance()
                self.reward_ratio = self.distance / 100

        self.step_num = 0
        self.finished = False
        self.max_steps = max_steps
        self.eval_mode = eval_mode

    def step(self, action):
        if self.use_action_set:
            action = self.actions[action]

        if self.finished:
            self.new_episode()
            self.distance = self.get_distance()
            self.finished = False

        if self.game.is_player_dead():
            state = self.get_state()
            return state, -100, True, True

        reward = self.game.make_action(action, self.skiprate)

        episode_finished = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        finished = episode_finished or dead
        if finished:
            self.episode_return = self.game.get_total_reward()

        state = self.get_state()

        return state, reward, finished, dead

    def is_dead(self):
        self.game.is_player_dead()

    def step_normalized(self, action):
        state, reward, finished, dead = self.step(action)
        reward = 0
        if finished:
            reward = 10 if not dead else -10
            #print('{}!!!! id = {}, step = {}, distance = {}'.format("DEAD!!!" if dead else "FINISHED!!!", self.id, self.step_num, self.distance))
            self.step_num = 0
            self.episode_return = 0 if not dead else -self.get_distance()
        else:
            distance = self.get_distance()
            if self.distance is not None:
                diff = self.distance - distance
                reward = diff if diff != 0 else -1
            self.distance = distance

        self.step_num += 1
        if self.step_num >= self.max_steps:
            self.step_num = 0
            self.finished = True
            finished = True
            self.episode_return = -self.distance

        state = self.normalize(state)

        return state, reward, finished

    def normalize(self, state):
        state = super().normalize(state)

        if self.variables is not None:
            var_diff = state.variables - self.variables
            self.variables = state.variables
        else:
            var_diff = np.zeros_like(state.variables)

        variables = np.zeros(len(var_diff) * 2)
        for i in range(len(var_diff)):
            if var_diff[i] > 0:
                variables[i * 2] = 1
            elif var_diff[i] < 0:
                variables[i*2+1] = 1
        state.variables = variables

        return state

    def normalize_screen(self, state):
        if state.screen_buffer is not None:
            screen = state.screen_buffer.astype(np.float32) / 127.5 - 1.
        else:
            screen = None

        if state.game_variables is not None:
            variables = state.game_variables
        else:
            variables = None

        return self.NormalizedState(screen=screen, variables=variables)

    def get_distance(self):
        pose = self.get_pose()
        x1 = pose[DoomObject.X]
        y1 = pose[DoomObject.Y]
        distance = self.level_map.get_exit_distance(y1, x1)
        return distance

    def get_pose(self):
        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        z = self.game.get_game_variable(GameVariable.POSITION_Z)
        heading = self.game.get_game_variable(GameVariable.ANGLE)
        return DoomObject.get_pose(DoomObject.Type.AGENT, x, y, z, heading)

    @staticmethod
    def get_game_levels(config):
        levels = []
        # assume config in a separate dir with wad files
        dir = os.path.dirname(config)
        file_list = glob.glob(os.path.join(dir, '*.wad'))
        file_list.sort()
        for wad_file in file_list:
            wad = Wad(wad_file)
            map_num = len(wad.levels)
            levels.extend([[wad_file, i] for i in range(map_num) if wad.levels[i].get_map().get_exits()])
        return levels
