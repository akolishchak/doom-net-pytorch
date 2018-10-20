#
# doom_instance_cig.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import glob
from vizdoom import *
from doom_instance_map import DoomInstanceMap
from doom_instance import DoomInstance
import numpy as np
from wad import Wad
from doom_object import DoomObject
import math


class DoomInstanceObligeMap(DoomInstanceMap):
    exits = None

    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, wad_file=None, map_id=0, max_steps=150, eval_mode=False):
        args = (""
                #" +viz_nocheat 1"
                #" +viz_debug 0"
        )

        if wad_file is None:
            dir = os.path.dirname(config)
            wad_file = glob.glob(os.path.join(dir, '*.wad'))[0]
        #dir = os.path.dirname(config)
        #wad_file = '{}/oblige_{:04d}.wad'.format(dir, id)
        #wad_file = '{}/mock.wad'.format(dir)
        #map_id = 0

        super().__init__(config, wad, skiprate, visible, mode, actions, id, args, config_wad=wad_file, map_id=map_id)
        self.distance = 1000
        self.level_map = None

        if not eval_mode:
            wad = Wad(wad_file)
            if wad.levels:
                self.level_map = wad.levels[map_id].get_map()
                self.distance = self.get_distance()
                self.reward_ratio = 10 / self.distance

        self.step_num = 0
        self.finished = False
        self.max_steps = max_steps
        self.eval_mode = eval_mode
        self.var_diff = np.zeros(6)
        self.shot = False
        self.shot_counter = 0
        self.enemy_in_view = False
        self.killcount = 0
        '''
        width = self.game.get_screen_width()
        half_width = width / 2
        self.tan = np.ndarray(shape=(width,), dtype=float)
        fov = np.pi / 2.0
        ratio = math.tan(0.5 * fov) / half_width
        for i in range(width):
            self.tan[i] = (i - half_width) * ratio
        '''

    def step(self, action):
        reset_variables = False
        if self.use_action_set:
            action = self.actions[action]

        if self.finished:
            self.new_episode()
            self.distance = self.get_distance()
            self.finished = False

        if self.game.is_player_dead():
            state = self.get_state()
            return state, -100, True, True

        if self.visible is False:
            reward = self.game.make_action(action, self.skiprate)
            #if action[0] != 0 or action[2] != 0 or action[3] != 0:
            #    skip = 2 + np.random.randint(5)
            #else:
            #    skip = self.skiprate
            #reward = self.game.make_action(action, skip)
            '''
            '''
        else:
            # assume set_render_all_frames(True)
            reward = self.game.make_action(action, self.skiprate)
            #if action[0] != 0 or action[2] != 0 or action[3] != 0:
            #    skip = 2 + np.random.randint(5)
            #else:
            #    skip = self.skiprate
            #reward = self.game.make_action(action, skip)
            '''
            '''
            #self.game.set_action(action)
            #for i in range(self.skiprate):
            #    self.game.advance_action(1, True)
            #reward = self.game.get_last_reward()

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
                reward = diff if diff != 0 else -1 # if action != 3 else 0
                #reward = 0 if diff > 0 else -1 # if action != 3 else 0
                #if diff == 0:
                #    print("DIFF == 0")
                #print(reward)
            self.distance = distance

        self.step_num += 1
        if self.step_num >= self.max_steps:
            self.step_num = 0
            self.finished = True
            finished = True
            self.episode_return = -self.distance

        state = self.normalize(state)

        half_width = state.objects.shape[0]//2
        enemy_in_view = state.objects[half_width] == DoomObject.Type.ENEMY
        if self.var_diff[0] < 0:
            #print('HEALTH LOSS!!!')
            reward -= 2
        # hit reward
        #if (self.enemy_in_view or enemy_in_view) and self.var_diff[3] > 0:
        if action == 6:
            if self.enemy_in_view or enemy_in_view:
                #print('HIT!!!')
                reward += 5
                self.killcount += 1
        '''
            else:
                reward -= 2
        else:
            reward -= 1
        '''
        self.enemy_in_view = enemy_in_view

        if finished:
            #self.episode_return = self.killcount
            self.killcount = 0

        return state, reward, finished

    def normalize(self, state):
        '''
        labels_list = state.labels
        objects = None
        distance = None
        object_distance = np.ndarray([8, *state.screen_buffer.shape[1:]], dtype=np.float32)
        labels = np.zeros([16, 32], dtype=np.long)
        object_distance.fill(256)
        for label in labels_list:
            # print(label.object_name)
            channel = DoomObject.get_id(label)
            if channel >= 0:
                idx = state.labels_buffer == label.value
                object_distance[channel, idx] = state.depth_buffer[idx]

        height = state.depth_buffer.shape[0]
        mid_height = height // 2

        object_distance[7, mid_height] = state.depth_buffer[mid_height]
        distance = object_distance.min(axis=1)
        objects = np.argmin(distance, axis=0)
        distance = distance.min(axis=0)
        distance /= 4
        distance[distance > 15] = 15
        x = np.around(16 + self.tan * distance).astype(int)
        y = np.around(distance).astype(int)
        todelete = np.where(y == 15)
        y = np.delete(y, todelete, axis=0)
        x = np.delete(x, todelete, axis=0)
        channels = np.delete(objects, todelete, axis=0)
        labels[y, x] = channels

        state = super().normalize(state)

        state.labels = labels
        '''
        '''
        labels = np.zeros([1, 8], dtype=np.long)
        for label in labels_list:
            object_id = DoomObject.get_id(label)
            if object_id >= 0:
                labels[0, object_id] = 1
        '''
        '''
        label_sections = [state.labels[:, :50], state.labels[:, 50:110], state.labels[:, 110:]]
        labels = np.zeros([len(label_sections), 8], dtype=np.long)
        for label in labels_list:
            object_id = DoomObject.get_id(label)
            if object_id >= 0:
                for section_id in range(len(label_sections)):
                    if (label_sections[section_id] == label.value).sum() > 0:
                        labels[section_id, object_id] = 1
        state.labels = labels
        '''
        state = super().normalize(state)

        if self.variables is not None:
            self.variables[3] = self.game.get_game_variable(GameVariable.HITCOUNT)
            var_diff = state.variables - self.variables
            self.variables = state.variables
        else:
            var_diff = np.zeros_like(state.variables)
        self.var_diff = var_diff

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
