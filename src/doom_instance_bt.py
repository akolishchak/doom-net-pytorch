#
# doom_instance_bt.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import glob
from vizdoom import *
from doom_instance import DoomInstance
from doom_object import DoomObject
import numpy as np
import math
import itertools
from wad import Wad


class DoomInstanceBt(DoomInstance):
    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, args=None, config_wad=None, map_id=None):
        super().__init__(config, wad, skiprate, visible, mode, actions, id, args, config_wad, map_id)
        self.angles = None
        self.map_mode = self.game.is_depth_buffer_enabled() and self.game.is_labels_buffer_enabled()
        if self.map_mode:
            width = self.game.get_screen_width()
            half_width = width / 2
            self.tan = np.ndarray(shape=(width,), dtype=float)
            fov = np.pi/2.0
            ratio = math.tan(0.5 * fov) / half_width
            for i in range(width):
                self.tan[i] = (i - half_width) * ratio

        wad = Wad(config_wad)
        if wad.levels:
            self.level_map = wad.levels[map_id].get_map()

        self.finished = False

    def step_normalized(self, action):
        state, reward, finished, dead = self.step(action)
        state = self.normalize(state)
        return state, reward, finished

    def step(self, action):
        state, reward, finished, dead = super().step(action)
        if finished and not dead:
            self.finished = True
        return state, reward, finished, dead

    class NormalizedState:
        def __init__(self, screen, variables=None, depth=None, labels=None,
                     automap=None, distance=None, objects=None):
            self.screen = screen
            self.depth = depth
            self.labels = labels
            self.variables = variables
            self.labels = labels
            self.automap = automap
            self.distance = distance
            self.objects = objects

    def normalize(self, state):
        assert state.labels_buffer is not None and state.depth_buffer is not None

        objects = None
        distance = None
        if state.labels_buffer is not None and state.depth_buffer is not None:
            screen = np.zeros([DoomObject.Type.MAX, 128, 256], dtype=np.float32)
            object_distance = np.ndarray([DoomObject.Type.MAX, *state.screen_buffer.shape[1:]], dtype=np.float32)
            object_distance.fill(256)
            for label in state.labels:
                channel = DoomObject.get_id(label)
                if channel >= 0:
                    idx = state.labels_buffer == label.value
                    object_distance[channel, idx] = state.depth_buffer[idx]

            height = state.depth_buffer.shape[0]
            mid_height = height // 2
            width = state.depth_buffer.shape[1]
            mid_width = width // 2

            object_distance[DoomObject.Type.WALLS, mid_height] = state.depth_buffer[mid_height]
            distance = object_distance.min(axis=1)
            objects = np.argmin(distance, axis=0)
            distance = distance.min(axis=0)
            distance[distance > 127.5] = 127.5
            x = np.around(128 + self.tan * distance).astype(int)
            y = np.around(distance).astype(int)
            todelete = np.where(y == 128)
            y = np.delete(y, todelete, axis=0)
            x = np.delete(x, todelete, axis=0)
            channels = np.delete(objects, todelete, axis=0)
            screen[channels, y, x] = 1

        if state.game_variables is not None:
            variables = state.game_variables
        else:
            variables = None

        if state.depth_buffer is not None:
            depth = state.depth_buffer
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

        return self.NormalizedState(screen=screen, variables=variables, depth=depth,
                               labels=labels, automap=automap, distance=distance, objects=objects)

    def get_pose(self):
        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        z = self.game.get_game_variable(GameVariable.POSITION_Z)
        heading = self.game.get_game_variable(GameVariable.ANGLE)
        return DoomObject.get_pose(DoomObject.Type.UNKNOWN, x, y, z, heading)

    def get_object_info(self, state):
        objects = []
        object_groups = [(key, len(list(group))) for key, group in itertools.groupby(state.objects)]
        pos = 0
        for type, size in object_groups:
            if type != DoomObject.Type.WALLS:
                idx = pos + size // 2
                distance = state.distance[idx]
                angle = math.degrees(math.atan(self.tan[idx]))
                objects.append([type, distance, angle])
            pos += size

        return objects

    def is_finished(self):
        return self.finished or super().is_finished()

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
