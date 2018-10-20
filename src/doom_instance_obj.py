#
# doom_instance_obj.py, doom-net
#
# Created by Andrey Kolishchak on 04/21/18.
#
from vizdoom import *
from doom_instance_cig import DoomInstanceCig
from doom_object import DoomObject
import numpy as np
import math
import itertools


class DoomInstanceObj(DoomInstanceCig):
    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, color=0, bot_num=0):
        super().__init__(config, wad, skiprate, visible, mode, actions, id, color, bot_num)

    def get_pose(self):
        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        z = self.game.get_game_variable(GameVariable.POSITION_Z)
        heading = self.game.get_game_variable(GameVariable.ANGLE)
        return DoomObject.get_pose(DoomObject.Type.AGENT, x, y, z, heading)

    @staticmethod
    def get_objects(state):
        objects = []
        if state.labels_buffer is not None and state.depth_buffer is not None:
            for label in state.labels:
                object_type = DoomObject.get_id(label)
                if object_type >= 0 and object_type != DoomObject.Type.WALLS:
                    x = label.object_position_x
                    y = label.object_position_y
                    heading = label.object_angle
                    velocity_x = label.object_velocity_x
                    velocity_y = label.object_velocity_y
                    objects.append([object_type, x, y, 0, heading, velocity_x, velocity_y])

        return np.array(objects, dtype=np.float32)
