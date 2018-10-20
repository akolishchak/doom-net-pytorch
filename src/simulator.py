#
# simulator.py, doom-net
#
# Created by Andrey Kolishchak on 07/04/17.
#
#
import os
import argparse
import os.path
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import vizdoom
from doom_env import init_doom_env
from doom_object import DoomObject
from multiprocessing import Process, Manager


class State:
    def __init__(self, map_points, pose, agents, bullets, objects):
        # put all into a single array
        self.array = np.vstack([
            pose,
            agents,
            bullets,
            objects,
            [0, 0, 0, 0, 0, 0, 0]
        ])
        bullets_offset = len(agents)+1
        objects_offset = bullets_offset + len(bullets)
        self.agents = self.array[:bullets_offset]
        self.bullets = self.array[bullets_offset:objects_offset]
        self.objects = self.array[objects_offset:len(objects)]
        self.moving = self.array[:objects_offset]
        # TODO: variables
        self.variables = self.array[-1]
        self.key = self.array.data.tobytes()
        #
        # TODO: compute model state
        #
        self.policy_state = None


class Simulator:
    def __init__(self, policy_model):
        map_image = plt.imread('/home/andr/gdrive/research/ml/doom-net-pytorch/environments/cig_map.png')
        map_image = (map_image*255).astype(np.uint8)
        self.map = np.flip(map_image, axis=0)
        map_image = plt.imread('/home/andr/gdrive/research/ml/doom-net-pytorch/environments/cig_map_walls.png')
        map_image = np.flip(map_image, axis=0)
        self.wall_points = np.transpose(np.nonzero(map_image))
        self.y_ratio = self.map.shape[0]/1856
        self.x_ratio = self.map.shape[1]/1824
        self.y_shift = 352
        self.x_shift = 448
        self.theta_shift = 90
        self.policy_model = policy_model

    def get_state(self, game):
        # get pose and objects
        pose = game.get_pose()
        state = game.get_state_normalized()
        objects = game.get_objects(state)

        # agent pose
        pose = self.convert_pose(pose)
        # objects
        objects = objects[DoomObject.Type.OBSTACLE <= objects[:, DoomObject.TYPE] <= DoomObject.Type.AMMO]
        if len(objects) > 0:
            # -90 to convert from heading to x-axis of local frame (theta)
            theta = math.radians(pose[DoomObject.HEADING] - 90)
            # convert to world frame
            cos = math.cos(theta)
            sin = math.sin(theta)
            transform = np.array([
                [cos, sin],
                [-sin, cos],
                [pose[DoomObject.X], pose[DoomObject.Y]]
            ])
            coord = np.hstack([objects[:, DoomObject.X:DoomObject.Y+1], np.ones([objects.shape[0], 1])])
            coord = np.around(np.matmul(coord, transform)).astype(np.int)
            objects[:, DoomObject.X:DoomObject.Y+1] = coord

        # TODO: retrieve agent and bullet info
        agents, bullets = [], []
        state = State(self.wall_points, pose, agents, bullets, objects)
        #self.draw_objects(objects)

        return state

    def get_next_state(self, state, action):
        pass

    def rollout(self, state):
        pass

    def is_finished(self, state):
        pass

    def get_reward(self, state):
        pass

    def get_available_actions_mask(self, state):
        pass

    def get_action_size(self, state):
        pass

    def get_policy_model(self):
        pass

    def convert_pose(self, pose):
        x = np.around((pose[DoomObject.X] + self.x_shift) * self.x_ratio).astype(np.int)
        y = np.around((pose[DoomObject.Y] + self.y_shift) * self.y_ratio).astype(np.int)
        pose[DoomObject.X] = x
        pose[DoomObject.Y] = y
        return pose

    def adjust_pose(self, pose, x, y):
        if self.map[y, x] == 0:
            pose.x += x
            pose.y += y

    def move(self, pose, action):
        throttle, lateral_throttle = action[0]*0.01, action[2]*0.01
        steering = action[1]
        #
        # assume throttle equal to velocity
        velocity_forward = throttle
        velocity_lateral = lateral_throttle
        #
        # heading
        if steering != 0:
            pose.set_heading(pose.heading + steering)
        #
        # forward
        if velocity_forward != 0:
            x = velocity_forward * math.cos(pose.heading_rad)
            y = velocity_forward * math.sin(pose.heading_rad)
            self.adjust_pose(pose, x, y)
        #
        # lateral movement
        if velocity_lateral != 0:
            heading = pose.heading_rad - math.pi/2
            x = velocity_lateral * math.cos(heading)
            y = velocity_lateral * math.sin(heading)
            self.adjust_pose(pose, x, y)

    def draw_objects(self, objects):
        view = self.map.copy()
        view[objects[:, 2], objects[:, 1], 0] = 255
        view[objects[:, 2], objects[:, 1], 1] = 0
        view[objects[:, 2], objects[:, 1], 2] = 0
        plt.imsave('map_points.png', np.flip(view, axis=0))


def play():
    _vzd_path = os.path.dirname(vizdoom.__file__)
    parser = argparse.ArgumentParser(description='Doom Network')
    parser.add_argument('--action_set', default='noset', help='model to work with')
    parser.add_argument('--doom_instance', default='map', choices=('basic', 'cig', 'map'), help='doom instance type')
    parser.add_argument('--vizdoom_config', default='../environments/cig_test.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=_vzd_path, help='path to vizdoom')
    parser.add_argument('--wad_path', default=_vzd_path + '/freedoom2.wad', help='wad file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--checkpoint_file', default=None, help='check point file name')
    parser.add_argument('--checkpoint_rate', type=int, default=500, help='number of batches per checkpoit')
    parser.add_argument('--bot_cmd', default=None, help='command to launch a bot')
    parser.add_argument('--h5_path', default=None, help='hd5 files path')
    args = parser.parse_args()
    print(args)
    init_doom_env(args)

    game = args.instance_class(
                args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.PLAYER, actions=args.action_set)

    step_state = game.get_state_normalized()
    for i in range(50):
        step_state, _, finished = game.step_normalized([0, 0, 0, 0])

    pose = game.get_pose()
    state = game.get_state_normalized()
    objects = game.get_objects(state)

    sim = Simulator(pose, objects)
    k = 1


if __name__ == '__main__':
    play()
