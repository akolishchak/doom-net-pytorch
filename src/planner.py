#
# planner.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import itertools
import math
import numpy as np
from doom_object import DoomObject
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn.functional as F


import vizdoom


class Map:
    def __init__(self):
        self.map = None
        self.map_points = None
        self.pose_x = 0
        self.pose_y = 0
        self.pose_theta = 0
        self.origin_x = 0
        self.origin_y = 0
        self.channel_num = 3
        self.rotation = torch.empty(360, 2, 2)
        rotation_num = self.rotation.shape[0]
        for i in range(rotation_num):
            theta = math.radians(i*360/rotation_num)
            cos = math.cos(theta)
            sin = math.sin(theta)
            self.rotation[i] = torch.tensor([
                [cos, -sin],
                [sin, cos]
            ])

    @staticmethod
    def get_screen(state):
        screen = state.screen[5:8]
        return screen

    @staticmethod
    def get_points(screen):
        points = np.transpose(np.nonzero(screen))
        return points

    def update(self, state):
        if type(state) != list:
            view = self.get_screen(state)
            shape = view.shape
            points = torch.tensor(np.transpose(np.nonzero(view)), dtype=torch.float)
            #
            im = np.flip(view, axis=1).transpose(1, 2, 0)
            plt.imsave('view.png', im, cmap=cm.gray)
        else:
            # round view
            shape = self.get_screen(state[0]).shape
            points = [
                self.get_points(self.get_screen(state[0])) + np.array([0, shape[1], 0]),
                self.get_points(np.rot90(self.get_screen(state[1]), 1, (-2, -1))) + np.array([0, 0, shape[2] // 2]),
                self.get_points(np.rot90(self.get_screen(state[2]), 2, (-2, -1))),
                self.get_points(np.rot90(self.get_screen(state[3]), 3, (-2, -1))),
            ]

            points = np.vstack(points)
            points = torch.tensor(points)

        if self.map_points is None:
            self.map_points = points
            self.map, shift_y, shift_x = self.expand_points(points)
            self.origin_y = shape[1] + shift_y
            self.origin_x = shape[2]//2 + shift_x

            self.pose_y = self.origin_y
            self.pose_x = self.origin_x
            return

        # get conv kernel with rotation in channels
        points_num = points.shape[0]
        # normalize point coordinates to center mass
        _, max_y, max_x = points.max(dim=0)[0]
        _, min_y, min_x = points.min(dim=0)[0]
        mid_y = (max_y.item() + min_y.item()) // 2
        mid_x = (max_x.item() + min_x.item()) // 2
        points -= torch.tensor([0, mid_y, mid_x])

        points = points.repeat(self.rotation.shape[0], 1).view(self.rotation.shape[0], points_num, -1).float()

        coordinate_points = points[:, :, 1:]
        points[:, :, 1:] = coordinate_points.bmm(self.rotation)
        squeezed_points = points.view(-1, 3).long()

        _, max_y, max_x = squeezed_points.max(dim=0)[0]
        _, min_y, min_x = squeezed_points.min(dim=0)[0]
        size_y = max_y - min_y + 1
        size_x = max_x - min_x + 1
        squeezed_points -= torch.tensor([0, min_y, min_x])

        channels = torch.arange(self.rotation.shape[0], dtype=torch.long)[:, None].repeat(1, points_num).view(-1)
        kernel = torch.zeros(self.rotation.shape[0], self.channel_num, size_y, size_x)
        kernel[channels[:], squeezed_points[:, 0], squeezed_points[:, 1], squeezed_points[:, 2]] = 1

        # convolve with map
        kernel = kernel.view(kernel.shape[0], 1, *kernel.shape[1:])
        map = self.map[None, None, :]
        conv = F.conv3d(map, kernel)
        #kernel = kernel.view(kernel.shape[0]*3, 1, *kernel.shape[2:])
        #map = self.map[None, :]
        #conv = F.conv2d(map, kernel, groups=3)

        # compute max activated locations
        conv = conv.squeeze()
        _, max_idx = conv.view(-1).max(dim=0)
        max_idx = max_idx.item()
        loc_r = max_idx // (conv.shape[1]*conv.shape[2])
        loc_x = max_idx % conv.shape[2] + size_x.item() // 2
        loc_y = (max_idx // conv.shape[2]) - loc_r * conv.shape[1] + size_y.item() // 2

        points = points[loc_r].long()
        points += torch.tensor([0, loc_y, loc_x])

        # debug trace
        map = map.squeeze()
        map[points[:, 0], points[:, 1], points[:, 2]] = 1
        im = np.flip(map.numpy(), axis=1).transpose(1, 2, 0)
        plt.imsave('map_new.png', im, cmap=cm.gray)
        selected_view = kernel[loc_r].squeeze().numpy()
        im = np.flip(selected_view, axis=1).transpose(1, 2, 0)
        plt.imsave('sel_view.png', im, cmap=cm.gray)

        # update map
        joint_points = torch.cat([self.map_points, points], dim=0)
        self.map, shift_y, shift_x = self.expand_points(joint_points)
        self.origin_y += shift_y
        self.origin_x += shift_x
        self.pose_y = loc_y + shift_y + mid_y - shape[1]
        self.pose_x = loc_x + shift_x + mid_x - shape[2]//2
        self.map_points = self.map.nonzero()

        map = self.map.numpy().transpose(1, 2, 0)
        map[self.origin_y, self.origin_x] = np.array([1, 1, 1])
        map[self.pose_y, self.pose_x] = np.array([0, 1, 1])
        #im = np.flip(map, axis=0)
        im = map
        plt.imsave('map.png', im, cmap=cm.gray)
        print(self.map_points.shape)


    def expand_points(self, points, shift=80):
        _, max_y, max_x = points.max(dim=0)[0]
        _, min_y, min_x = points.min(dim=0)[0]
        size_y = max_y - min_y + 1 + shift * 2
        size_x = max_x - min_x + 1 + shift * 2
        points -= torch.tensor([0, min_y - shift, min_x - shift])
        view = torch.zeros(*(self.channel_num, size_y, size_x))
        view[points[:, 0], points[:, 1], points[:, 2]] = 1

        return view, -min_y + shift, -min_x + shift

    @staticmethod
    def draw_points(points):
        _, max_y, max_x = np.max(points, axis=0)
        _, min_y, min_x = np.min(points, axis=0)
        shift = 10
        size_y = max_y - min_y + shift * 2
        size_x = max_x - min_x + shift * 2
        points = points - np.array([0, min_y - shift, min_x - shift])
        view = np.zeros([3, size_y, size_x])
        view[points[:, 0], points[:, 1], points[:, 2]] = 1
        im = np.flip(view, axis=1).transpose(1, 2, 0)
        # im = view.transpose(1, 2, 0)
        plt.imsave('points.png', im, cmap=cm.gray)


class Planner:
    def __init__(self, args):
        super().__init__()
        self.map = Map()

    def run_train(self, args):
        pass

    @staticmethod
    def get_object_lead(objects, distance):
        object_groups = [(key, len(list(group))) for key, group in itertools.groupby(objects)]
        exit_signs = []
        pos = 0
        for type, size in object_groups:
            if type == DoomObject.Type.EXIT_SIGN:
                idx = pos + size // 2
                exit_signs.append(idx)
            pos += size

        exits = []
        pos = 0
        for type, size in object_groups:
            if type == DoomObject.Type.EXIT:
                idx = pos + size // 2
                exits.append(idx)
            pos += size

        lead = None
        if exits:
            lead = int(np.array(exits).mean())
        elif len(exit_signs) >= 1:
            if len(exit_signs) == 1:
                lead = exit_signs[0]
            else:
                lead = exit_signs[0] + distance[exit_signs[0]:exit_signs[-1]].argmax()
            #lead = int(np.array(exit_signs).mean())
            #lead = exit_signs[distance[exit_signs].argmin()]
            #lead -= 10
            #lead = lead + distance[lead:lead+10].argmax()

        print(lead)
        return lead, exits

    @staticmethod
    def get_distance_lead(distance):
        lead = distance.argmax()
        max_distance = distance[lead]
        size = 1
        for i in range(lead + 1, len(distance)):
            if distance[i] != max_distance:
                break
            size += 1
        lead += size // 2
        return lead

    def wait(self, game):
        for _ in range(10):
            step_state, _, finished = game.step_normalized([0, 0, 0, 0])

    def get_round_view(self, game):
        round_view = []
        step_state = game.get_state_normalized()
        for _ in range(4):
            #im = np.flip(step_state.screen[5:8], axis=1).transpose(1, 2, 0)
            #plt.imsave('view.png', im, cmap=cm.gray)
            round_view.append(step_state)
            step_state, _, finished = game.step_normalized_noskip([0, 0, 90, 0, 0])
        return round_view

    def run_test(self, args):
        print("testing...")

        game = args.instance_class(
            args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.ASYNC_PLAYER,
            actions=args.action_set)
        step_state = game.get_state_normalized()
        step = 0

        #self.map.update(step_state)

        '''
        round_view = self.get_round_view(game)
        self.map.update(round_view)
        step_state, _, finished = game.step_normalized([100, 0, -45/args.skiprate, 0, 0])
        self.wait(game)
        round_view = self.get_round_view(game)
        self.map.update(round_view)

        step_state, _, finished = game.step_normalized([100, 0, 0, 0, 0])
        self.wait(game)
        round_view = self.get_round_view(game)
        self.map.update(round_view)

        step_state, _, finished = game.step_normalized([100, 0, 0, 0, 0])
        step_state, _, finished = game.step_normalized([100, 0, 0, 0, 0])
        self.wait(game)
        round_view = self.get_round_view(game)
        self.map.update(round_view)
        '''
        is_exit = False

        while True:
            distance = step_state.distance
            objects = step_state.objects

            im = np.flip(step_state.screen[5:8], axis=1).transpose(1, 2, 0)
            plt.imsave('view.png', im, cmap=cm.gray)

            lead = len(distance)//2
            if step % 20 == 0:
                # round view
                round_view = self.get_round_view(game)
                self.map.update(round_view)
                objects = np.hstack([view.objects for view in round_view])
                distance = np.hstack([view.distance for view in round_view])

                lead, is_exit = self.get_object_lead(objects, distance)
                if lead is None:
                    lead = distance[np.r_[:640, 960:]].argmax()
                    if lead >= 640:
                        lead += 320

            print(lead)
            if lead < 320:
                angle = math.atan(game.tan[lead])
                angle = math.degrees(angle / args.skiprate)
            else:
                half_views = lead // 160
                angle = half_views * 45
                lead = lead - half_views*160
                angle += math.degrees(math.atan(game.tan[lead]))
                turns = int(angle // 90)
                angle = angle % 90
                for _ in range(turns):
                    step_state, _, finished = game.step_normalized_noskip([0, 0, 90, 0, 0])
                angle = angle / args.skiprate

            print(angle)
            #print('='*40)

            if distance[lead] > 1:
                action = [100, 0, angle, 0, 0]
            else:
                action = [100, 0, angle, 0, 1]

            # render
            step_state, _, finished = game.step_normalized(action)
            step += 1
            if finished:
                print("episode return: {}".format(game.get_episode_return()))

