#
# es_base.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#

import os
import datetime
from threading import Thread, Event
from queue import Queue
import time
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from device import device
from model import Model
from es_map import ESMap
import vizdoom


class ESBase:
    def __init__(self, args):
        super().__init__()
        torch.set_grad_enabled(False)


    def run_train(self, args):
        workers_num = 16

        def worker(id, args, params, out_queue):
            controller = Model.create(ESMap, args, args.load)
            controller.train()
            in_queue = Queue()
            params_num = len(params)

            game = args.instance_class(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set, id=id)
            step_state = game.get_state_normalized()

            episode_num = int(1e6)
            max_step = 1000

            state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
            state.screen = torch.Tensor(1, *args.screen_size)
            state.variables = torch.Tensor(1, args.variable_num)

            for episode in range(episode_num):

                # generate noise
                noise = torch.randn(params_num, device=device)
                new_params = params + noise*0.1
                param_start = 0

                # update controller parameters
                for param in controller.parameters():
                    if param.requires_grad:
                        size = param.numel()
                        param.data.copy_(new_params[param_start:param_start + size].view_as(param))
                        param_start += size

                for step in range(max_step):
                    # convert state to torch tensors
                    state.screen[0, :] = torch.from_numpy(step_state.screen)
                    state.variables[0, :] = torch.from_numpy(step_state.variables)

                    action = controller(state.screen.to(device), state.variables.to(device))
                    step_state, _, finished = game.step_normalized(action[0, 0])
                    if finished:
                        break
                #
                # report reward with noise
                #
                episode_return = game.get_episode_return()
                #print('{}: return {}', id, episode_return)
                out_queue.put([in_queue, game.get_episode_return(), noise])
                #
                # get updated params and start new episode
                #
                params = in_queue.get()
                game.new_episode()

        #
        # init reference controller
        #
        ref_controller = Model.create(ESMap, args, args.load)
        params = list(ref_controller.parameters())
        params_num = sum(param.numel() for param in params if param.requires_grad)
        print("Paremeters = ", params_num)
        ref_params = torch.empty(params_num, device=device)
        # copy initial parameters
        start_pos = 0
        for param in params:
            if param.requires_grad:
                size = param.numel()
                ref_params[start_pos:start_pos + size] = param.view(-1)
                start_pos += size
        ref_controller = None

        result_queue = Queue()
        threads = []
        for i in range(workers_num):
            thread = Thread(target=worker, args=(i, args, ref_params, result_queue))
            thread.start()
            threads.append(thread)

        pool_size = workers_num
        queues, rewards, noises = [], [], []
        while True:
            queue, reward, noise = result_queue.get()
            queues.append(queue)
            rewards.append(reward)
            noises.append(noise)
            if len(rewards) == pool_size:
                # compute weight updates
                advantage = torch.tensor(rewards)
                print('mean reward = {}'.format(advantage.mean()))
                advantage = F.softmax(advantage)
                advantage = advantage[:, None].to(device)
                wiegth_noise = torch.stack(noises)
                updates = wiegth_noise.mul_(advantage).sum(dim=0)

                # update weights
                #print(ref_params)
                ref_params = ref_params + 1e-3 / (pool_size * 0.1) * updates
                #print(ref_params)
                # report update to workers
                for q in queues:
                    q.put(ref_params)
                # clear the lists
                queues, rewards, noises = [], [], []

        for thread in threads:
            thread.join()


    def run_test(self, args):
        print("testing...")

        controller = Model.create(ESMap, args, args.load)
        controller.eval()

        game = args.instance_class(
            args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.ASYNC_PLAYER,
            actions=args.action_set)
        step_state = game.get_state_normalized()

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(1, *args.screen_size)
        state.variables = torch.Tensor(1, args.variable_num)

        while True:
            # convert state to torch tensors
            state.screen[0, :] = torch.from_numpy(step_state.screen)
            state.variables[0, :] = torch.from_numpy(step_state.variables)
            # compute an action
            action = controller(state)
            print(action)
            # render
            step_state, _, finished = game.step_normalized(action[0][0])
            if finished:
                print("episode return: {}".format(game.get_episode_return()))

