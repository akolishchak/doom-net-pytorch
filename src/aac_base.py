#
# aac_base.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
from multiprocessing.pool import ThreadPool
import time
import torch
import torch.optim as optim
from device import device
from model import Model
import vizdoom
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


class AACBase(Model):
    def __init__(self):
        super(AACBase, self).__init__()

    def run_train(self, args):
        print("training...")
        params = list(self.parameters())
        params_num = sum(param.numel() for param in params if param.requires_grad)
        print("Parameters = ", params_num)
        self.train()

        optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=1e-5, amsgrad=True)
        if args.load is not None and os.path.isfile(args.load + '_optimizer.pth'):
            optimizer_dict = torch.load(args.load+'_optimizer.pth')
            optimizer.load_state_dict(optimizer_dict)

        optimizer.zero_grad()

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(args.batch_size, *args.screen_size)
        state.variables = torch.Tensor(args.batch_size, args.variable_num)
        reward = torch.Tensor(args.batch_size, 1)
        terminal = torch.Tensor(args.batch_size, 1)
        episode_return = torch.zeros(args.batch_size)

        games = []
        for i in range(args.batch_size):
            games.append(args.instance_class(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set, id=i))

        pool = ThreadPool()

        def get_state(game):
            id = game.get_id()
            normalized_state = game.get_state_normalized()
            state.screen[id, :] = torch.from_numpy(normalized_state.screen)
            state.variables[id, :] = torch.from_numpy(normalized_state.variables)

        pool.map(get_state, games)
        # start training
        for episode in range(args.episode_num):
            batch_time = time.time()
            for step in range(args.episode_size):
                # get action
                action = self.get_action(state)
                # step and get new state
                def step_game(game):
                    id = game.get_id()
                    normalized_state, step_reward, finished = game.step_normalized(action[id][0])
                    state.screen[id, :] = torch.from_numpy(normalized_state.screen)
                    state.variables[id, :] = torch.from_numpy(normalized_state.variables)
                    reward[id, 0] = step_reward
                    episode_return[id] = float(game.get_episode_return())
                    if finished:
                        #episode_return[id] = float(game.get_episode_return())
                        # cut rewards from future actions
                        terminal[id] = 0
                    else:
                        terminal[id] = 1
                pool.map(step_game, games)
                self.set_reward(reward)
                self.set_terminal(terminal)

            # update model
            self.backward()

            grads = []
            weights = []
            for p in self.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                    weights.append(p.view(-1))
            grads = torch.cat(grads, 0)
            weights = torch.cat(weights, 0)
            grads_norm = grads.norm()
            weights_norm = weights.norm()

            assert grads_norm == grads_norm

            optimizer.step()
            optimizer.zero_grad()

            if episode % 1 == 0:
                print("{}: mean_return = {:f}, grads_norm = {:f}, weights_norm = {:f}, batch_time = {:.3f}".format(episode, episode_return.mean(), grads_norm, weights_norm, time.time()-batch_time))

            if episode % args.checkpoint_rate == 0:
                torch.save(self.state_dict(), args.checkpoint_file)
                torch.save(optimizer.state_dict(), args.checkpoint_file+'_optimizer.pth')

        # terminate games
        pool.map(lambda game: game.release(), games)

        torch.save(self.state_dict(), args.checkpoint_file)
        torch.save(optimizer.state_dict(), args.checkpoint_file+'_optimizer.pth')

    def run_test(self, args):
        print("testing...")
        self.eval()

        game = args.instance_class(
            args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.ASYNC_PLAYER, actions=args.action_set)
        step_state = game.get_state_normalized()

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(1, *args.screen_size)
        state.variables = torch.Tensor(1, args.variable_num)

        while True:

            #im = np.flip(step_state.screen[[0, 1, 7]], axis=1).transpose(1, 2, 0)
            #plt.imsave('map_view.png', im, cmap=cm.gray)

            # convert state to torch tensors
            state.screen[0, :] = torch.from_numpy(step_state.screen)
            state.variables[0, :] = torch.from_numpy(step_state.variables)
            # compute an action
            action = self.get_action(state)
            print(action)
            # render
            step_state, _, finished = game.step_normalized(action[0][0])
            if finished:
                print("episode return: {}".format(game.get_episode_return()))
                self.set_terminal(torch.zeros(1))
