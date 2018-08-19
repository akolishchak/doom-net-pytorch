#
# ppo_base.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import datetime
from shutil import copyfile
from multiprocessing.pool import ThreadPool
import time
from colorama import Fore, Back, Style
import torch
from device import device
import vizdoom
from tensorboardX import SummaryWriter
import numpy as np


class PPOBase:
    def __init__(self):
        super(PPOBase, self).__init__()

    def run_train(self, args):
        policy = self

        log_basedir = 'logs/{:%Y-%m-%d %H-%M-%S}/'.format(datetime.datetime.now())
        train_writer = SummaryWriter(log_basedir)

        print("training...")
        params = list(policy.model.parameters())
        params_num = sum(param.numel() for param in params if param.requires_grad)
        print("Number of parameters = {:,}".format(params_num))

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(args.batch_size, *args.screen_size)
        state.variables = torch.Tensor(args.batch_size, args.variable_num)
        reward = torch.Tensor(args.batch_size, 1)
        non_terminal = torch.Tensor(args.batch_size, 1)
        episode_return = torch.zeros(args.batch_size)
        action = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)

        games = []
        game_levels = args.instance_class.get_game_levels(args.vizdoom_config)
        print('Game levels: ', len(game_levels))
        for i, [wad_file, map_id] in enumerate(game_levels):
            games.append(
                args.instance_class(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set, id=i, wad_file=wad_file, map_id=map_id))

        args.batch_size = len(games)

        pool = ThreadPool()

        def get_state(game):
            idx = game.get_id()
            normalized_state = game.get_state_normalized()
            state.screen[idx, :] = torch.from_numpy(normalized_state.screen)
            state.variables[idx, :] = torch.from_numpy(normalized_state.variables)

        pool.map(get_state, games)
        # start training
        for episode in range(args.episode_num):
            batch_time = time.time()
            policy.train()
            game_status = np.zeros(args.batch_size)
            for step in range(args.episode_size):
                # get action
                action = policy.get_save_action(state, action)
                #
                # step and get new state
                def step_game(game):
                    idx = game.get_id()
                    normalized_state, step_reward, finished = game.step_normalized(action[idx][0])
                    state.screen[idx, :] = torch.from_numpy(normalized_state.screen)
                    state.variables[idx, :] = torch.from_numpy(normalized_state.variables)
                    reward[idx, 0] = step_reward
                    if finished:
                        ret = game.get_episode_return()
                        episode_return[idx] = float(ret)
                        non_terminal[idx] = 0
                        action[idx] = 0
                        if ret == 0:
                            game_status[idx] = 1
                        # pick a new game
                        #with episode_games_lock:
                        #    episode_games[idx] = random_game_pool.pop(-1)
                        #    random_game_pool.insert(0, game_id)
                    else:
                        non_terminal[idx] = 1
                pool.map(step_game, games)
                policy.set_reward(reward)
                policy.set_non_terminal(non_terminal)

            # update model
            policy.set_last_state(state, action)
            grads_norm, weights_norm = policy.backward()

            if episode % 1 == 0:
                mean_return = episode_return.mean()
                finished_counter = game_status.sum()
                print(Fore.GREEN + "{}: mean_return = {:f}, finished={}, grads_norm = {:f}, weights_norm = {:f}, batch_time = {:.3f}".format(episode, mean_return, finished_counter, grads_norm, weights_norm, time.time()-batch_time) + Style.RESET_ALL)
                # tensorboard
                train_writer.add_scalar('ppo/mean_return', mean_return, episode)
                train_writer.add_scalar('ppo/finished', finished_counter, episode)
                train_writer.add_scalar('ppo/grads_norm', grads_norm, episode)
                train_writer.add_scalar('ppo/weights_norm', weights_norm, episode)

            if episode % args.checkpoint_rate == 0:
                policy.save()
                copyfile(self.args.checkpoint_file, self.args.checkpoint_file + '_{:04d}.pth'.format(episode))
                optimizer_checkpoint = self.args.checkpoint_file + '_optimizer.pth'
                copyfile(optimizer_checkpoint, optimizer_checkpoint + '_{:04d}.pth'.format(episode))

        # terminate games
        pool.map(lambda game: game.release(), games)
        # save
        policy.save()

    def run_test(self, args):
        policy = self
        print("testing...")
        policy.eval()

        game_levels = args.instance_class.get_game_levels(args.vizdoom_config)
        print('Game levels: ', len(game_levels))
        completed_games = 0
        failed_games = []
        for i, [wad_file, map_id] in enumerate(game_levels):
            print('Game: ', os.path.basename(wad_file), map_id)
            game = args.instance_class(
                args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.PLAYER, actions=args.action_set, id=i,  wad_file=wad_file, map_id=map_id, max_steps=1000, eval_mode=False)
            step_state = game.get_state_normalized()

            state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
            state.screen = torch.Tensor(1, *args.screen_size)
            state.variables = torch.Tensor(1, args.variable_num)
            action = torch.zeros(1, 1, dtype=torch.long, device=device)

            while True:
                # convert state to torch tensors
                state.screen[0, :] = torch.from_numpy(step_state.screen)[None, :]
                state.variables[0, :] = torch.from_numpy(step_state.variables)
                # compute an action
                action = policy.get_action(state, action, action_dist=False)
                step_state, _, finished = game.step_normalized(action[0][0])
                policy.set_non_terminal(torch.zeros(1, 1) if finished else torch.ones(1, 1))
                if finished:
                    action[0] = 0
                    ret = game.get_episode_return()
                    print("episode return: {}".format(ret))
                    if ret == 0:
                        completed_games += 1
                    else:
                        failed_games.append([wad_file, map_id+1])
                    break
                time.sleep(0.035)

            game.release()

        print("Completed games = {}, {}%".format(completed_games, completed_games*100/len(game_levels)))
        print("Failed games ({}):".format(len(failed_games)))
        for wad_file, map_id in failed_games:
            print("{}, map{:02d}".format(os.path.basename(wad_file), map_id))
