#
# map_dataset.py, doom-net
#
# Created by Andrey Kolishchak on 03/03/18.
#
import os
import sys
import numpy as np
import glob
import h5py
import datetime
import argparse
import multiprocessing as mp
from random import shuffle
from model_utils import get_model
from doom_env import init_doom_env
from doom_instance import *
import torch
from device import device


def worker(args, worker_id, levels):
    model = get_model(args)
    model.eval()

    while levels:
        try:
            wad_file, map_id = levels.pop(0)
        except:
            break

        print('{}, map{:02d}'.format(os.path.basename(wad_file), map_id))

        game = args.instance_class(
            args.vizdoom_config, args.wad_path, args.skiprate, visible=False, mode=Mode.PLAYER,
            actions=args.action_set, id=worker_id, wad_file=wad_file, map_id=map_id, max_steps=10000, eval_mode=True
        )
        step_state = game.get_state_normalized()

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(1, *args.screen_size)
        state.variables = torch.Tensor(1, args.variable_num)
        action = torch.zeros(1, 1, dtype=torch.long, device=device)

        screen_list = []
        distances_list = []
        object_list = []
        added_per_episode = 0

        for i in range(args.frames_per_worker):
            # convert state to torch tensors
            state.screen[0, :] = torch.from_numpy(step_state.screen)
            state.variables[0, :] = torch.from_numpy(step_state.variables)
            # compute an action
            action = model.get_action(state, action)
            # render
            raw_state, reward, finished, _ = game.step(action[0][0])
            step_state = game.normalize(raw_state)

            if not screen_list or not (np.array(screen_list) == raw_state.screen_buffer).any(0).all():
                #screen_list.append(raw_state.screen_buffer.astype(np.float32) / 127.5 - 1.)
                screen_list.append(raw_state.screen_buffer)
                #distances_list.append(step_state.distance.astype(np.float32) / (127.5*0.5) - 1.)
                distances_list.append(np.around(step_state.distance)[None, :].astype(np.long))
                object_list.append(step_state.objects[None, :].astype(np.long))
                added_per_episode += 1
            #else:
            #    print('dup')

            #if i % 100 == 0:
            #    print('step', i)

            if finished:
                print("episode return: {}".format(game.get_episode_return()))
                #print(added_per_episode)
                model.set_non_terminal(torch.zeros(1, 1) if finished else torch.ones(1, 1))
                if added_per_episode == 0:
                    # early exit if no new frames added
                    break
                added_per_episode = 0

        screen_list = [item.astype(np.float32) / 127.5 - 1. for item in screen_list]

        timestamp = '{:%Y-%m-%d %H-%M-%S}'.format(datetime.datetime.now())
        filename = os.path.join(args.h5_path, '{}-map{:02d}-{}.hd5'.format(os.path.basename(wad_file), map_id, timestamp))
        with h5py.File(filename, 'w') as file:
            file.create_dataset('screens', data=screen_list, dtype='float32', compression='gzip')
            #file.create_dataset('distances', data=distances_list, dtype='float32', compression='gzip')
            file.create_dataset('distances', data=distances_list, dtype='long', compression='gzip')
            file.create_dataset('objects', data=object_list, dtype='long', compression='gzip')


if __name__ == '__main__':
    _vzd_path = os.path.dirname(vizdoom.__file__)
    parser = argparse.ArgumentParser(description='Doom Dataset')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/oblige',
                        help='hd5 files path')
    parser.add_argument('--model', default='ppo_map', help='model to work with')
    parser.add_argument('--load', default='../checkpoints/oblige_ppo_map_cp.pth', help='path to model file')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='number of game instances running in parallel')
    parser.add_argument('--worker_num', type=int, default=8, help='number of workers')
    parser.add_argument('--frames_per_worker', type=int, default=2000, help='number of workers')
    parser.add_argument('--base_model', default=None, help='path to base model file')
    parser.add_argument('--episode_discount', type=float, default=0.99, help='number of episodes for training')
    parser.add_argument('--doom_instance', default='oblige', choices=('basic', 'cig', 'map', 'oblige'), help='doom instance type')
    parser.add_argument('--vizdoom_config', default='../environments/oblige2/oblige-map.cfg', help='vizdoom config path')
    parser.add_argument('--action_set', default='../actions/action_set_test_forward.npy', help='action set')
    parser.add_argument('--vizdoom_path', default=_vzd_path, help='path to vizdoom')
    parser.add_argument('--wad_path', default=_vzd_path + '/freedoom2.wad', help='wad file path')
    parser.add_argument('--skiprate', type=int, default=4, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--bot_cmd', default=None, help='command to launch a bot')
    parser.add_argument('--seed', type=int, default=1, help='seed value')

    args = parser.parse_args()
    print(args)
    init_doom_env(args)

    mp.set_start_method('spawn')

    game_levels = args.instance_class.get_game_levels(args.vizdoom_config)
    levels = mp.Manager().list(game_levels)
    threads = []
    for i in range(args.worker_num):
        thread = mp.Process(target=worker, args=(args, i, levels))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    sys.exit()

    #worker(args, 0, levels)
    #
    # partition files to train, test, and validation sets
    #
    filenames = []
    for filename in glob.glob(os.path.join(args.h5_path, '*.hd5')):
        filenames.append(os.path.basename(filename))

    total_size = len(filenames)
    train_size = round(total_size * 1)
    test_size = round((total_size - train_size)*0.95)
    validation_size = total_size - train_size - test_size

    shuffle(filenames)
    train_dir = os.path.join(args.h5_path, 'train')
    test_dir = os.path.join(args.h5_path, 'test')
    validation_dir = os.path.join(args.h5_path, 'val')

    file_idx = 0
    for directory, size in [(train_dir, train_size), (test_dir, test_size), (validation_dir, validation_size)]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for idx in range(file_idx, file_idx+size):
            os.rename(os.path.join(args.h5_path, filenames[idx]), os.path.join(directory, filenames[idx]))
        file_idx += size



