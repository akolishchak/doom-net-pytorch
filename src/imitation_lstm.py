#
# imitation_lstm.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from device import device
import argparse
from doom_instance import *
from aac_lstm import BaseModelLSTM


def data_generator(args, screens, variables, labels, episodes):
    # remove short episodes
    episode_min_size = args.episode_size*args.skiprate
    episodes = episodes[episodes[:, 1]-episodes[:, 0] > episode_min_size]
    episodes_num = len(episodes)
    #
    batch_size = args.batch_size
    step_idx = episodes[:, 0].copy() + np.random.randint(args.skiprate, size=episodes_num)
    step_screens = np.ndarray(shape=(batch_size, *screens.shape[1:]), dtype=np.float32)
    step_variables = np.ndarray(shape=(batch_size, *variables.shape[1:]), dtype=np.float32)
    step_labels = np.ndarray(shape=(batch_size,), dtype=np.int)
    step_terminals = np.ones(shape=(batch_size, 1), dtype=np.float32)
    # select episodes for the initial batch
    batch_episodes = np.random.randint(episodes_num, size=batch_size)
    while True:
        for i in range(batch_size):
            idx = batch_episodes[i]
            step_screens[i, :] = screens[step_idx[idx]] / 127.5 - 1.0
            step_variables[i, :] = variables[step_idx[idx]] / 100
            step_labels[i] = labels[step_idx[idx]]
            step_idx[idx] += args.skiprate
            if step_idx[idx] > episodes[idx][1]:
                step_idx[idx] = episodes[idx][0] + np.random.randint(args.skiprate)
                step_terminals[i] = 0
                # reached terminal state, select a new episode
                batch_episodes[i] = np.random.randint(episodes_num)
            else:
                step_terminals[i] = 1

        yield torch.from_numpy(step_screens), \
              torch.from_numpy(step_variables), \
              torch.from_numpy(step_labels), \
              torch.from_numpy(step_terminals)


def train(args):

    data_file = h5py.File(args.h5_path, 'r')
    screens = data_file['screens']
    variables = data_file['variables']
    labels = data_file['action_labels']
    print('Dataset size =', len(screens))
    action_sets = data_file['action_sets'][:]
    episodes = data_file['episodes'][:]
    input_shape = screens[0].shape
    train_generator = data_generator(args, screens, variables, labels, episodes)

    model = BaseModelLSTM(input_shape[0], len(action_sets), variables.shape[1]).to(device)

    #source_model = torch.load('imitation_model_lstm_bn0.pth')
    #model.load_state_dict(source_model.state_dict())
    #del source_model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    optimizer.zero_grad()
    running_loss = 0
    running_accuracy = 0
    batch_time = time.time()
    cp = 0

    for batch, (screens, variables, labels, terminals) in enumerate(train_generator):
        screens, variables, labels = screens.to(device), variables.to(device), labels.to(device)
        outputs = model(screens, variables)
        loss = criterion(outputs, labels)
        model.set_terminal(terminals)

        running_loss += loss.item()
        _, pred = outputs.max(1)
        accuracy = (pred == labels).float().mean()
        running_accuracy += accuracy

        if batch % args.episode_size == args.episode_size - 1:
            loss.backward()
            optimizer.step()
            model.reset()
            optimizer.zero_grad()

            running_loss /= args.episode_size
            running_accuracy /= args.episode_size

            print(
                '[{:d}] loss: {:.3f}, accuracy: {:.3f}, time: {:.6f}'.format(
                    batch + 1, running_loss, running_accuracy, time.time()-batch_time
                )
            )
            running_loss = 0
            running_accuracy = 0
            batch_time = time.time()

        if batch % args.checkpoint_rate == args.checkpoint_rate - 1:
            cp += 1
            torch.save(model, args.checkpoint_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imitation Model Trainer')
    parser.add_argument('--batch_size', type=int, default=100, help='number of game instances running in parallel')
    parser.add_argument('--load', default=None, help='path to model file')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/cig_map01/flat.h5',
                        help='hd5 file path')

    parser.add_argument('--skiprate', type=int, default=4, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--checkpoint_file', default=None, help='check point file name')
    parser.add_argument('--checkpoint_rate', type=int, default=5000, help='number of batches per checkpoit')
    parser.add_argument('--episode_size', type=int, default=20, help='episode length')

    args = parser.parse_args()

    train(args)
