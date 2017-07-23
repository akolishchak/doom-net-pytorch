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
from cuda import *
import argparse
from doom_instance import *
from aac_lstm import BaseModelLSTM


def data_generator(screens, variables, labels, episodes, step_size):
    # remove short episodes
    episode_min_size = 20*step_size
    episodes = episodes[episodes[:, 1]-episodes[:, 0] > episode_min_size]
    episodes_num = len(episodes)
    #
    batch_size = 64
    step_idx = episodes[:, 0].copy() + np.random.randint(step_size, size=episodes_num)
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
            step_idx[idx] += step_size
            if step_idx[idx] > episodes[idx][1]:
                step_idx[idx] = episodes[idx][0] + np.random.randint(step_size)
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
    train_generator = data_generator(screens, variables, labels, episodes, args.skiprate)

    model = BaseModelLSTM(input_shape[0], len(action_sets), variables.shape[1])

    #source_model = torch.load('imitation_model_lstm_bn0.pth')
    #model.load_state_dict(source_model.state_dict())
    #del source_model

    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    optimizer.zero_grad()
    running_loss = 0
    running_accuracy = 0
    batch_time = time.time()
    cp = 0
    batches_per_cp = 5000

    for batch, (screens, variables, labels, terminals) in enumerate(train_generator):
        screens, variables, labels = Variable(screens), Variable(variables), Variable(labels)
        outputs = model(screens, variables)
        loss = criterion(outputs, labels)
        model.set_terminal(terminals)

        running_loss += loss.data[0]
        _, pred = outputs.data.max(1)
        accuracy = (pred == labels.data).float().mean()
        running_accuracy += accuracy

        if batch % args.episode_length == args.episode_length - 1:
            loss.backward()
            optimizer.step()
            model.reset()
            optimizer.zero_grad()

            running_loss /= args.episode_length
            running_accuracy /= args.episode_length

            print(
                '[{:d}] loss: {:.3f}, accuracy: {:.3f}, time: {:.6f}'.format(
                    batch + 1, running_loss, running_accuracy, time.time()-batch_time
                )
            )
            running_loss = 0
            running_accuracy = 0
            batch_time = time.time()

        if batch % batches_per_cp == batches_per_cp - 1:
            cp += 1
            torch.save(model, 'imitation_model_lstm_cp' + str(cp) + '.pth')

    torch.save(model, 'cig_map01_imitation_model_lstm.pth')
    np.save('action_set', action_sets)


def test(args):
    print("testing...")
    model = torch.load('cig_map01_imitation_model_lstm.pth')
    if USE_CUDA:
        model.cuda()
    model.eval()

    action_sets = np.load('action_set.npy').tolist()

    game = DoomInstance(args.vizdoom_config, args.wad_path, args.skiprate, visible=True, actions=action_sets)
    step_state = game.get_state_normalized()

    while True:
        # convert state to torch tensors
        inputs = torch.from_numpy(step_state.screen)
        inputs = Variable(inputs, volatile=True)
        # compute an action
        outputs = model(inputs)
        _, action = outputs.data.max(1)

        # render
        step_state, _, finished = game.step_normalized(action[0][0])
        if finished:
            print("episode return: {}".format(game.get_episode_return()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Recorder')
    #parser.add_argument('--vizdoom_config', default='environments/health_gathering.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_config', default='environments/cig.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=os.path.expanduser('~') + '/tools/ViZDoom/bin/vizdoom',
                        help='path to vizdoom')
    parser.add_argument('--wad_path', default=os.path.expanduser('~') + '/tools/ViZDoom/scenarios/Doom2.wad',
                        help='wad file path')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/cig_map01/flat.h5',
                        help='hd5 file path')
    parser.add_argument('--skiprate', type=int, default=4, help='number of skipped frames')
    parser.add_argument('--episode_length', type=int, default=20, help='episode length')

    args = parser.parse_args()

    train(args)

    test(args)