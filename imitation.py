#
# imitation.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import time
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cuda import *
import argparse
from doom_instance import *
from base_model import BaseModel


class DoomDataset(Dataset):
    def __init__(self, h5_path):
        super(DoomDataset, self).__init__()
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as data:
            inputs = data['screens']
            print('Dataset size =', len(inputs))
            self.action_sets = data['action_sets'][:]
            self.input_shape = inputs[0].shape
            self.length = len(inputs)
        #
        # hd5 has issues with fork at DataLoader, so file is opened on first getitem()
        # https://groups.google.com/forum/#!topic/h5py/bJVtWdFtZQM
        #
        self.data = None
        self.inputs = None
        self.labels = None

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.h5_path, 'r')
            self.inputs = self.data['screens']
            self.labels = self.data['action_labels']
        return self.inputs[index].astype(np.float32) / 127.5 - 1.0, self.labels[index].astype(np.int)

    def __len__(self):
        return self.length


def train(args):

    train_set = DoomDataset(args.h5_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=100, shuffle=True)

    model = BaseModel(train_set.input_shape, len(train_set.action_sets))
    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(10):
        running_loss = 0
        running_accuracy = 0
        batch_time = time.time()
        for batch, (inputs, labels) in enumerate(training_data_loader):
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            _, pred = outputs.data.max(1)
            accuracy = (pred == labels.data).float().mean()
            running_accuracy += accuracy

            batches_per_print = 10
            if batch % batches_per_print == batches_per_print-1:  # print every batches_per_print mini-batches
                print(
                    '[{:d}, {:5d}] loss: {:.3f}, accuracy: {:.3f}, time: {:.6f}'.format(
                    epoch + 1, batch + 1, running_loss/batches_per_print, running_accuracy/batches_per_print, (time.time()-batch_time)/batches_per_print
                    )
                )
                running_loss = 0
                running_accuracy = 0
                batch_time = time.time()

    torch.save(model, 'imitation_model.pth')
    np.save('action_set', train_set.action_sets)


def test(args):
    print("testing...")
    model = torch.load('imitation_model.pth')
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
    parser.add_argument('--vizdoom_config', default='environments/health_gathering.cfg', help='vizdoom config path')
    #parser.add_argument('--vizdoom_config', default='environments/deathmatch.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=os.path.expanduser('~') + '/tools/ViZDoom/bin/vizdoom',
                        help='path to vizdoom')
    parser.add_argument('--wad_path', default=os.path.expanduser('~') + '/tools/ViZDoom/scenarios/Doom2.wad',
                        help='wad file path')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/health_gathering/flat.h5',
                        help='hd5 file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')

    args = parser.parse_args()

    train(args)

    test(args)