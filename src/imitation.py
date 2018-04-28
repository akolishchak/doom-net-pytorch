#
# imitation.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from device import device
import argparse
from doom_instance import *
from aac import BaseModel


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
        self.variables = None

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.h5_path, 'r')
            self.inputs = self.data['screens']
            self.labels = self.data['action_labels']
            self.variables = self.data['variables']
        return self.inputs[index].astype(np.float32) / 127.5 - 1.0, \
               self.variables[index].astype(np.float32) / 100, \
               self.labels[index].astype(np.int)

    def __len__(self):
        return self.length


def train(args):

    train_set = DoomDataset(args.h5_path)
    np.save('action_set', train_set.action_sets)
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=100, shuffle=True)

    model = BaseModel(train_set.input_shape[0], len(train_set.action_sets), 3, args.frame_num).to(device)

    if args.load is not None and os.path.isfile(args.load):
        print("loading model parameters {}".format(args.load))
        source_model = torch.load(args.load)
        model.load_state_dict(source_model.state_dict())
        del source_model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(1500000):
        running_loss = 0
        running_accuracy = 0
        batch_time = time.time()
        for batch, (screens, variables, labels) in enumerate(training_data_loader):
            screens, variables, labels = screens.to(device), variables.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(screens, variables)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            accuracy = (pred == labels).float().mean()
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

        if epoch % args.checkpoint_rate == args.checkpoint_rate - 1:
            torch.save(model, args.checkpoint_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Recorder')
    parser.add_argument('--batch_size', type=int, default=100, help='number of game instances running in parallel')
    parser.add_argument('--load', default=None, help='path to model file')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/cig_map01/flat.h5',
                        help='hd5 file path')

    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--checkpoint_file', default=None, help='check point file name')
    parser.add_argument('--checkpoint_rate', type=int, default=5000, help='number of batches per checkpoit')
    args = parser.parse_args()

    train(args)
