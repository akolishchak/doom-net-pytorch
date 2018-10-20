#
# mcts_dataset.py, doom-net
#
# Created by Andrey Kolishchak on 04/29/18.
#
import os
import glob
import datetime
import h5py
import bisect
from torch.utils.data import Dataset


class MCTSDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.h5_path = args.h5_path
        self.length = 0
        for filename in glob.glob(os.path.join(args.h5_path, '*.hd5')):
            self.length += args.records_per_file
        #
        # hd5 has issues with fork at DataLoader, so file is opened on first getitem()
        # https://groups.google.com/forum/#!topic/h5py/bJVtWdFtZQM
        #
        self.states = None
        self.actions = None
        self.rewards = None
        self.indexes = None

    def __getitem__(self, index):
        if self.states is None:
            self.states = []
            self.actions = []
            self.rewards = []
            self.indexes = []
            length = 0
            for filename in glob.glob(os.path.join(self.h5_path, '*.hd5')):
                file = h5py.File(filename, 'r')
                self.states.append(file['states'])
                self.actions.append(file['actions'])
                self.rewards.append(file['rewards'])
                length += len(file['states'])
                self.indexes.append(length)

        file_num = bisect.bisect(self.indexes, index)
        offset = index - self.indexes[file_num-1] if file_num > 0 else index
        return self.states[file_num][offset], self.actions[file_num][offset], self.rewards[file_num][offset]

    def __len__(self):
        return self.length
