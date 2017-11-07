#
# model.py, doom-net
#
# Created by Andrey Kolishchak on 10/29/17.
#
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def run_train(self, args):
        assert not hasattr(super(), 'run_train')

    def run_test(self, args):
        assert not hasattr(super(), 'run_test')
