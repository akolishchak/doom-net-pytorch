#
# model.py, doom-net
#
# Created by Andrey Kolishchak on 10/29/17.
#
import os
import torch
import torch.nn as nn
from device import device

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def run_train(self, args):
        assert not hasattr(super(), 'run_train')

    def run_test(self, args):
        assert not hasattr(super(), 'run_test')

    @staticmethod
    def create(model_class, args, param_file=None):
        model = model_class(args).to(device)
        if param_file is not None and os.path.isfile(param_file):
            assert os.path.isfile(param_file)
            print("Loading {} parameters from {}".format(type(model).__name__, param_file))
            state_dict = torch.load(param_file)
            model.load_state_dict(state_dict)
        return model
