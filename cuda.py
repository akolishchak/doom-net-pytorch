import torch
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
