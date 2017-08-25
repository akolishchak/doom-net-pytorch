import torch
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False


def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var