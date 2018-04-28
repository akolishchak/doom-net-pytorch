#
# noisy_linear.py, doom-net
#
# Created by Andrey Kolishchak on 07/04/17.
#
# based on https://arxiv.org/abs/1706.10295
#
import torch
import math
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from device import device


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(out_features, in_features)
        self.weight_epsilon = torch.Tensor(out_features, in_features)
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias_epsilon = torch.Tensor(out_features)
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
            self.bias_epsilon = None
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
        self.reset_parameters()
        self.sampled = False

    def sample(self):
        if self.training:
            self.weight_epsilon.normal_()
            self.weight = self.weight_epsilon.mul(self.weight_sigma).add_(self.weight_mu)
            if self.bias is not None:
                self.bias_epsilon.normal_()
                self.bias = self.bias_epsilon.mul(self.bias_sigma).add_(self.bias_mu)
        else:
            self.weight = self.weight_mu.detach()
            if self.bias is not None:
                self.bias = self.bias_mu.detach()
        self.sampled = True

    def reset_parameters(self):
        stdv = math.sqrt(3.0 / self.weight.size(1))
        self.weight_mu.uniform_(-stdv, stdv)
        self.weight_sigma.fill_(0.017)
        if self.bias is not None:
            self.bias_mu.uniform_(-stdv, stdv)
            self.bias_sigma.fill_(0.017)

    def forward(self, input):
        if not self.sampled:
            self.sample()
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
