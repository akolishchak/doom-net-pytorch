#
# egreedy.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import random
from torch.autograd.stochastic_function import StochasticFunction
from cuda import *

class EGreedy(StochasticFunction):

    def __init__(self, epsilon=0):
        super(EGreedy, self).__init__()
        self.epsilon = epsilon

    def forward(self, probs):
        if self.epsilon != 0 and random.random() < self.epsilon:
            samples = torch.rand(probs.size(0), 1)*probs.size(1)
            samples = samples.long()
            if USE_CUDA:
                samples = samples.cuda()
        else:
           _, samples = probs.max(1)

        self.save_for_backward(probs, samples)
        self.mark_non_differentiable(samples)
        return samples

    def backward(self, reward):
        probs, samples = self.saved_tensors
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
            samples = samples.unsqueeze(0)
        grad_probs = probs.new().resize_as_(probs).zero_()
        output_probs = probs.gather(1, samples)
        output_probs.add_(1e-6).reciprocal_()
        output_probs.neg_().mul_(reward)
        # TODO: add batched index_add
        for i in range(probs.size(0)):
            grad_probs[i].index_add_(0, samples[i], output_probs[i])
        return grad_probs