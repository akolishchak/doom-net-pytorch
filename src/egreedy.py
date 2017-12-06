#
# egreedy.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import random
from torch.autograd import Variable
from torch.distributions import Distribution
from cuda import *

class EGreedy(Distribution):
    def __init__(self, probs, epsilon=0):
        self.probs = probs
        self.epsilon = epsilon

    def sample(self):
        if self.epsilon != 0 and random.random() < self.epsilon:
            samples = torch.rand(self.probs.size(0), 1)*self.probs.size(1)
            samples = Variable(samples.long())
            if USE_CUDA:
                samples = samples.cuda()
        else:
           _, samples = self.probs.max(1, keepdim=True)

        return samples

    def log_prob(self, value):
        return self.probs.gather(-1, value).log()
