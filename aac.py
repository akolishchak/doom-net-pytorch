#
# aac.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch.nn as nn
import torch.nn.functional as F
from cuda import *
from collections import namedtuple
from base_model import BaseModel
import numpy as np

ModelOutput = namedtuple('ModelOutput', ['action', 'value'])


class AdvantageActorCritic(BaseModel):
    def __init__(self, args):
        super(AdvantageActorCritic, self).__init__(args.screen_size, args.button_num)
        if args.base_model is not None:
            # load weights from the base model
            base_model = torch.load(args.base_model)
            self.conv1.load_state_dict(base_model.conv1.state_dict())
            self.conv2.load_state_dict(base_model.conv2.state_dict())
            self.conv3.load_state_dict(base_model.conv3.state_dict())
            self.conv4.load_state_dict(base_model.conv4.state_dict())
            self.features.load_state_dict(base_model.features.state_dict())
            self.batch_norm.load_state_dict(base_model.batch_norm.state_dict())
            self.action1.load_state_dict(base_model.action1.state_dict())
            self.action2.load_state_dict(base_model.action2.state_dict())

        self.discount = args.episode_discount
        self.value1 = nn.Linear(self.feature_num, 512)
        self.value2 = nn.Linear(512, 1)
        self.outputs = []
        self.rewards = []

    def reset(self):
        self.outputs = []
        self.rewards = []

    def forward(self, input):
        # cnn
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        input = F.relu(self.conv3(input))
        input = F.relu(self.conv4(input))

        #input = self.softmax(input / 1e-1)

        input = input.view(input.size(0), -1)
        # shared features
        features = F.relu(self.batch_norm(self.features(input)))
        # action
        action = F.relu(self.action1(features))
        action = F.softmax(self.action2(action))
        if self.training:
            action = action.multinomial()
        else:
            _, action = action.max(1)
            return action, None
        # value prediction - critic
        value = F.relu(self.value1(features))
        value = self.value2(value)
        # save output for backpro
        self.outputs.append(ModelOutput(action, value))
        return action, value

    def get_action(self, state):
        input = Variable(state.screen, volatile=not self.training)
        action, _ = self.forward(input)
        return action.data

    def set_reward(self, reward):
        self.rewards.append(reward*0.01)

    def backward(self):
        #
        # calculate step returns in reverse order
        rewards = torch.stack(self.rewards, dim=0)
        #rewards = self.rewards
        #rewards = (rewards - rewards.mean(0).expand_as(rewards)) / (rewards.std(0).expand_as(rewards) + 1e-5)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        returns = torch.Tensor(len(rewards)-1, *self.outputs[-1].value.data.size())
        step_return = self.outputs[-1].value.data.cpu()
        for i in range(len(rewards) - 2, -1, -1):
            step_return.mul_(self.discount).add_(rewards[i])
            returns[i] = step_return

        #returns = (returns - returns.mean(0).expand_as(returns)) / (returns.std(0).expand_as(returns))
        #returns = (returns - returns.mean()) / (returns.std())
        #min, _ = returns.min(0)
        #min = min.expand_as(returns)
        #max, _ = returns.max(0)
        #max = max.expand_as(returns)
        #returns = (returns - min) / (max - min)
        if USE_CUDA:
            returns = returns.cuda()
        #
        # calculate losses
        value_loss = 0
        for i in range(len(self.outputs)-1):
            self.outputs[i].action.reinforce(returns[i] - self.outputs[i].value.data)
            value_loss += F.smooth_l1_loss(self.outputs[i].value, Variable(returns[i]))
        #
        # backpro all variables at once
        variables = [value_loss] + [output.action for output in self.outputs[:-1]]
        gradients = [torch.ones(1).cuda() if USE_CUDA else torch.ones(1)] + [None for _ in self.outputs[:-1]]
        autograd.backward(variables, gradients)
        # reset state
        self.reset()

