#
# aac_state_controller.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from device import device
from collections import namedtuple
from state_model import Features, StateModel
import random
from torch.distributions import Categorical


class Controller(nn.Module):
    def __init__(self, action_num):
        super().__init__()
        layer1_size = StateModel.rnn_size // 2
        self.action1 = nn.Linear(Features.size + StateModel.size, layer1_size)
        self.action2 = nn.Linear(layer1_size, action_num)
        self.value1 = nn.Linear(Features.size + StateModel.size, 128)
        self.value2 = nn.Linear(128, 1)
        self.action_bn1 = nn.BatchNorm1d(layer1_size)
        self.value_bn1 = nn.BatchNorm1d(128)

    def forward(self, features):
        action = self.action1(features)
        action = self.action_bn1(action)
        action = F.relu(action, inplace=True)
        action = self.action2(action)

        return action


ModelOutput = namedtuple('ModelOutput', ['log_action', 'value'])


class AdvantageActorCriticController(Controller):
    def __init__(self, args):
        super().__init__(args.button_num)
        if args.base_model is not None:
            # load weights from the base model
            base_model = torch.load(args.base_model)
            self.load_state_dict(base_model.state_dict())
            del base_model

        self.discount = args.episode_discount
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def reset(self):
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def forward(self, observation, state):
        features = torch.cat([observation, state], 1)
        action_prob = super().forward(features)

        if not self.training:
            _, action = action_prob.max(1, keepdim=True)
            #action = Categorical(F.log_softmax(action_prob, dim=1).exp()).sample()
            #action = action[None, :]
            return action

        # greedy actions
        if random.random() < 0.1:
            action = torch.LongTensor(action_prob.size(0), 1).random_(0, action_prob.size(1)).to(device)
        else:
           _, action = action_prob.max(1, keepdim=True)

        # value prediction - critic
        value = self.value1(features)
        value = self.value_bn1(value)
        value = F.relu(value, inplace=True)
        value = self.value2(value)

        # save output for backpro
        action_prob = F.log_softmax(action_prob, dim=1)
        self.outputs.append(ModelOutput(action_prob.gather(-1, action), value))
        return action

    def backward(self, rewards, nonterminals):
        #
        # calculate step returns in reverse order

        returns = torch.Tensor(len(rewards) - 1, *self.outputs[-1].value.size()).to(device)
        step_return = self.outputs[-1].value.detach().cpu()
        for i in range(len(rewards) - 2, -1, -1):
            step_return.mul_(self.discount * nonterminals[i]).add_(rewards[i])
            returns[i] = step_return

        #
        # calculate losses
        policy_loss = 0
        value_loss = 0
        steps = len(self.outputs) - 1
        for i in range(steps):
            advantage = returns[i] - self.outputs[i].value.detach()
            policy_loss += -self.outputs[i].log_action * advantage
            value_loss += F.smooth_l1_loss(self.outputs[i].value, returns[i])

        loss = policy_loss.mean()/steps + value_loss/steps
        loss.backward()

        # reset state
        self.reset()
