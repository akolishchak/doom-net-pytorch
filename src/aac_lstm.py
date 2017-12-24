#
# aac_lstm.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch.nn as nn
import torch.nn.functional as F
from cuda import *
from collections import namedtuple
from lstm import LSTM
from aac_base import AACBase
import random


ModelOutput = namedtuple('ModelOutput', ['log_action', 'value'])

class BaseModelLSTM(AACBase):
    def __init__(self, in_channels, button_num, variable_num):
        super(BaseModelLSTM, self).__init__()
        self.screen_feature_num = 512
        self.feature_num = self.screen_feature_num
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.screen_features1 = LSTM(512 * 2 * 4, self.screen_feature_num)
        self.hx = None
        self.cx = None

        layer1_size = 128
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size + variable_num, button_num)

        self.value1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.value2 = nn.Linear(layer1_size + variable_num, 1)


    def forward(self, screen, variables):
        # cnn
        screen_features = F.relu(self.conv1(screen))
        screen_features = F.relu(self.conv2(screen_features))
        screen_features = F.relu(self.conv3(screen_features))
        screen_features = F.relu(self.conv4(screen_features))
        screen_features = F.relu(self.conv5(screen_features))
        screen_features = F.relu(self.conv6(screen_features))
        screen_features = screen_features.view(screen_features.size(0), -1)
        # lstm
        if self.hx is None:
            self.hx = Variable(torch.zeros(screen_features.size(0), self.feature_num), volatile=not self.training)
            self.cx = Variable(torch.zeros(screen_features.size(0), self.feature_num), volatile=not self.training)
        self.hx, self.cx = self.screen_features1(screen_features, (self.hx, self.cx))

        # action
        action = F.relu(self.action1(self.hx))
        action = torch.cat([action, variables], 1)
        action = self.action2(action)
        return action

    def set_terminal(self, terminal):
        terminal = Variable(terminal.view(-1, 1).expand_as(self.cx))
        self.cx = self.cx * terminal
        self.hx = self.hx * terminal

    def reset(self):
        if self.hx is not None:
            self.hx = Variable(self.hx.data, volatile=not self.training)
        if self.cx is not None:
            self.cx = Variable(self.cx.data, volatile=not self.training)


class AdvantageActorCriticLSTM(BaseModelLSTM):
    def __init__(self, args):
        super(AdvantageActorCriticLSTM, self).__init__(args.screen_size[0]*args.frame_num, args.button_num, args.variable_num)
        if args.base_model is not None:
            # load weights from the base model
            base_model = torch.load(args.base_model)
            self.load_state_dict(base_model.state_dict())

        self.discount = args.episode_discount
        self.value = nn.Linear(self.feature_num, 1)
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def reset(self):
        if self.hx is not None:
            self.hx = Variable(self.hx.data)
        if self.cx is not None:
            self.cx = Variable(self.cx.data)
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def forward(self, screen, variables):
        action_prob = super(AdvantageActorCriticLSTM, self).forward(screen, variables)

        if not self.training:
            _, action = action_prob.max(1, keepdim=True)
            return action, None

        # greedy actions
        if random.random() < 0.1:
            action = torch.LongTensor(action_prob.size(0), 1).random_(0, action_prob.size(1))
            action = Variable(action)
            if USE_CUDA:
                action = action.cuda()
        else:
            _, action = action_prob.max(1, keepdim=True)

        # value prediction - critic
        value = F.relu(self.value1(self.hx))
        value = torch.cat([value, variables], 1)
        value = self.value2(value)

        # save output for backpro
        action_prob = F.log_softmax(action_prob, dim=1)
        self.outputs.append(ModelOutput(action_prob.gather(-1, action), value))
        return action, value

    def get_action(self, state):
        screen = Variable(state.screen, volatile=not self.training)
        variables = Variable(state.variables / 100, volatile=not self.training)
        action, _ = self.forward(screen, variables)
        return action.data

    def set_reward(self, reward):
        self.rewards.append(reward * 0.01)  # no clone() b/c of * 0.01

    def set_terminal(self, terminal):
        super(AdvantageActorCriticLSTM, self).set_terminal(terminal)
        self.discounts.append(self.discount * terminal)

    def backward(self):
        #
        rewards = self.rewards

        returns = torch.Tensor(len(rewards) - 1, *self.outputs[-1].value.data.size())
        step_return = self.outputs[-1].value.data.cpu()
        for i in range(len(rewards) - 2, -1, -1):
            step_return.mul_(self.discounts[i]).add_(rewards[i])
            returns[i] = step_return

        if USE_CUDA:
            returns = returns.cuda()
        #
        # calculate losses
        policy_loss = 0
        value_loss = 0
        steps = len(self.outputs) - 1
        for i in range(steps):
            advantage = Variable(returns[i] - self.outputs[i].value.data)
            policy_loss += -self.outputs[i].log_action * advantage
            value_loss += F.smooth_l1_loss(self.outputs[i].value, Variable(returns[i]))

        loss = policy_loss.mean() / steps + value_loss / steps
        loss.backward()

        # reset state
        self.reset()

