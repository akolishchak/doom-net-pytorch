#
# aac_noisy.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#
import torch.nn as nn
import torch.nn.functional as F
from cuda import *
from collections import namedtuple
from noisy_linear import NoisyLinear
from aac_base import AACBase
import random


class BaseModelNoisy(AACBase):
    def __init__(self, in_channels, button_num, variable_num, frame_num):
        super(BaseModelNoisy, self).__init__()
        self.screen_feature_num = 256
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.screen_features1 = nn.Linear(512 * 2 * 4, self.screen_feature_num)
        #self.screen_features1 = nn.Linear(128 * 6 * 9, self.screen_feature_num)
        #self.screen_features1 = nn.Linear(64 * 14 * 19, self.screen_feature_num)

        self.batch_norm = nn.BatchNorm1d(self.screen_feature_num)

        layer1_size = 128
        self.action1 = NoisyLinear(self.screen_feature_num, layer1_size)
        self.action2 = NoisyLinear(layer1_size + variable_num, button_num)

        self.value1 = NoisyLinear(self.screen_feature_num, layer1_size)
        self.value2 = NoisyLinear(layer1_size + variable_num, 1)

        self.screens = None
        self.frame_num = frame_num


    def forward(self, screen, variables):
        # cnn
        screen_features = F.relu(self.conv1(screen))
        screen_features = F.relu(self.conv2(screen_features))
        screen_features = F.relu(self.conv3(screen_features))
        screen_features = F.relu(self.conv4(screen_features))
        screen_features = F.relu(self.conv5(screen_features))
        screen_features = F.relu(self.conv6(screen_features))
        screen_features = screen_features.view(screen_features.size(0), -1)

        # features
        input = self.screen_features1(screen_features)
        input = self.batch_norm(input)
        input = F.relu(input)

        # action
        action = F.relu(self.action1(input))
        action = torch.cat([action, variables], 1)
        action = self.action2(action)

        return action, input

    def transform_input(self, screen, variables):
        screen_batch = []
        if self.frame_num > 1:
            if self.screens is None:
                self.screens = [[]] * len(screen)
            for idx, screens in enumerate(self.screens):
                if len(screens) >= self.frame_num:
                    screens.pop(0)
                screens.append(screen[idx])
                if len(screens) == 1:
                    for i in range(self.frame_num - 1):
                        screens.append(screen[idx])
                screen_batch.append(torch.cat(screens, 0))
            screen = torch.stack(screen_batch)

        screen = Variable(screen, volatile=not self.training)
        variables = Variable(variables / 100, volatile=not self.training)
        return screen, variables

    def set_terminal(self, terminal):
        if self.screens is not None:
            indexes = torch.nonzero(terminal == 0).squeeze()
            for idx in range(len(indexes)):
                self.screens[indexes[idx]] = []

    def sample_noisy_weight(self):
        self.action1.sample()
        self.action2.sample()
        self.value1.sample()
        self.value2.sample()


ModelOutput = namedtuple('ModelOutput', ['log_action', 'value'])


class AdvantageActorCriticNoisy(BaseModelNoisy):
    def __init__(self, args):
        super(AdvantageActorCriticNoisy, self).__init__(args.screen_size[0]*args.frame_num, args.button_num, args.variable_num, args.frame_num)
        if args.base_model is not None:
            # load weights from the base model
            base_model = torch.load(args.base_model)
            self.conv1.load_state_dict(base_model.conv1.state_dict())
            self.conv2.load_state_dict(base_model.conv2.state_dict())
            self.conv3.load_state_dict(base_model.conv3.state_dict())
            self.conv4.load_state_dict(base_model.conv4.state_dict())
            self.conv5.load_state_dict(base_model.conv5.state_dict())
            self.conv6.load_state_dict(base_model.conv6.state_dict())
            self.screen_features1.load_state_dict(base_model.screen_features1.state_dict())
            self.batch_norm.load_state_dict(base_model.batch_norm.state_dict())
            self.action1.weight_mu.data = base_model.action1.weight.data.clone()
            self.action2.weight_mu.data = base_model.action2.weight.data.clone()
            del base_model

        self.discount = args.episode_discount
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def reset(self):
        self.outputs = []
        self.rewards = []
        self.discounts = []
        self.sample_noisy_weight()

    def forward(self, screen, variables):
        action_prob, input = super(AdvantageActorCriticNoisy, self).forward(screen, variables)

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
        value = F.relu(self.value1(input))
        value = torch.cat([value, variables], 1)
        value = self.value2(value)

        # save output for backpro
        action_prob = F.log_softmax(action_prob, dim=1)
        self.outputs.append(ModelOutput(action_prob.gather(-1, action), value))
        return action, value

    def get_action(self, state):
        action, _ = self.forward(*self.transform_input(state.screen, state.variables))
        return action.data

    def set_reward(self, reward):
        self.rewards.append(reward * 0.01)  # no clone() b/c of * 0.01

    def set_terminal(self, terminal):
        super(AdvantageActorCriticNoisy, self).set_terminal(terminal)
        self.discounts.append(self.discount * terminal)

    def backward(self):
        #
        # calculate step returns in reverse order
        #rewards = torch.stack(self.rewards, dim=0)
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

        weights_l2 = 0
        for param in self.parameters():
            weights_l2 += param.norm(2)

        loss = policy_loss.mean() / steps + value_loss / steps + 0.00001 * weights_l2
        loss.backward()

        # reset state
        self.reset()
