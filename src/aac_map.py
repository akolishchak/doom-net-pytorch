#
# aac_map.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from device import device
from collections import namedtuple
from aac_base import AACBase
import random


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.relu(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        residual = self.residual(x)
        residual = self.residual_bn(residual)

        output += residual
        output = self.relu(output)
        return output


class BaseModel(AACBase):
    def __init__(self, in_channels, button_num, variable_num, frame_num):
        super(BaseModel, self).__init__()
        self.screen_feature_num = 256
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, dilation=8)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, dilation=16)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=1)

        #self.screen_features1 = nn.Linear(13 * 18 * 128, self.screen_feature_num)
        self.screen_features1 = nn.Linear(5 * 21 * 128, self.screen_feature_num)

        self.batch_norm = nn.BatchNorm1d(self.screen_feature_num)

        layer1_size = 64
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size + variable_num, button_num)
        self.batch_norm_action = nn.BatchNorm1d(layer1_size + variable_num)

        self.value1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.value2 = nn.Linear(layer1_size + variable_num, 1)
        self.batch_norm_value = nn.BatchNorm1d(layer1_size + variable_num)

        self.screens = None
        self.frame_num = frame_num


    def forward(self, screen, variables):
        # cnn
        screen_features = F.relu(self.conv1(screen))
        screen_features = F.relu(self.conv2(screen_features))
        screen_features = F.relu(self.conv3(screen_features))
        screen_features = F.relu(self.conv4(screen_features))
        screen_features = F.relu(self.conv5(screen_features))
        screen_features = screen_features.view(screen_features.size(0), -1)

        # features
        input = self.screen_features1(screen_features)
        input = self.batch_norm(input)
        input = F.relu(input)

        # action
        action = F.relu(self.action1(input))
        action = torch.cat([action, variables], 1)
        action = self.batch_norm_action(action)
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

        variables /= 100
        return screen.to(device), variables.to(device)

    def set_terminal(self, terminal):
        if self.screens is not None:
            indexes = torch.nonzero(terminal.view(-1) == 0)
            for idx in range(len(indexes)):
                self.screens[indexes[idx]] = []


ModelOutput = namedtuple('ModelOutput', ['log_action', 'value'])


class AdvantageActorCriticMap(BaseModel):
    def __init__(self, args):
        super(AdvantageActorCriticMap, self).__init__(args.screen_size[0]*args.frame_num, args.button_num, args.variable_num, args.frame_num)
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

    def forward(self, screen, variables):
        action_prob, input = super(AdvantageActorCriticMap, self).forward(screen, variables)
        if not self.training:
            _, action = action_prob.max(1, keepdim=True)
            return action, None

        # greedy actions
        if random.random() < 0.1:
            action = torch.LongTensor(action_prob.size(0), 1).random_(0, action_prob.size(1)).to(device)
        else:
           _, action = action_prob.max(1, keepdim=True)

        # value prediction - critic
        value = F.relu(self.value1(input))
        value = torch.cat([value, variables], 1)
        value = self.batch_norm_value(value)
        value = self.value2(value)

        # save output for backpro
        action_prob = F.log_softmax(action_prob, dim=1)
        self.outputs.append(ModelOutput(action_prob.gather(-1, action), value))
        return action, value

    def get_action(self, state):
        action, _ = self.forward(*self.transform_input(state.screen, state.variables))
        return action

    def set_reward(self, reward):
        self.rewards.append(reward * 0.01)  # no clone() b/c of * 0.01

    def set_terminal(self, terminal):
        super(AdvantageActorCriticMap, self).set_terminal(terminal)
        self.discounts.append(self.discount * terminal)

    def backward(self):
        #
        # calculate step returns in reverse order
        #rewards = torch.stack(self.rewards, dim=0)
        rewards = self.rewards

        returns = torch.Tensor(len(rewards) - 1, *self.outputs[-1].value.size())
        step_return = self.outputs[-1].value.detach().cpu()
        for i in range(len(rewards) - 2, -1, -1):
            step_return.mul_(self.discounts[i]).add_(rewards[i])
            returns[i] = step_return

        returns = returns.to(device)
        #
        # calculate losses
        policy_loss = 0
        value_loss = 0
        steps = len(self.outputs) - 1
        for i in range(steps):
            advantage = returns[i] - self.outputs[i].value.detach()
            policy_loss += -self.outputs[i].log_action * advantage
            value_loss += F.smooth_l1_loss(self.outputs[i].value, returns[i])

        weights_l2 = 0
        for param in self.parameters():
            # TODO: check for non-grad params
            weights_l2 += param.norm(2)

        loss = policy_loss.mean() / steps + value_loss / steps + 0.00001 * weights_l2
        loss.backward()

        # reset state
        self.reset()
