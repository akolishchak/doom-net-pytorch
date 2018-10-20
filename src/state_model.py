#
# state_model.py, doom-net
#
# Created by Andrey Kolishchak on 04/29/18.
#
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from device import device
from lstm import LSTM
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicBlock, self).__init__()
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        padding = (0, 0)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        out = F.pad(x, self.padding, mode='replicate')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, self.padding, mode='replicate')
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Features(nn.Module):
    cnn_size = 256 * 4 * 5
    size = 256

    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(2, 2), dilation=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 2), dilation=(1, 1), padding=(0, 0))
        self.conv3 = BasicBlock(64, 128, stride=(2, 2), dilation=(1, 1), padding=(1, 1))
        self.conv4 = BasicBlock(128, 128, stride=(2, 2), dilation=(1, 1), padding=(1, 1))
        self.conv5 = BasicBlock(128, 256, stride=(2, 2), dilation=(1, 1), padding=(1, 1))
        self.conv6 = BasicBlock(256, 256, stride=(2, 2), dilation=(1, 1), padding=(1, 1))
        self.fc = nn.Linear(self.cnn_size + args.variable_num, self.size)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn_fc = nn.BatchNorm1d(self.size)

    def forward(self, screen, variables):
        cnn = F.pad(screen, (3, 3, 0, 0), mode='replicate')
        cnn = self.conv1(cnn)
        cnn = self.bn1(cnn)
        cnn = F.relu(cnn, inplace=True)
        cnn = F.pad(cnn, (1, 1, 0, 0), mode='replicate')
        cnn = self.conv2(cnn)
        cnn = self.bn2(cnn)
        cnn = F.relu(cnn, inplace=True)
        cnn = self.conv3(cnn)
        cnn = self.conv4(cnn)
        cnn = self.conv5(cnn)
        cnn = self.conv6(cnn)
        cnn = cnn.view(cnn.size(0), -1)
        features = torch.cat([cnn, variables], 1)
        features = self.fc(features)
        #features = self.bn_fc(features)
        #features = F.relu(features)

        return features


class StateModel(nn.Module):
    rnn_size = 256
    size = rnn_size

    def __init__(self, args):
        super().__init__()
        self.features = Features(args)
        self.rnn1 = LSTM(Features.size + args.button_num, self.rnn_size)
        self.rnn2 = LSTM(self.rnn_size, self.rnn_size)
        self.features_bn = nn.BatchNorm1d(Features.size + args.button_num)

        self.pred = nn.Linear(self.rnn_size, args.variable_num*3)

    def forward(self, observation, actions, cells=None):
        features = torch.cat([observation, actions], 1)
        #features = self.features_bn(features)

        h1, c1, h2, c2 = cells
        h1, c1 = self.rnn1(features, (h1, c1))
        h2, c2 = self.rnn2(h1, (h2, c2))

        pred = self.pred(h2)
        pred = pred.view(pred.shape[0], 3, -1)
        pred = F.log_softmax(pred, dim=-1)

        return [h1, c1, h2, c2], pred

    @staticmethod
    def set_nonterminal(cells, nonterminal):
        h1, c1, h2, c2 = cells
        nonterminal = nonterminal.view(-1, 1).expand_as(c1).to(device)
        return [cell * nonterminal for cell in cells]

    @staticmethod
    def reset(cells):
        if cells is not None:
            return [cell.detach() for cell in cells]
        else:
            return None

    @staticmethod
    def get_cells(batch_size):
        return [torch.zeros(batch_size, StateModel.rnn_size, device=device) for _ in range(4)]


def test():
    input = torch.rand(1, 3, 240, 320)

    class Args:
        button_num = 8
        variable_num = 7

    state_model = StateModel(Args())
    #output = state_model(input)
    torch.save(state_model.state_dict(), '../checkpoints/state_model_cp.pth')



#test()