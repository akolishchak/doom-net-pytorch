#
# base_model.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, screen_size, button_num):
        super(BaseModel, self).__init__()
        self.feature_num = 64
        self.conv1 = nn.Conv2d(in_channels=screen_size[0], out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)

        self.softmax = nn.Softmax2d()

        self.features = nn.Linear(self.feature_num * 14 * 19, self.feature_num)
        self.batch_norm = nn.BatchNorm1d(self.feature_num)
        self.action1 = nn.Linear(self.feature_num, 512)
        self.action2 = nn.Linear(512, button_num)

    def forward(self, input):
        # cnn
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        input = F.relu(self.conv3(input))
        input = F.relu(self.conv4(input))

        input = input.view(input.size(0), -1)

        # input = self.softmax(input / 1e-1)

        # features
        features = F.relu(self.batch_norm(self.features(input)))
        #features = F.relu(self.features(input))
        # action
        action = F.relu(self.action1(features))
        action = self.action2(action)
        return action
