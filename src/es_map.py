#
# es_map.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

#F.relu = F.leaky_relu


class ESMap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.screen_feature_num = 256
        self.conv1 = nn.Conv2d(in_channels=args.screen_size[0]*args.frame_num, out_channels=64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=self.screen_feature_num, kernel_size=(1, 3), stride=1)

        layer1_size = 64
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size + args.variable_num, args.button_num)

        self.ln1 = nn.LayerNorm(self.screen_feature_num)

    def forward(self, screen, variables):
        # features
        features = self.conv1(screen)
        features = F.relu(features)
        features = self.conv2(features)
        features = F.relu(features)
        features = self.conv3(features)
        features = F.relu(features)
        features = self.conv4(features)
        features = features.view(features.size(0), -1)
        features = F.relu(features)

        # action
        action = F.relu(self.action1(features))
        action = torch.cat([action, variables], 1)
        action = self.action2(action)

        _, action = action.max(1, keepdim=True)

        return action

