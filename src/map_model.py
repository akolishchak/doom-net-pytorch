#
# map_model.py, doom-net
#
# Created by Andrey Kolishchak on 03/03/18.
#
import torch.nn as nn
import torch.nn.functional as F


class ObjectModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), dilation=(2, 1), padding=(0, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), dilation=(4, 1), padding=(0, 1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), dilation=(8, 1), padding=(0, 1))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), dilation=(6, 1), padding=(0, 1))
        self.conv6 = nn.Conv2d(256, 6, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

    def forward(self, screen):
        output = self.conv1(screen)
        output = self.bn1(output)
        output = F.relu(output, inplace=True)
        output = self.conv2(output)
        output = self.bn2(output)
        output = F.relu(output, inplace=True)
        output = self.conv3(output)
        output = self.bn3(output)
        output = F.relu(output, inplace=True)
        output = self.conv4(output)
        output = self.bn4(output)
        output = F.relu(output, inplace=True)
        output = self.conv5(output)
        output = self.bn5(output)
        output = F.relu(output, inplace=True)
        output = self.conv6(output)

        return output


class DistanceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), dilation=(2, 1), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), dilation=(4, 1), padding=(0, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), dilation=(8, 1), padding=(0, 1))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), dilation=(6, 1), padding=(0, 1))
        self.conv6 = nn.Conv2d(512, 129, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

    def forward(self, screen):
        output = self.conv1(screen)
        output = self.bn1(output)
        output = F.relu(output, inplace=True)
        output = self.conv2(output)
        output = self.bn2(output)
        output = F.relu(output, inplace=True)
        output = self.conv3(output)
        output = self.bn3(output)
        output = F.relu(output, inplace=True)
        output = self.conv4(output)
        output = self.bn4(output)
        output = F.relu(output, inplace=True)
        output = self.conv5(output)
        output = self.bn5(output)
        output = F.relu(output, inplace=True)
        output = self.conv6(output)

        return output


class ObjectDistanceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), dilation=(2, 1), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), dilation=(4, 1), padding=(0, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), dilation=(8, 1), padding=(0, 1))
        self.conv51 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), dilation=(6, 1), padding=(0, 1))
        self.conv52 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), dilation=(6, 1), padding=(0, 1))
        self.conv61 = nn.Conv2d(512, 6, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 1))
        self.conv62 = nn.Conv2d(512, 129, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn51 = nn.BatchNorm2d(512)
        self.bn52 = nn.BatchNorm2d(512)

    def forward(self, screen):
        shared = self.conv1(screen)
        shared = self.bn1(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv2(shared)
        shared = self.bn2(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv3(shared)
        shared = self.bn3(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv4(shared)
        shared = self.bn4(shared)
        shared = F.relu(shared, inplace=True)

        output1 = self.conv51(shared)
        output1 = self.bn51(output1)
        output1 = F.relu(output1, inplace=True)
        output1 = self.conv61(output1)

        output2 = self.conv52(shared)
        output2 = self.bn52(output2)
        output2 = F.relu(output2, inplace=True)
        output2 = self.conv62(output2)

        return output1, output2


class DistanceModel2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.drn = drn.drn_d_22(out_map=True, num_classes=1)
        self.fc1 = nn.Linear(1*30*40, 320)

    def forward(self, screen):
        output = self.drn(screen)
        output = output.view(output.size(0), -1)
        #output = F.relu(output)
        output = self.fc1(output)
        output = F.sigmoid(output)

        return output


class ObjectDistanceModel2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 3), bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), dilation=(2, 1), padding=(0, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), dilation=(4, 1), padding=(0, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), dilation=(8, 1), padding=(0, 1))
        self.conv51 = nn.Conv2d(512, 1024, kernel_size=3, stride=(2, 1), dilation=(6, 1), padding=(0, 1))
        self.conv52 = nn.Conv2d(512, 1024, kernel_size=3, stride=(2, 1), dilation=(6, 1), padding=(0, 1))
        self.conv61 = nn.Conv2d(1024, 6, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 1))
        self.conv62 = nn.Conv2d(1024, 129, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn51 = nn.BatchNorm2d(1024)
        self.bn52 = nn.BatchNorm2d(1024)

    def forward(self, screen):
        shared = self.conv1(screen)
        shared = self.bn1(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv2(shared)
        shared = self.bn2(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv3(shared)
        shared = self.bn3(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv4(shared)
        shared = self.bn4(shared)
        shared = F.relu(shared, inplace=True)

        output1 = self.conv51(shared)
        output1 = self.bn51(output1)
        output1 = F.relu(output1, inplace=True)
        output1 = self.conv61(output1)

        output2 = self.conv52(shared)
        output2 = self.bn52(output2)
        output2 = F.relu(output2, inplace=True)
        output2 = self.conv62(output2)

        return output1, output2


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


class ObjectDistanceModel3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv3 = BasicBlock(64, 128, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv4 = BasicBlock(128, 128, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv5 = BasicBlock(128, 256, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv6 = BasicBlock(256, 256, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv71 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv72 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv81 = nn.Conv2d(512, 6, kernel_size=3, stride=(1, 1), dilation=(1, 1), padding=(0, 0))
        self.conv82 = nn.Conv2d(512, 129, kernel_size=3, stride=(1, 1), dilation=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn71 = nn.BatchNorm2d(512)
        self.bn72 = nn.BatchNorm2d(512)

    def forward(self, screen):
        screen = F.pad(screen, (3, 3, 0, 0), mode='replicate')
        shared = self.conv1(screen)
        shared = self.bn1(shared)
        shared = F.relu(shared, inplace=True)
        shared = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        shared = self.conv2(shared)
        shared = self.bn2(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv3(shared)
        shared = self.conv4(shared)
        shared = self.conv5(shared)
        shared = self.conv6(shared)

        output1 = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        output1 = self.conv71(output1)
        output1 = self.bn71(output1)
        output1 = F.relu(output1, inplace=True)
        output1 = F.pad(output1, (1, 1, 0, 0), mode='replicate')
        output1 = self.conv81(output1)

        output2 = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        output2 = self.conv72(output2)
        output2 = self.bn72(output2)
        output2 = F.relu(output2, inplace=True)
        output2 = F.pad(output2, (1, 1, 0, 0), mode='replicate')
        output2 = self.conv82(output2)

        return output1, output2


class ObjectDistanceModel4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv3 = BasicBlock(64, 128, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv4 = BasicBlock(128, 128, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv5 = BasicBlock(128, 256, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv6 = BasicBlock(256, 256, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv71 = nn.Conv2d(256, 9, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv72 = nn.Conv2d(256, 65, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, screen):
        screen = F.pad(screen, (3, 3, 0, 0), mode='replicate')
        shared = self.conv1(screen)
        shared = self.bn1(shared)
        shared = F.relu(shared, inplace=True)
        shared = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        shared = self.conv2(shared)
        shared = self.bn2(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv3(shared)
        shared = self.conv4(shared)
        shared = self.conv5(shared)
        shared = self.conv6(shared)

        output1 = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        output1 = self.conv71(output1)

        output2 = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        output2 = self.conv72(output2)

        return output1, output2

'''
class MapModel(nn.Module):
    def __init__(self, args, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax
        #self.object_model = ObjectModel(args)
        #self.distance_model = DistanceModel(args)
        self.object_distance_model = ObjectDistanceModel4(args)

    def forward(self, screen):
        #objects = self.object_model(screen)
        #if self.use_softmax:
        #    objects = F.log_softmax(objects, dim=1)

        #distances = self.distance_model(screen)
        #if self.use_softmax:
        #    distances = F.log_softmax(distances, dim=1)
        objects, distances = self.object_distance_model(screen)
        if self.use_softmax:
            objects = F.log_softmax(objects, dim=1)
            distances = F.log_softmax(distances, dim=1)

        return objects, distances
'''


class MapModel(nn.Module):
    def __init__(self, args, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv3 = BasicBlock(64, 128, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv4 = BasicBlock(128, 128, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv5 = BasicBlock(128, 256, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv6 = BasicBlock(256, 256, stride=(2, 1), dilation=(1, 1), padding=(1, 1))
        self.conv71 = nn.Conv2d(256, 9, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.conv72 = nn.Conv2d(256, 65, kernel_size=3, stride=(2, 1), dilation=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, screen):
        screen = F.pad(screen, (3, 3, 0, 0), mode='replicate')
        shared = self.conv1(screen)
        shared = self.bn1(shared)
        shared = F.relu(shared, inplace=True)
        shared = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        shared = self.conv2(shared)
        shared = self.bn2(shared)
        shared = F.relu(shared, inplace=True)
        shared = self.conv3(shared)
        shared = self.conv4(shared)
        shared = self.conv5(shared)
        shared = self.conv6(shared)

        objects = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        objects = self.conv71(objects)

        distances = F.pad(shared, (1, 1, 0, 0), mode='replicate')
        distances = self.conv72(distances)

        if self.use_softmax:
            objects = F.log_softmax(objects, dim=1)
            distances = F.log_softmax(distances, dim=1)

        return objects, distances
