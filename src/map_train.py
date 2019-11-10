#
# map_train.py, doom-net
#
# Created by Andrey Kolishchak on 03/03/18.
#
import os
import time
import glob
import h5py
import datetime
import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from device import device
import argparse
from torch.utils.tensorboard import SummaryWriter
from map_model import MapModel
from focal_loss import FocalLoss
from doom_object import DoomObject


log_basedir = '../logs/{:%Y-%m-%d %H-%M-%S}/'.format(datetime.datetime.now())
train_writer = SummaryWriter(log_basedir + 'train')
val_writer = SummaryWriter(log_basedir + 'val')


class MapDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.length = 0
        for filename in glob.glob(os.path.join(h5_path, '*.hd5')):
            with h5py.File(filename, 'r') as file:
                self.length += len(file['screens'])
        #
        # hd5 has issues with fork at DataLoader, so file is opened on first getitem()
        # https://groups.google.com/forum/#!topic/h5py/bJVtWdFtZQM
        #
        self.screens = None
        self.distances = None
        self.objects = None
        self.indexes = None

    def __getitem__(self, index):
        if self.screens is None:
            self.screens = []
            self.distances = []
            self.objects = []
            self.indexes = []
            length = 0
            for filename in glob.glob(os.path.join(self.h5_path, '*.hd5')):
                file = h5py.File(filename, 'r')
                self.screens.append(file['screens'])
                self.distances.append(file['distances'])
                self.objects.append(file['objects'])
                length += len(file['screens'])
                self.indexes.append(length)

        file_num = bisect.bisect(self.indexes, index)
        offset = index - self.indexes[file_num - 1] if file_num > 0 else index
        return self.screens[file_num][offset], self.distances[file_num][offset], self.objects[file_num][offset]

    def __len__(self):
        return self.length


width = 160
half_width = width / 2
tan = np.ndarray(shape=(width,), dtype=float)
fov = np.pi/2.0
ratio = math.tan(0.5 * fov) / half_width
for i in range(width):
    tan[i] = (i - half_width) * ratio


def draw(distance, objects, file_name):
    distance = distance.view(-1).cpu().numpy()
    objects = objects.view(-1).cpu().numpy()

    distance = np.around(distance / 4.0)
    distance[distance > 15] = 15

    screen = np.zeros([DoomObject.Type.MAX, 16, 32], dtype=np.float32)
    x = np.around(16 + tan * distance).astype(int)
    y = np.around(distance).astype(int)
    todelete = np.where(y == 15)
    y = np.delete(y, todelete, axis=0)
    x = np.delete(x, todelete, axis=0)
    channels = np.delete(objects, todelete, axis=0)
    screen[channels, y, x] = 1

    img = screen[[8, 7, 6], :]
    img = img.transpose(1, 2, 0)
    plt.imsave(file_name, img)


def test(model, data_loader):
    model.eval()

    epoch_loss_obj = 0
    epoch_loss_dist = 0
    epoch_accuracy_obj = 0
    epoch_accuracy_dist = 0
    batch = 0
    for batch, (screens, distances, objects) in enumerate(data_loader):
        screens, distances, objects = screens.to(device), distances.to(device), objects.to(device)

        pred_objects, pred_distances = model(screens)
        loss_obj = objects_criterion(pred_objects, objects)
        loss_dist = distances_criterion(pred_distances, distances)

        epoch_loss_obj += loss_obj.item()
        epoch_loss_dist += loss_dist.item()

        _, pred_objects = pred_objects.max(1)
        accuracy = (pred_objects == objects).float().mean()
        epoch_accuracy_obj += accuracy

        _, pred_distances = pred_distances.max(1)
        accuracy = (pred_distances == distances).float().mean()
        epoch_accuracy_dist += accuracy

    batch_num = batch + 1
    epoch_loss_obj /= batch_num
    epoch_loss_dist /= batch_num
    epoch_accuracy_obj /= batch_num
    epoch_accuracy_dist /= batch_num

    model.train()
    return (epoch_loss_obj, epoch_loss_dist), (epoch_accuracy_obj, epoch_accuracy_dist)

#objects_criterion = nn.NLLLoss2d()
#distances_criterion = nn.NLLLoss2d()
objects_criterion = FocalLoss(alfa=0.25, gamma=2)
distances_criterion = FocalLoss(alfa=0.25, gamma=2)

def train(args):

    train_set = MapDataset(os.path.join(args.h5_path, 'train'))
    train_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size, shuffle=True)

    test_set = MapDataset(os.path.join(args.h5_path, 'test'))
    test_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=10, shuffle=False)

    validation_set = MapDataset(os.path.join(args.h5_path, 'val'))
    validation_data_loader = DataLoader(dataset=validation_set, num_workers=4, batch_size=10, shuffle=False)

    model = MapModel(args).to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    if args.load is not None and os.path.isfile(args.load):
        print("loading model parameters {}".format(args.load))
        state_dict = torch.load(args.load)
        model.load_state_dict(state_dict)
        optimizer_dict = torch.load(args.load + '_optimizer.pth')
        optimizer.load_state_dict(optimizer_dict)

    for epoch in range(args.epoch_num):
        epoch_loss_obj = 0
        epoch_loss_dist = 0
        epoch_accuracy_obj = 0
        epoch_accuracy_dist = 0
        running_loss_obj = 0
        running_loss_dist = 0
        running_accuracy_obj = 0
        running_accuracy_dist = 0
        batch_time = time.time()
        batch = 0
        for batch, (screens, distances, objects) in enumerate(train_data_loader):
            screens, distances, objects = screens.to(device), distances.to(device), objects.to(device)

            #for i in range(len(distances)):
            #    draw(distances[i], objects[i], 'view-image.png')

            optimizer.zero_grad()

            pred_objects, pred_distances = model(screens)

            loss_obj = objects_criterion(pred_objects, objects)
            loss_dist = distances_criterion(pred_distances, distances)
            loss = loss_obj + loss_dist
            loss.backward()
            optimizer.step()

            running_loss_obj += loss_obj.item()
            running_loss_dist += loss_dist.item()
            epoch_loss_obj += loss_obj.item()
            epoch_loss_dist += loss_dist.item()

            _, pred_objects = pred_objects.max(1)
            accuracy = (pred_objects == objects).float().mean()
            running_accuracy_obj += accuracy
            epoch_accuracy_obj += accuracy

            _, pred_distances = pred_distances.max(1)
            accuracy = (pred_distances == distances).float().mean()
            running_accuracy_dist += accuracy
            epoch_accuracy_dist += accuracy

            if batch % 1000 == 999:
                torch.save(model.state_dict(), args.checkpoint_file)
                torch.save(optimizer.state_dict(), args.checkpoint_file + '_optimizer.pth')

            batches_per_print = 10
            if batch % batches_per_print == batches_per_print-1:  # print every batches_per_print mini-batches
                running_loss_obj /= batches_per_print
                running_loss_dist /= batches_per_print
                running_accuracy_obj /= batches_per_print
                running_accuracy_dist /= batches_per_print
                print(
                    '[{:d}, {:5d}] loss: {:.3f}, {:.3f}, accuracy: {:.3f}, {:.3f}, time: {:.6f}'.format(
                        epoch + 1, batch + 1, running_loss_obj, running_loss_dist,
                        running_accuracy_obj, running_accuracy_dist, (time.time()-batch_time)/batches_per_print
                    )
                )
                running_loss_obj, running_loss_dist = 0, 0
                running_accuracy_obj, running_accuracy_dist = 0, 0
                batch_time = time.time()

        batch_num = batch + 1
        epoch_loss_obj /= batch_num
        epoch_loss_dist /= batch_num
        epoch_accuracy_obj /= batch_num
        epoch_accuracy_dist /= batch_num

        if epoch % args.checkpoint_rate == args.checkpoint_rate - 1:
            torch.save(model.state_dict(), args.checkpoint_file)
            torch.save(optimizer.state_dict(), args.checkpoint_file + '_optimizer.pth')

        val_loss, val_accuracy = test(model, validation_data_loader)

        print('[{:d}] TRAIN loss: {:.3f}, {:.3f} accuracy: {:.3f}, {:.3f}, VAL loss: {:.3f}, {:.3f}, accuracy: {:.3f}, {:.3f}'.format(
            epoch + 1, epoch_loss_obj, epoch_loss_dist, epoch_accuracy_obj, epoch_accuracy_dist,
            *val_loss, *val_accuracy
        ))

        train_writer.add_scalar('map/loss_obj', epoch_loss_obj, epoch)
        train_writer.add_scalar('map/loss_dist', epoch_loss_dist, epoch)
        train_writer.add_scalar('map/accuracy_obj', epoch_accuracy_obj, epoch)
        train_writer.add_scalar('map/accuracy_dist', epoch_accuracy_dist, epoch)
        val_writer.add_scalar('map/loss_obj', val_loss[0], epoch)
        val_writer.add_scalar('map/loss_dist', val_loss[1], epoch)
        val_writer.add_scalar('map/accuracy_obj', val_accuracy[0], epoch)
        val_writer.add_scalar('map/accuracy_dist', val_accuracy[1], epoch)

    test_loss, test_accuracy = test(model, test_data_loader)
    print('[TEST] loss: {:.3f}, {:.3f}, accuracy: {:.3f}, {:.3f}'.format(*test_loss, *test_accuracy))


def run(args):
    #model = torch.load(args.checkpoint_file) #MapModel(args)
    model = MapModel(args).to(device)
    state_dict = torch.load(args.checkpoint_file)
    model.load_state_dict(state_dict)
    model.eval()

    test_set = MapDataset(os.path.join(args.h5_path, 'test'))
    test_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=10, shuffle=True)

    for batch, (screens, distances, objects) in enumerate(test_data_loader):
        screens, distances, objects = screens.to(device), distances.to(device), objects.to(device)

        pred_objects, pred_distances = model(screens)
        _, pred_objects = pred_objects.max(1)
        _, pred_distances = pred_distances.max(1)

        for i in range(len(distances)):
            draw(distances[i], objects[i], 'view-image-label.png')
            draw(pred_distances[i], pred_objects[i], 'view-image-pred.png')
            print(1)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Map View Trainer')
    parser.add_argument('--batch_size', type=int, default=25, help='number of game instances running in parallel')
    parser.add_argument('--epoch_num', type=int, default=50, help='number of game instances running in parallel')
    parser.add_argument('--load', default=None, help='path to model file')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/oblige', help='hd5 file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--checkpoint_file', default='../checkpoints/map_oblige_cp.pth', help='check point file name')
    parser.add_argument('--checkpoint_rate', type=int, default=1, help='number of batches per checkpoit')
    args = parser.parse_args()

    train(args)

    #run(args)