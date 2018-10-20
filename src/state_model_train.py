#
# state_model_train.py, doom-net
#
# Created by Andrey Kolishchak on 04/29/18.
#
import argparse
import os.path
from model_utils import get_model
from doom_env import init_doom_env
import vizdoom


if __name__ == '__main__':
    _vzd_path = os.path.dirname(vizdoom.__file__)
    parser = argparse.ArgumentParser(description='Doom Network - State Model Training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--episode_size', type=int, default=20, help='number of steps in an episode')
    parser.add_argument('--batch_size', type=int, default=20, help='number of game instances running in parallel')
    parser.add_argument('--episode_num', type=int, default=150000, help='number of episodes for training')
    parser.add_argument('--epoch_game_steps', type=int, default=10000, help='number of steps per epoch')
    parser.add_argument('--action_set', default=None, help='action set')
    parser.add_argument('--load', default=None, help='path to model file')
    parser.add_argument('--doom_instance', default='basic', choices=('basic', 'cig', 'map', 'obj'), help='doom instance type')
    parser.add_argument('--vizdoom_config', default='environments/basic.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=_vzd_path, help='path to vizdoom')
    parser.add_argument('--wad_path', default=_vzd_path + '/freedoom2.wad', help='wad file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--checkpoint_file', default=None, help='check point file name')
    parser.add_argument('--checkpoint_rate', type=int, default=500, help='number of batches per checkpoit')
    parser.add_argument('--h5_path', default=None, help='hd5 files path')
    args = parser.parse_args()
    print(args)
    init_doom_env(args)
