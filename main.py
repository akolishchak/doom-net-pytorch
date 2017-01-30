#
# main.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import argparse
import os.path
from cuda import *
from aac import AdvantageActorCritic
from aac_lstm import AdvantageActorCriticLSTM
from doom_env import init_doom_env
from train import train
from test import test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Network')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--episode_size', type=int, default=10, help='number of steps in an episode')
    parser.add_argument('--batch_size', type=int, default=10, help='number of game instances running in parallel')
    parser.add_argument('--episode_num', type=int, default=1000, help='number of episodes for training')
    parser.add_argument('--episode_discount', type=float, default=0.95, help='number of episodes for training')
    parser.add_argument('--seed', type=int, default=1, help='seed value')
    parser.add_argument('--model', default='aac', choices=('aac', 'aac_lstm'), help='model to work with')
    parser.add_argument('--load', default=None, help='path to model file')
    parser.add_argument('--vizdoom_config', default='environments/basic.cfg', help='vizdoom config path')
    # parser.add_argument('--vizdoom_config', default='environments/rocket_basic.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=os.path.expanduser('~')+'/tools/ViZDoom/bin/vizdoom', help='path to vizdoom')
    parser.add_argument('--wad_path', default=os.path.expanduser('~')+'/tools/ViZDoom/scenarios/Doom2.wad', help='wad file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')

    args = parser.parse_args()
    init_doom_env(args)

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)

    if args.load is None or not os.path.isfile(args.load + '_model.pth'):
        model_class = {
            'aac': AdvantageActorCritic,
            'aac_lstm': AdvantageActorCriticLSTM
        }
        model = model_class[args.model](args)
    else:
        print("loading model {}".format(args.load))
        model = torch.load(args.load + '_model.pth')

    if USE_CUDA:
        model.cuda()

    train(args, model)

    test(args, model)
