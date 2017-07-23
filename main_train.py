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
from aac_intrinsic import AdvantageActorCriticIntrinsic
from aac_duel import AdvantageActorCriticDuel
from aac_noisy import AdvantageActorCriticNoisy
from aac_big import AdvantageActorCriticBig
from doom_env import init_doom_env
from train import train
import vizdoom

if __name__ == '__main__':
    _vzd_path = os.path.dirname(vizdoom.__file__)
    parser = argparse.ArgumentParser(description='Doom Network')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--episode_size', type=int, default=20, help='number of steps in an episode')
    parser.add_argument('--batch_size', type=int, default=20, help='number of game instances running in parallel')
    parser.add_argument('--episode_num', type=int, default=150000, help='number of episodes for training')
    parser.add_argument('--episode_discount', type=float, default=0.95, help='number of episodes for training')
    parser.add_argument('--seed', type=int, default=1, help='seed value')
    parser.add_argument(
        '--model',
        default='aac',
        choices=('aac', 'aac_lstm', 'aac_intrinsic', 'aac_duel', 'aac_noisy', 'aac_big'),
        help='model to work with')
    parser.add_argument('--base_model', default=None, help='path to base model file')
    parser.add_argument('--action_set', default=None, help='model to work with')
    parser.add_argument('--load', default=None, help='path to model file')
    parser.add_argument('--vizdoom_config', default='environments/basic.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=_vzd_path, help='path to vizdoom')
    parser.add_argument('--wad_path', default=_vzd_path + '/doom2.wad', help='wad file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=1, help='number of frames per input')
    parser.add_argument('--checkpoint_file', default=None, help='check point file name')
    parser.add_argument('--checkpoint_rate', type=int, default=500, help='number of batches per checkpoit')
    parser.add_argument('--bot_cmd', default=None, help='command to launch a bot')
    args = parser.parse_args()
    print(args)
    init_doom_env(args)

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)

    model_class = {
        'aac': AdvantageActorCritic,
        'aac_lstm': AdvantageActorCriticLSTM,
        'aac_intrinsic': AdvantageActorCriticIntrinsic,
        'aac_duel': AdvantageActorCriticDuel,
        'aac_noisy': AdvantageActorCriticNoisy,
        'aac_big': AdvantageActorCriticBig
    }
    model = model_class[args.model](args)

    if args.load is not None and os.path.isfile(args.load):
        print("loading model parameters {}".format(args.load))
        source_model = torch.load(args.load)
        model.load_state_dict(source_model.state_dict())
        del source_model

    if USE_CUDA:
        model.cuda()

    train(args, model)