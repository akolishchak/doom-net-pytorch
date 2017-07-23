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
from doom_env import init_doom_env
from train_server import train
from test import test
import vizdoom

if __name__ == '__main__':
    _vzd_path = os.path.dirname(vizdoom.__file__)
    parser = argparse.ArgumentParser(description='Doom Network')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--episode_size', type=int, default=30, help='number of steps in an episode')
    parser.add_argument('--batch_size', type=int, default=10, help='number of game instances running in parallel')
    parser.add_argument('--episode_num', type=int, default=15000, help='number of episodes for training')
    parser.add_argument('--episode_discount', type=float, default=0.97, help='number of episodes for training')
    parser.add_argument('--seed', type=int, default=1, help='seed value')
    parser.add_argument('--model', default='aac', choices=('aac', 'aac_lstm', 'aac_intrinsic', 'aac_duel'), help='model to work with')
    parser.add_argument('--base_model', default='aac_model_server_cp_start.pth', help='path to base model file')
    parser.add_argument('--action_set', default='action_set_speed_shot_backward_right.npy', help='model to work with')
    parser.add_argument('--load', default=None, help='path to model file')
    #parser.add_argument('--vizdoom_config', default='environments/basic.cfg', help='vizdoom config path')
    #parser.add_argument('--vizdoom_config', default='environments/rocket_basic.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_config', default='environments/cig_server.cfg', help='vizdoom config path')
    #parser.add_argument('--vizdoom_config', default='environments/deathmatch.cfg', help='vizdoom config path')
    # parser.add_argument('--vizdoom_config', default='environments/D3_battle.cfg', help='vizdoom config path')
    #parser.add_argument('--vizdoom_config', default='environments/health_gathering.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=_vzd_path, help='path to vizdoom')
    parser.add_argument('--wad_path', default=_vzd_path + '/doom2.wad', help='wad file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--frame_num', type=int, default=4, help='number of frames per input')
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
        'aac_duel': AdvantageActorCriticDuel
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
    """
    model.share_memory()
    processes = []
    mp.set_start_method('spawn')
    for rank in range(2):
        p = mp.Process(target=test, args=(args, model), )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    """