#
# model.py, doom-net
#
# Created by Andrey Kolishchak on 10/29/17.
#

import os
import torch
from device import device
from aac import AdvantageActorCritic
from aac_lstm import AdvantageActorCriticLSTM
from aac_noisy import AdvantageActorCriticNoisy
from aac_map import AdvantageActorCriticMap
from aac_depth import AdvantageActorCriticDepth


def get_model(args):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_class = {
        'aac': AdvantageActorCritic,
        'aac_lstm': AdvantageActorCriticLSTM,
        'aac_noisy': AdvantageActorCriticNoisy,
        'aac_depth': AdvantageActorCriticDepth,
        'aac_map': AdvantageActorCriticMap
    }
    model = model_class[args.model](args).to(device)

    if args.load is not None and os.path.isfile(args.load):
        print("loading model parameters {}".format(args.load))
        state_dict = torch.load(args.load)
        model.load_state_dict(state_dict)

    return model



