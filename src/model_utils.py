#
# model.py, doom-net
#
# Created by Andrey Kolishchak on 10/29/17.
#

import os
import torch
from device import device
from model import Model
from aac import AdvantageActorCritic
from aac_lstm import AdvantageActorCriticLSTM
from aac_noisy import AdvantageActorCriticNoisy
#from aac_map import AdvantageActorCriticMap
#from aac_depth import AdvantageActorCriticDepth
#from ppo import PPO
#from ppo_map import PPOMap
from ppo_screen import PPOScreen
#from mcts_policy import MCTSPolicy
#from state_base import StateBase
#from es_base import ESBase
#from planner import Planner


def get_model(args):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_class = {
        'aac': AdvantageActorCritic,
        'aac_lstm': AdvantageActorCriticLSTM,
        'aac_noisy': AdvantageActorCriticNoisy,
        #'aac_depth': AdvantageActorCriticDepth,
        #'aac_map': AdvantageActorCriticMap,
        #'ppo': PPO,
        #'ppo_map': PPOMap,
        'ppo_screen': PPOScreen,
        #'mcts': MCTSPolicy,
        #'state': StateBase,
        #'es': ESBase,
        #'planner': Planner
    }

    #
    # if model class derived from nn.Module then convert it to the current device
    # and load parameters if needed
    if issubclass(model_class[args.model], torch.nn.Module):
        model = Model.create(model_class[args.model], args, args.load)
    else:
        model = model_class[args.model](args)

    return model



