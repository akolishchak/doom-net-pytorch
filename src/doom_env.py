#
# doom_env.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import numpy as np
from vizdoom import *
from doom_instance import DoomInstance
from doom_instance_cig import DoomInstanceCig


def init_doom_env(args):
    if args.action_set == 'noset':
        args.action_set = []
    elif args.action_set is not None:
        args.action_set = np.load(args.action_set).tolist()

    instance_class = {
        'basic': DoomInstance,
        'cig': DoomInstanceCig
    }
    try:
        args.doom_instance
    except NameError:
        args.doom_instance = 'basic'
    args.instance_class = instance_class[args.doom_instance]

    doom = args.instance_class(
        args.vizdoom_config,
        wad=args.wad_path,
        skiprate=args.skiprate,
        visible=False,
        mode=Mode.PLAYER,
        actions=args.action_set)
    state = doom.get_state_normalized()

    args.button_num = doom.get_button_num()
    args.screen_size = state.screen.shape
    args.variable_num = len(state.variables)
    if state.variables is not None:
        args.variables_size = state.variables.shape
