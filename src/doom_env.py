#
# doom_env.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import numpy as np
from doom_instance import DoomInstance


def init_doom_env(args):
    if args.action_set is not None:
        args.action_set = np.load(args.action_set).tolist()

    doom = DoomInstance(
        args.vizdoom_config,
        wad=args.wad_path,
        skiprate=args.skiprate,
        id=None,
        visible=False,
        actions=args.action_set)
    state = doom.get_state_normalized()

    args.button_num = doom.get_button_num()
    args.screen_size = state.screen.shape
    args.variable_num = len(state.variables)
    if state.variables is not None:
        args.variables_size = state.variables.shape
