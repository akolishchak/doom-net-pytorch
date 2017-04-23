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

    doom = DoomInstance(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set)
    state = doom.get_state()

    args.button_num = doom.get_button_num()
    args.screen_size = state.screen_buffer.shape
    #args.screen_size = (1, state.screen_buffer.shape[1], state.screen_buffer.shape[2])
    if state.game_variables is not None:
        args.variables_size = state.game_variables.shape
