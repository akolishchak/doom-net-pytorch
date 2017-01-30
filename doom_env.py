#
# doom_env.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from doom_instance import DoomInstance


def init_doom_env(args):
    doom = DoomInstance(args.vizdoom_config, args.wad_path, 1)
    state = doom.get_state()

    args.button_num = doom.get_button_num()
    args.screen_size = state.screen_buffer.shape
    if state.game_variables is not None:
        args.variables_size = state.game_variables.shape
