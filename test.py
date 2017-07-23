#
# test.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
from doom_instance import *
from cuda import *


def test(args, model):
    print("testing...")
    model.eval()

    game = DoomInstance(
        args.vizdoom_config, args.wad_path, args.skiprate, visible=True, actions=args.action_set, bot_cmd=args.bot_cmd)
    step_state = game.get_state_normalized()

    state = NormalizedState(screen=None, depth=None, labels=None, variables=None)
    state.screen = torch.Tensor(1, *args.screen_size)
    state.variables = torch.Tensor(1, args.variable_num)

    while True:
        # convert state to torch tensors
        state.screen[0, :] = torch.from_numpy(step_state.screen)
        state.variables[0, :] = torch.from_numpy(step_state.variables)
        # compute an action
        action = model.get_action(state)
        # render
        step_state, _, finished = game.step_normalized(action[0][0])
        #img = step_state.automap.transpose(1, 2, 0)
        #img = step_state.labels
        #plt.imsave('map.jpeg', img)
        if finished:
            print("episode return: {}".format(game.get_episode_return()))
            model.set_terminal(torch.zeros(1))
