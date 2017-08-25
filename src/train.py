#
# train.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
from multiprocessing.pool import ThreadPool
import time
import torch.optim as optim
from doom_instance import *
from cuda import *

def train(args, model):
    print("training...")
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.load is not None and os.path.isfile(args.load + '_optimizer.pth'):
        source_optimizer = torch.load(args.load+'_optimizer.pth')
        optimizer.load_state_dict(source_optimizer.state_dict())
        del source_optimizer

    optimizer.zero_grad()

    state = NormalizedState(screen=None, depth=None, labels=None, variables=None)
    state.screen = torch.Tensor(args.batch_size, *args.screen_size)
    state.variables = torch.Tensor(args.batch_size, args.variable_num)
    reward = torch.Tensor(args.batch_size, 1)
    terminal = torch.Tensor(args.batch_size, 1)
    episode_return = torch.zeros(args.batch_size)

    games = []
    for i in range(args.batch_size):
        games.append(DoomInstance(args.vizdoom_config, args.wad_path, args.skiprate, i, actions=args.action_set, bot_cmd=args.bot_cmd))

    pool = ThreadPool()

    def get_state(game):
        id = game.get_id()
        normalized_state = game.get_state_normalized()
        state.screen[id, :] = torch.from_numpy(normalized_state.screen)
        state.variables[id, :] = torch.from_numpy(normalized_state.variables)

    pool.map(get_state, games)
    # start training
    for episode in range(args.episode_num):
        batch_time = time.time()
        for step in range(args.episode_size):
            # get action
            action = model.get_action(state)
            # step and get new state
            def step_game(game):
                id = game.get_id()
                step_state, step_reward, finished = game.step_normalized(action[id][0])
                normalized_state = game.get_state_normalized()
                state.screen[id, :] = torch.from_numpy(normalized_state.screen)
                state.variables[id, :] = torch.from_numpy(normalized_state.variables)
                reward[id, 0] = step_reward
                if finished:
                    episode_return[id] = float(game.get_episode_return())
                    # cut rewards from future actions
                    terminal[id] = 0
                else:
                    terminal[id] = 1
            pool.map(step_game, games)
            model.set_reward(reward)
            model.set_terminal(terminal)

        # update model
        model.backward()
        optimizer.step()
        optimizer.zero_grad()

        if episode % 1 == 0:
            print("{}: mean_return = {:f}, batch_time = {:.3f}".format(episode, episode_return.mean(), time.time()-batch_time))

        if episode % args.checkpoint_rate == 0:
            torch.save(model, args.checkpoint_file)
            torch.save(optimizer, args.checkpoint_file+'_optimizer.pth')

    # terminate games
    pool.map(lambda game: game.release(), games)

    torch.save(model, args.checkpoint_file)
    torch.save(optimizer, args.checkpoint_file+'_optimizer.pth')
