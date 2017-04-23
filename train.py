#
# train.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from multiprocessing.pool import ThreadPool
import time
import torch.optim as optim
from doom_instance import *
from cuda import *

def train(args, model):
    print("training...")
    model.train()

    if args.load is None or not os.path.isfile(args.load + '_optimizer.pth'):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.load(args.load+'_optimizer.pth')
    optimizer.zero_grad()

    state = NormalizedState(screen=None, depth=None, labels=None, variables=None)
    state.screen = torch.Tensor(args.batch_size, *args.screen_size)
    reward = torch.Tensor(args.batch_size, 1)
    episode_return = torch.zeros(args.batch_size)

    games = []
    for i in range(args.batch_size):
        games.append(DoomInstance(args.vizdoom_config, args.wad_path, args.skiprate, i, actions=args.action_set))

    pool = ThreadPool()

    def get_state(game):
        id = game.get_id()
        state.screen[id, :] = torch.from_numpy(game.get_state_normalized().screen)
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
                state.screen[id, :] = torch.from_numpy(game.get_state_normalized().screen)
                reward[id, 0] = step_reward
                if finished:
                    episode_return[id] = float(game.get_episode_return())
            pool.map(step_game, games)
            model.set_reward(reward)

        # update model
        model.backward()
        optimizer.step()
        optimizer.zero_grad()

        if episode % 1 == 0:
            print("{}: mean_return = {:f}, batch_time = {:.3f}".format(episode, episode_return.mean(), time.time()-batch_time))
    # terminate games
    pool.map(lambda game: game.release(), games)

    torch.save(model, args.model+'_model.pth')
    torch.save(optimizer, args.model+'_optimizer.pth')
