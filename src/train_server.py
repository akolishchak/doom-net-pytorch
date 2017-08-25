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
import torch.multiprocessing as mp


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
    action = torch.LongTensor(args.batch_size, 1)
    reward = torch.Tensor(args.batch_size, 1)
    terminal = torch.Tensor(args.batch_size)
    episode_return = torch.zeros(args.batch_size)

    state.screen.share_memory_()
    state.variables.share_memory_()
    action.share_memory_()
    reward.share_memory_()
    terminal.share_memory_()
    episode_return.share_memory_()

    counter = torch.zeros(1)
    counter.share_memory_()

    def instance(args, state, main_lock, main_event, event, id):
        game = DoomInstance(args.vizdoom_config, args.wad_path, args.skiprate, id,
                            actions=args.action_set, join=True, visible=False, color=id)
        first_pass = True
        while True:
            event.clear()
            if first_pass:
                first_pass = False
                normalized_state = game.get_state_normalized()
                state.screen[id, :] = torch.from_numpy(normalized_state.screen)
                state.variables[id, :] = torch.from_numpy(normalized_state.variables)
            else:
                normalized_state, step_reward, finished = game.step_normalized(action[id, 0])
                #normalized_state = game.get_state_normalized()
                state.screen[id, :] = torch.from_numpy(normalized_state.screen)
                state.variables[id, :] = torch.from_numpy(normalized_state.variables)
                reward[id, 0] = step_reward
                if finished:
                    episode_return[id] = float(game.get_episode_return())
                    # cut rewards from future actions
                    terminal[id] = 0
                else:
                    terminal[id] = 1
            # increase counter and wait main process
            with main_lock:
                counter[0] += 1
                if counter[0] >= args.batch_size:
                    main_event.set()
            event.wait()

    main_event = mp.Event()
    main_lock = mp.Lock()

    procs = []
    events = []
    #mp.set_start_method('spawn')
    for i in range(args.batch_size):
        event = mp.Event()
        p = mp.Process(target=instance, args=(args, state, main_lock, main_event, event, i))
        p.start()
        procs.append(p)
        events.append(event)
    main_event.wait()
    main_event.clear()
    counter[0] = 0

    # start training
    for episode in range(args.episode_num):
        batch_time = time.time()
        for step in range(args.episode_size):
            # get action
            action.copy_(model.get_action(state))
            # step
            for event in events:
                event.set()
            main_event.wait()
            main_event.clear()
            counter[0] = 0
            # get step info
            model.set_reward(reward)
            model.set_terminal(terminal)

        # update model
        model.backward()
        optimizer.step()
        optimizer.zero_grad()

        if episode % 1 == 0:
            print("{}: mean_return = {:f}, batch_time = {:.3f}".format(episode, episode_return.mean(), time.time()-batch_time))

        if episode % 500 == 0:
            torch.save(model, args.model + '_model_server_cp.pth')
            torch.save(optimizer, args.model + '_optimizer_server_cp.pth')

    # terminate games

    torch.save(model, args.model+'_model.pth')
    torch.save(optimizer, args.model+'_optimizer.pth')
