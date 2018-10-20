#
# aac_state_base.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import datetime
import glob
from multiprocessing.pool import ThreadPool
from threading import Thread
import time
import h5py
import bisect
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from device import device
from model import Model
from state_model import StateModel
from state_controller import AdvantageActorCriticController
import vizdoom


class StateBase:
    def __init__(self, args):
        super().__init__()

    def run_train(self, args):

        for iter in range(100):
            self.generate_data(args)
            self.train_state_model(args)
            self.train_controller(args)

    def generate_data(self, args):
        print("Generate data...")

        def worker(id, args):
            state_model = Model.create(StateModel, args, args.state_model)
            state_model.eval()
            controller = Model.create(AdvantageActorCriticController, args, args.checkpoint_file)
            controller.eval()
            new_controller = not os.path.isfile(args.checkpoint_file)

            game = args.instance_class(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set, id=id)
            state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
            state.screen = torch.Tensor(1, *args.screen_size)
            state.variables = torch.Tensor(1, args.variable_num)
            action_onehot = torch.zeros(1, args.button_num, device=device)
            cells = StateModel.get_cells(1)
            episode_num = 2
            max_step = 1000

            for episode in range(episode_num):
                step_state = game.get_state_normalized()
                episode_screens = []
                episode_variables = []
                episode_actions = []
                episode_vars = []
                action = 0
                for step in range(max_step):
                    # convert state to torch tensors
                    state.screen[0, :] = torch.from_numpy(step_state.screen)
                    state.variables[0, :] = torch.from_numpy(step_state.variables)
                    # compute an action
                    if not new_controller:
                        with torch.set_grad_enabled(False):
                            observation = state_model.features(state.screen.to(device), state.variables.to(device))
                            action = controller.forward(observation, cells[-2])
                            cells, pred = state_model(observation, action_onehot.zero_().scatter_(-1, action, 1), cells)
                    else:
                        action = torch.randint(0, args.button_num, (1, 1), dtype=torch.long, device=device)
                        action_onehot.zero_().scatter_(-1, action, 1)

                    episode_screens.append(step_state.screen)
                    episode_variables.append(step_state.variables)
                    episode_actions.append(action_onehot.cpu().numpy()[0])
                    # render
                    step_state, _, finished, vars = game.step_normalized(action[0, 0])
                    episode_vars.append(vars)
                    if finished:
                        print("episode return: {}".format(game.get_episode_return()))
                        cells = state_model.set_nonterminal(cells, torch.zeros(1, 1))
                        break
                #
                # save episodes data to file
                #
                filename = os.path.join(args.h5_path,
                                        '{:%Y-%m-%d %H-%M-%S}-{}-{}.hd5'.format(datetime.datetime.now(), id, episode))
                print(filename)
                file = h5py.File(filename, 'w')
                file.create_dataset('screens', data=episode_screens, dtype='float32', compression='gzip')
                file.create_dataset('variables', data=episode_variables, dtype='float32', compression='gzip')
                file.create_dataset('actions', data=episode_actions, dtype='float32', compression='gzip')
                file.create_dataset('vars', data=episode_vars, dtype='float32', compression='gzip')

                game.new_episode()

        threads = []
        for i in range(5):
            thread = Thread(target=worker, args=(i, args))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def train_state_model(self, args):
        print("Train state model...")
        state_model = Model.create(StateModel, args, args.state_model)
        state_model.train()

        def data_generator(args):
            batch_size = args.batch_size
            episode_size = args.episode_size
            screens = []
            variables = []
            actions = []
            vars = []

            for filename in glob.glob(os.path.join(args.h5_path, '*.hd5')):
                file = h5py.File(filename, 'r')
                screens.append(file['screens'])
                variables.append(file['variables'])
                actions.append(file['actions'])
                vars.append(file['vars'])
            #
            episodes_num = len(screens)
            step_screens = np.ndarray(shape=(batch_size, *screens[0].shape[1:]), dtype=np.float32)
            step_variables = np.ndarray(shape=(batch_size, *variables[0].shape[1:]), dtype=np.float32)
            step_actions = np.ndarray(shape=(batch_size, *actions[0].shape[1:]), dtype=np.float32)
            step_vars = np.ndarray(shape=(batch_size, *vars[0].shape[1:]), dtype=np.int)
            step_nonterminals = np.ones(shape=(batch_size, 1), dtype=np.float32)

            # select episodes for the initial batch
            batch_episodes = np.random.randint(episodes_num, size=batch_size)
            batch_episodes_length = np.array([len(actions[episode]) for episode in batch_episodes])
            batch_episodes_step = np.zeros(batch_size, dtype=np.int)
            iter_num = batch_episodes_length.mean().astype(np.int)*episodes_num//batch_size
            for iter in range(iter_num):
                for i in range(batch_size):
                    episode = batch_episodes[i]
                    step = batch_episodes_step[i]
                    length = batch_episodes_length[i]

                    step_screens[i, :] = screens[episode][step]
                    step_variables[i, :] = variables[episode][step]
                    step_actions[i, :] = actions[episode][step]
                    step_vars[i, :] = vars[episode][step]+1
                    batch_episodes_step[i] += 1
                    if batch_episodes_step[i] >= length:
                        step_nonterminals[i] = 0.0
                        # reached terminal state, select a new episode
                        episode = np.random.randint(episodes_num)
                        batch_episodes[i] = episode
                        batch_episodes_step[i] = 0
                    else:
                        if step_variables[i, -1] == 0:
                            step_nonterminals[i] = 1.0
                        else:
                            step_nonterminals[i] = 0.0

                yield torch.from_numpy(step_screens), \
                      torch.from_numpy(step_variables), \
                      torch.from_numpy(step_actions), \
                      torch.from_numpy(step_vars), \
                      torch.from_numpy(step_nonterminals)

        training_data_loader = data_generator(args)

        optimizer = optim.Adam(state_model.parameters(), lr=5e-4, weight_decay=1e-4, amsgrad=True)

        cells = StateModel.get_cells(args.batch_size)

        epoch_num = 1

        for epoch in range(epoch_num):
            mean_loss = 0
            mean_accuracy = 0
            updates = 0
            batch_time = time.time()

            for batch, (screens, variables, actions, vars, nonterminals) in enumerate(training_data_loader):
                screens, variables, actions, vars, nonterminals = \
                    screens.to(device), variables.to(device), actions.to(device), vars.to(device), nonterminals.to(device)

                observation = state_model.features(screens, variables)
                cells, pred = state_model(observation, actions, cells)
                cells = state_model.set_nonterminal(cells, nonterminals)

                loss = F.nll_loss(pred, vars)
                mean_loss += loss.item()
                updates += 1

                _, pred_vars = pred.max(1)
                mean_accuracy += (pred_vars == vars).float().mean()

                if batch % args.episode_size == args.episode_size - 1:
                    loss.backward()

                    grads = []
                    weights = []
                    for p in state_model.parameters():
                        if p.grad is not None:
                            grads.append(p.grad.data.view(-1))
                            weights.append(p.data.view(-1))
                    grads = torch.cat(grads, 0)
                    grads_norm = grads.norm()
                    weights = torch.cat(weights, 0)
                    weights_norm = weights.norm()

                    assert grads_norm == grads_norm

                    optimizer.step()
                    optimizer.zero_grad()
                    cells = state_model.reset(cells)

                    mean_loss /= updates
                    mean_accuracy /= updates
                    print("episode loss = {:f}, accuracy = {:f}, grads_norm = {:f}, weights_norm = {:f} train_time = {:.3f}".format(mean_loss, mean_accuracy, grads_norm, weights_norm, time.time() - batch_time))
                    mean_loss = 0
                    mean_accuracy = 0
                    updates = 0
                    batch_time = time.time()

                if batch >= 5000:
                    break

        torch.save(state_model.state_dict(), args.state_model)

    def train_controller(self, args):
        print("Controller training...")
        controller = Model.create(AdvantageActorCriticController, args) #, args.load)
        controller.train()

        optimizer = optim.Adam(controller.parameters(), lr=5e-4, amsgrad=True)
        #if args.load is not None and os.path.isfile(args.load + '_optimizer.pth'):
        #    optimizer_dict = torch.load(args.load+'_optimizer.pth')
        #    optimizer.load_state_dict(optimizer_dict)

        assert args.state_model is not None
        state_model = Model.create(StateModel, args, args.state_model)
        state_model.eval()

        optimizer.zero_grad()

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(args.batch_size, *args.screen_size)
        state.variables = torch.Tensor(args.batch_size, args.variable_num)
        vars = torch.Tensor(args.batch_size, args.variable_num).long()
        reward = torch.Tensor(args.batch_size, 1)
        nonterminal = torch.Tensor(args.batch_size, 1)
        action_onehot = torch.zeros(args.batch_size, len(args.action_set), device=device)
        cells = StateModel.get_cells(args.batch_size)

        games = []
        for i in range(args.batch_size):
            games.append(args.instance_class(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set, id=i))

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
            rewards = []
            nonterminals = []
            episode_return = 0
            episode_accuracy = 0
            for step in range(args.episode_size):
                # get action
                with torch.set_grad_enabled(False):
                    observation = state_model.features(state.screen.to(device), state.variables.to(device))
                action = controller.forward(observation, cells[-2])
                with torch.set_grad_enabled(False):
                    cells, pred = state_model(observation, action_onehot.zero_().scatter_(-1, action, 1), cells)
                action = action.cpu()
                #print(action.squeeze())
                # step and get new state
                def step_game(game):
                    id = game.get_id()
                    normalized_state, step_reward, finished, step_vars = game.step_normalized(action[id, 0])
                    state.screen[id, :] = torch.from_numpy(normalized_state.screen)
                    state.variables[id, :] = torch.from_numpy(normalized_state.variables)
                    reward[id, 0] = step_reward
                    vars[id] = torch.from_numpy(step_vars+1)
                    if finished:
                        #episode_return[id] = float(game.get_episode_return())
                        # cut rewards from future actions
                        nonterminal[id] = 0
                    else:
                        nonterminal[id] = 1
                pool.map(step_game, games)
                #rewards.append(reward.clone())
                # mse as reward for exploration policy
                _, pred_vars = pred.max(1)
                episode_accuracy += (pred_vars == vars.to(device)).float().mean()
                exploration_reward = F.nll_loss(pred, vars.to(device), reduce=False).mean(dim=-1)
                exploration_reward = exploration_reward[:, None].cpu()*0.1
                episode_return += exploration_reward.mean()
                rewards.append(exploration_reward)
                noterminal_copy = nonterminal.clone()
                nonterminals.append(noterminal_copy)
                cells = state_model.set_nonterminal(cells, noterminal_copy)

            # update model
            controller.backward(rewards, nonterminals)

            grads = []
            weights = []
            for p in controller.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                    weights.append(p.view(-1))
            grads = torch.cat(grads, 0)
            weights = torch.cat(weights, 0)
            grads_norm = grads.norm()
            weights_norm = weights.norm()

            assert grads_norm == grads_norm

            optimizer.step()
            optimizer.zero_grad()

            episode_accuracy /= args.episode_size
            episode_return /= args.episode_size

            if episode % 1 == 0:
                print("{}: mean_return = {:f}, mean_accuracy= {:f}, grads_norm = {:f}, weights_norm = {:f}, batch_time = {:.3f}".format(episode, episode_return, episode_accuracy, grads_norm, weights_norm, time.time()-batch_time))

            if episode % args.checkpoint_rate == 0:
                torch.save(controller.state_dict(), args.checkpoint_file)
                #torch.save(optimizer.state_dict(), args.checkpoint_file+'_optimizer.pth')

        # terminate games
        pool.map(lambda game: game.release(), games)

        torch.save(controller.state_dict(), args.checkpoint_file)
        #torch.save(optimizer.state_dict(), args.checkpoint_file+'_optimizer.pth')

    def run_test(self, args):
        print("testing...")
        controller = Model.create(AdvantageActorCriticController, args, args.load)
        controller.eval()

        assert args.state_model is not None
        state_model = Model.create(StateModel, args, args.state_model)
        state_model.eval()

        game = args.instance_class(
            args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.ASYNC_PLAYER, actions=args.action_set)
        step_state = game.get_state_normalized()

        state = args.instance_class.NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(1, *args.screen_size)
        state.variables = torch.Tensor(1, args.variable_num)
        action_onehot = torch.zeros(1, len(args.action_set), device=device)
        cells = StateModel.get_cells(1)

        while True:
            # convert state to torch tensors
            state.screen[0, :] = torch.from_numpy(step_state.screen)
            state.variables[0, :] = torch.from_numpy(step_state.variables)
            # compute an action
            with torch.set_grad_enabled(False):
                observation = state_model.features(state.screen.to(device), state.variables.to(device))
                action = controller.forward(observation, cells[-2])
                cells, pred = state_model(observation, action_onehot.zero_().scatter_(-1, action, 1), cells)
            action = action.cpu()
            print(action)
            # render
            step_state, _, finished, _ = game.step_normalized(action[0, 0])
            if finished:
                print("episode return: {}".format(game.get_episode_return()))
                cells = state_model.set_nonterminal(torch.zeros(1, 1))
