#
# mcts_base.py, doom-net
#
# Created by Andrey Kolishchak on 04/29/18.
#
import os
import time
import datetime
from torch.multiprocessing import Process
import numpy as np
import h5py
from simulator import Simulator
from mcts import MCTS
from mcts_dataset import MCTSDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from device import device
from model import Model
import vizdoom


class MCTSBase(Model):
    def __init__(self):
        super().__init__()

    def run_train(self, args):
        print("training...")

        model = self
        sim = Simulator(model)

        games = []
        for i in range(1):
            games.append(
                args.instance_class(args.vizdoom_config, args.wad_path, args.skiprate, actions=args.action_set, id=i)
            )

        for iter in range(100):
            print("iteration: ", iter)
            #
            # generate data
            #
            processes = []
            for game in games:
                process = Process(target=self.generate_data, args=(game, sim, args))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
            #
            # train model with new data
            #
            self.train_model(model)

    def run_test(self, args):
        print("testing...")
        model = self
        sim = Simulator(model)

        model.eval()

        game = args.instance_class(
            args.vizdoom_config, args.wad_path, args.skiprate, visible=True, mode=vizdoom.Mode.ASYNC_PLAYER,
            actions=args.action_set)
        step_state = game.get_state_normalized()

        while True:
            state = sim.get_state(step_state)
            # compute an action
            action = sim.get_action(state)
            # render
            step_state, _, finished = game.step_normalized(action[0][0])
            if finished:
                print("episode return: {}".format(game.get_episode_return()))

    def generate_data(self, game, sim, args, episode_num=100):
        model = sim.get_policy_model()
        model.eval()

        target_states, target_actions, target_rewards = [], [], []
        mean_reward = 0
        for i in range(episode_num):
            states, actions, rewards = self.get_episode_targets(game, sim, 10)
            #
            target_states.extend(states)
            target_actions.extend(actions)
            target_rewards.extend(rewards)
            #
            mean_reward += sum(rewards)/len(rewards)
        #
        # save episodes data to file
        #
        filename = os.path.join(args.h5_path, '{:%Y-%m-%d %H-%M-%S}-{}'.format(datetime.datetime.now(), i))
        file = h5py.File(filename, 'w')
        file.create_dataset('states', data=target_states, dtype='float32', compression='gzip')
        file.create_dataset('actions', data=target_actions, dtype='long', compression='gzip')
        file.create_dataset('rewards', data=target_rewards, dtype='float32', compression='gzip')

        mean_reward /= episode_num
        print("mean reward = ", mean_reward)

    def get_episode_targets(self, game, sim, max_length):
        mcts = MCTS(sim, 1000, c_puct=1)
        target_states, target_actions, target_rewards = [], [], []
        step = 0
        state = game.get_state_normalized()

        while step < max_length:
            prob, state = mcts.get_action_prob(state, 1)
            action = np.random.choice(len(prob), p=prob)
            target_states.append(state.policy_state)
            target_actions.append(action)
            state, reward, finished = game.step_normalized(action)
            target_rewards.append(reward)
            if finished:
                break
            step += 1

        return target_states, target_actions, target_rewards

    def train_model(self, model, args, epoch_num=10):
        dataset = MCTSDataset(args)
        training_data_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=args.batch_size, shuffle=True)

        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4, amsgrad=True)

        mean_value_loss = 0
        mean_policy_loss = 0
        mean_accuracy = 0
        updates = 0

        batch_time = time.time()
        for epoch in range(epoch_num):
            for batch, (state, target_action, target_value) in enumerate(training_data_loader):
                state, target_action, target_value = state.to(device), target_action.to(device), target_value.to(device)

                optimizer.zero_grad()
                value, log_action = model(state)
                value_loss = F.mse_loss(value, target_value[:, None])
                policy_loss = F.nll_loss(log_action, target_action)
                loss = value_loss + policy_loss

                loss.backward()
                optimizer.step()

                grads = []
                weights = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.data.view(-1))
                        weights.append(p.data.view(-1))
                grads = torch.cat(grads, 0)
                weights = torch.cat(weights, 0)
                grads_norm = grads.norm()
                weights_norm = weights.norm()

                assert grads_norm == grads_norm

                _, pred_action = log_action.max(1)
                accuracy = (pred_action == target_action.data).float().mean()

                if epoch == epoch_num - 1:
                    mean_value_loss += value_loss.item()
                    mean_policy_loss += policy_loss.item()
                    mean_accuracy += accuracy
                    updates += 1

        mean_value_loss /= updates
        mean_policy_loss /= updates
        mean_accuracy /= updates

        print(
            "value_loss = {:f} policy_loss = {:f} accuracy = {:f}, train_time = {:.3f}".format(mean_value_loss,
                                                                                               mean_policy_loss,
                                                                                               mean_accuracy,
                                                                                               time.time() - batch_time))

        torch.save(model.state_dict(), args.checkpoint_file)
        torch.save(optimizer.state_dict(), args.checkpoint_file + '_optimizer.pth')


