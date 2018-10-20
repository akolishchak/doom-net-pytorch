#
# mcts.py
#
# Created by Andrey Kolishchak on 04/21/18.
#
import numpy as np


class Node:
    def __init__(self):
        self.total_visits = 0
        self.reward = 0
        self.visits = None
        self.value = None
        self.mean_value = None
        self.policy = None
        self.action_mask = None


class MCTS:
    def __init__(self, simulator, sim_num, c_puct=1):
        self.simulator = simulator
        self.nodes = {}
        self.sim_num = sim_num
        self.c_puct = c_puct

    def search(self, state, is_root=False):

        try:
            node = self.nodes[state.key]
        except KeyError:
            # reached leaf node
            node = Node()
            if self.simulator.is_finished(state):
                node.reward = self.simulator.get_reward(state)
                return node.reward
            # init node
            node.action_mask = self.simulator.get_available_actions_mask(state)
            # policy
            value, node.policy = self.simulator.rollout(state)
            node.policy *= node.action_mask
            if node.policy.sum() == 0:
                print("zero actions")
                node.policy[node.action_mask == 1] = 1
            node.policy /= node.policy.sum()
            # values and visits
            action_size = self.simulator.get_action_size(state)
            node.value = np.zeros(action_size)
            node.mean_value = np.zeros(action_size)
            node.visits = np.zeros(action_size)
            node.total_visits = 1
            # minimize value of unavailable actions
            node.mean_value[node.action_mask == 0] = np.finfo(np.float32).min
            #
            self.nodes[state.key] = node
            return value

        if node.reward != 0:
            return node.reward

        # select best action
        policy = node.policy
        if is_root:
            # add Dirichlet noise to the root node
            noise = np.random.dirichlet(node.action_mask * 0.03)
            epsilon = 0.25
            policy = (1 - epsilon) * policy + epsilon * noise

        u = node.mean_value + self.c_puct * policy * np.sqrt(node.total_visits) / (1 + node.visits)
        best_action = np.argmax(u)

        # go to next state
        state = self.simulator.get_next_state(state, best_action)
        value = self.search(state)

        # backup
        node.value[best_action] += value
        node.visits[best_action] += 1
        node.mean_value[best_action] = node.value[best_action] / node.visits[best_action]
        node.total_visits += 1

        return value

    def get_action_prob(self, game, tau=1):

        state = self.simulator.get_state(game)

        for i in range(self.sim_num):
            self.search(state, is_root=True)

        root = self.nodes[state.key]

        prob = np.power(root.visits, 1/tau if tau != 0 else 1/1e-3)
        prob /= prob.sum()
        return prob, state






