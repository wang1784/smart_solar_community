#Import necessary libraries
import random
from matplotlib import pyplot as plt
from collections import defaultdict, namedtuple
from tqdm import tqdm
import numpy as np
import solar_power_env
from solar_power_env import solar_power_env

#Danielle Q learning agent from HW 5


Transition = namedtuple('Transition', ['state1',
                                       'action',
                                       'reward',
                                       'state2'])


class Q_Learning_Agent(object):
    def __init__(self, env, actions, alpha=0.5, epsilon=0.1, gamma=1):
        self._env = env #should have function that outputs state and reward
        self._actions = actions #list of actions
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self.episodes = []
        # init q table
        self._q = {}
        action_vals = {a: 0 for a in self._actions}
        for x in range(self._env._state_space_shape[0]):
            for y in range(self._env._state_space_shape[1]):
                for z in range(self._env._state_space_shape[1]):
                    self._q[(x, y, z)] = dict(action_vals)

    def random_policy(self, state):
        return random.choice(self._actions)

    def greedy_policy(self, state):
        return max(self._q[state], key=self._q[state].get)

    def e_greedy_policy(self, state):
        if np.random.rand() > self._epsilon:
            action = self.greedy_policy(state)
        else:
            action = self.random_policy(state)
        return action

    def play_episode(self):
        s1 = self._env._state
        transitions = []
        i = 0
        while i <= 8760:
            a = self.e_greedy_policy(s1)
            s2, r, term = self._env.step(a)

            self._q[s1][a] = self._q[s1][a] + self._alpha * (
                        r + self._gamma * np.max(self._q[s2][a])) - self._q[s1][a]

            s1 = s2

            print(Transition(s1, a, r, s2))
            transitions.append(Transition(s1, a, r, s2))

            if term:
                break
            i += 1

        return transitions

    def learn(self, n_episodes=500):
        for _ in tqdm(range(n_episodes)):
            transitions = self.play_episode()
            self.episodes.append(transitions)

def evaluate(agent):
    agent.learn()

    total_rewards = []
    episode_ids = []
    for e_id, episode in enumerate(agent.episodes):
        rewards = map(lambda e: e.reward, episode)
        total_rewards.append(sum(rewards))
        episode_ids.extend([e_id] * len(episode))

    return episode_ids

##### TONIA'S Q-LEARNING FUNCTIONS #####
# def q_learning(self):
#     self.initialize()
#     time_steps = 0
#     num_of_episode = []  # record which episode each time step is on
#     episode_counts = 0  # start episode count from 0
#     while time_steps <= 8000:
#         state = self.start
#         while state != self.goal:
#             q_s = self.q[state[0], state[1]]
#             action_s, __ = self.choose_action_from_q(q_s)
#             sprime = self.get_sprime(state, action_s)
#             reward = self.rewards[state[0], state[1], action_s]
#             q_sprime = self.q[sprime[0], sprime[1]]
#
#             #update q and state
#             self.q[state[0], state[1], action_s] = self.q[state[0], state[1], action_s] + self.alpha * (reward + max(q_sprime) - self.q[state[0], state[1], action_s]) #q update
#             state = sprime
#
#             #update increments
#             time_steps += 1
#             num_of_episode.append(episode_counts)
#         episode_counts += 1
#     return num_of_episode

test=Q_Learning_Agent(solar_power_env(),actions=[0,1])
test.play_episode()
print(test._q)
