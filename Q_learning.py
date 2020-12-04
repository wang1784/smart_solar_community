#Import necessary libraries
import random
from matplotlib import pyplot as plt
from collections import defaultdict, namedtuple
from tqdm import tqdm
import numpy as np

#Danielle Q learning agent from HW 5

class Q_Learning_Agent(object):
    def __init__(self, env, actions, alpha=0.5, epsilon=0.1, gamma=1):
        self._env = env
        self._actions = actions
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self.episodes = []
        # init q table
        self._q = {}
        action_vals = {a: 0 for a in self._actions}
        for x in range(self._env.shape[0]):
            for y in range(self._env.shape[1]):
                self._q[(x, y)] = dict(action_vals)

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
        s1 = self._env.reset()
        transitions = []
        while True:
            a = self.e_greedy_policy(s1)
            s2, r, term, _ = self._env.step(a)
            
            if term:
                target = 0.0 #CHANGE THIS PART 
            self._q[s1][a] = self._q[s1][a] + self._alpha * (
                        r + self._gamma * np.max(self._q[s2][a])) - self._q[s1][a]

            s1 = s2

            transitions.append(Transition(s1, a, r, s2))

            if term:
                break
        return transitions

    def learn(self, n_episodes=500):
        for _ in tqdm(range(n_episodes)):
            transitions = self.play_episode()
            self.episodes.append(transitions)


