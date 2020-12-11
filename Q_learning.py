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
                for z in range(self._env._state_space_shape[2]):
                    self._q[(x, y, z)] = dict(action_vals)

    def random_policy(self):
        return random.choice(self._actions)

    def greedy_policy(self, state):
        return max(self._q[state], key=self._q[state].get)

    def e_greedy_policy(self, state):
        if np.random.rand() > self._epsilon:
            action = self.greedy_policy(state)
        else:
            action = self.random_policy()
        return action

    def play_episode(self):
        s1 = self._env._state
        transitions = []
        i = 0
        battery_history= []
        battery_to_load_step = 0
        load_step = self._env._data.iloc[0]['COMED_W']
        b0 = [battery_to_load_step/load_step]
        p_grid = [load_step - battery_to_load_step]

        while True:
            a = self.e_greedy_policy(s1)
            s2, r, battery_change, term = self._env.step(a)
            # print(s2, r, term)
            self._q[s1][a] = self._q[s1][a] + self._alpha * (
                        r + self._gamma * np.max(self._q[s2][a])) - self._q[s1][a]

            s1 = s2


            #get values for b0
            if a == 0:
                battery_to_load_step += r
                if battery_change > 0: print('battery change error')
            load_step += self._env._data.iloc[i]['COMED_W']
            b0.append(battery_to_load_step/load_step)
            # print(p_grid[-1], load_step-battery_to_load_step)
            p_grid.append(load_step - battery_to_load_step)

            # print(Transition(s1, a, r, s2))
            transitions.append(Transition(s1, a, r, s2))
            battery_history.append(battery_change)

            i += 1
            if term:
                break

        return transitions, b0, p_grid, battery_history






test=Q_Learning_Agent(solar_power_env(),actions=[0,1])
__, b0, p_grid, battery = test.play_episode()
print(test._q)
neg_count = 0
zero_count = 0
for eachbattery in battery:
    if eachbattery<0: neg_count += 1
    elif eachbattery == 0: zero_count += 1
print(neg_count/len(battery))
print(zero_count/len(battery))
#plot battery level
plt.plot(range(9000), battery[:9000])
plt.xlabel('HOUR for a year')
plt.ylabel('Battery level')
plt.show()

# #plot b0
plt.plot(range(9000), b0[:9000])
plt.ylabel('Power from battery to load / load')
plt.xlabel('Hours')
plt.title('Utility of battery, epsilon = 0.1, alpha = 0.5, gamma = 1')
plt.show()
#plot grid
plt.plot(range(9000), p_grid[:9000])
plt.ylabel('Power from grid')
plt.xlabel('Hours')
plt.title('Grid power with Q-learning, epsilon = 0.1, alpha = 0.5, gamma = 1')
plt.show()
