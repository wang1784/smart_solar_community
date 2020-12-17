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
        days_count = 0
        battery_to_load_history= []
        battery_to_load_step = 0
        load_history = []
        load_step = 0
        # b_hourly_history = []
        # b_hourly_step = 0
        # load_hourly_history = []
        # load_hourly_step = 0

        while True:
            a = self.e_greedy_policy(s1)
            s2, r, battery_change, term = self._env.step(a)
            # print(s2, r, term)
            self._q[s1][a] = self._q[s1][a] + self._alpha * (
                        r + self._gamma * np.max(self._q[s2][a])) - self._q[s1][a]

            s1 = s2

            # if battery_change > 0:

            #get values for b0
            if days_count == 8760:
                battery_to_load_history.append(battery_to_load_step)
                load_history.append(load_step)
                # print('appending values: ', battery_to_load_step, load_step)
                days_count = 0
                battery_to_load_step = 0
                load_step = 0
            if a == 0:
                battery_to_load_step += r
            #     if battery_change > 0: print('battery change error')
            load_step += self._env._data.iloc[i]['COMED_W']
            # b0.append(battery_to_load_step/load_step)
            # # print(p_grid[-1], load_step-battery_to_load_step)
            # p_grid.append(load_step - battery_to_load_step)

            # print(Transition(s1, a, r, s2))
            transitions.append(Transition(s1, a, r, s2))
            # battery_history.append(battery_change)
            days_count += 1
            i += 1
            if term:
                break
        b0 = np.divide(battery_to_load_history, load_history)
        g0 = np.subtract(load_history, battery_to_load_history)
        return transitions, b0, g0


def parameter_tuning(epsilon, alpha, gamma):
    b0_epsilon = []
    g0_epsilon = []
    for eachepsilon in epsilon:
        print('Tuning epsilon: ', eachepsilon)
        b0_alpha = []
        g0_alpha = []
        for eachalpha in alpha:
            print('Tuning alpha: ', eachalpha)
            b0_gamma = []
            g0_gamma = []
            for eachgamma in gamma:
                env_tuning = Q_Learning_Agent(solar_power_env(), actions=[0, 1],
                                              epsilon = eachepsilon, alpha = eachalpha, gamma = eachgamma)
                __, b0, g0 = env_tuning.play_episode()
                b0_gamma.append(max(b0))
                g0_gamma.append(min(g0))
            b0_alpha.append(b0_gamma)
            g0_alpha.append(g0_gamma)
            b0_best_gamma = b0_gamma.index(max(b0_gamma))
            g0_best_gamma = g0_gamma.index(min(g0_gamma))
        b0_epsilon.append(b0_alpha)
        g0_epsilon.append(g0_alpha)
        b0_best_alpha = b0_alpha.index(max(b0_alpha))
        g0_best_alpha = g0_alpha.index(min(g0_alpha))
    b0_best_epsilon = b0_epsilon.index(max(b0_epsilon))
    g0_best_epsilon = g0_epsilon.index(min(g0_epsilon))
    b0_best_para = [epsilon[b0_best_epsilon], alpha[b0_best_alpha], gamma[b0_best_gamma]]  # gamma, alpha and then epsilon
    g0_best_para = [epsilon[g0_best_epsilon], alpha[g0_best_alpha], gamma[g0_best_gamma]]  # gamma, alpha and then epsilon
    return b0_epsilon, g0_epsilon, b0_best_para, g0_best_para



##########################################################
# epsilon_tune = [0, 0.25, 0.5, 0.75, 1]
# alpha_tune = [0.1, 0.3, 0.5, 0.7]
# gamma_tune = np.arange(0.8, 1.1, 0.1)
#
# b0, g0, b0_best, g0_best = parameter_tuning(epsilon_tune, alpha_tune, gamma_tune)
# print('the best parameters are: ', g0_best)
plot_policy = [0, 0.1, 1]
b_epsilon = []
g_epsilon = []
for eachepsilon in plot_policy:
    test=Q_Learning_Agent(solar_power_env(),actions=[0,1],
                          epsilon = eachepsilon, alpha = 0.1, gamma = 1)
    __, b0, g0 = test.play_episode()
    b_epsilon.append(b0)
    g_epsilon.append(g0)

for eachepsilon in range(len(plot_policy)):
    current_b = b_epsilon[eachepsilon]
    plt.plot(range(len(current_b)), current_b, label = plot_policy[eachepsilon])
    plt.ylabel('Power from battery to load / load')
    plt.xlabel('Year')
    plt.title('Utility of Battery')
plt.legend()
plt.show()

for eachepsilon in range(len(plot_policy)):
    current_g = g_epsilon[eachepsilon]
    plt.plot(range(len(current_g)), current_g, label = plot_policy[eachepsilon])
    plt.ylabel('Power from grid')
    plt.xlabel('Year')
    plt.title('Grid power')
plt.legend()
plt.show()

#
# #plot b0
# plt.plot(range(len(b0)), b0)
# plt.ylabel('Power from battery to load / load')
# plt.xlabel('Hours')
# plt.title('Utility of Battery with alpha=0.1, epsilon=0.1, gamma=1')
# plt.show()
# #plot grid
# plt.plot(range(len(g0)), g0)
# plt.ylabel('Power from grid')
# plt.xlabel('Hours')
# plt.title('Grid power with alpha=0.1, epsilon=0.1, gamma=1')
# plt.show()
