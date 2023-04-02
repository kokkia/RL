import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

class value_iteration:
    def __init__(self, states, actions, env, max_steps):
        self.states = states
        self.actions = actions
        self.ns = len(states)
        self.na = len(actions)
        self.env = env
        self.max_steps = max_steps
        self.V = np.zeros([self.ns])
        self.Q = np.zeros([self.ns, self.na])
        self.gamma = 0.95
        return 

    def learn(self):
        for step in range(self.max_steps):
            for state in self.states:
                Q_max = -1e6
                for action in self.actions:
                    R, next_state, fin = self.env.reward(state, action)
                    # print(R)
                    Q_tmp = R + self.gamma * self.V[next_state]
                    self.Q[state,action] = Q_tmp
                    if Q_tmp > Q_max:
                        Q_max = Q_tmp
                self.V[state] = Q_max
        return

    def get_action(self, state):
        ret_action = 0
        ret_action = np.argmax(self.Q[state,:])
        # V_max = -1e6
        # for action in self.actions:
        #     R, next_state, fin = self.env.reward(state, action)
        #     if self.V[next_state] > V_max:
        #         V_max = copy.deepcopy(self.V[next_state])
        #         ret_action = action 
        return ret_action
    