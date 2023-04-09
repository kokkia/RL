import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import random

class value_iteration:
    def __init__(self, states, actions, env, max_steps, fig=None, ax=None):
        self.states = states
        self.actions = actions
        self.ns = len(states)
        self.na = len(actions)
        self.env = env
        self.max_steps = max_steps
        self.V = np.zeros([self.ns])
        self.Q = np.zeros([self.ns, self.na])
        self.gamma = 0.95
        self.fig = fig
        self.ax = ax
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
        # ret_action = np.argmax(self.Q[state,:])
        Q_max = -1e6
        for action in self.actions:
            R, next_state, fin = self.env.reward(state, action)
            Q_tmp = R + self.gamma * self.V[next_state]
            if Q_tmp > Q_max:
                Q_max = Q_tmp
                ret_action = action 
        return ret_action
        
    def plot_state_value(self, x_range, y_range):
        if self.fig == None:
            return
        p = self.ax.imshow(np.reshape(self.V, (x_range, y_range)))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(p, cax=cax)
        return
        



    