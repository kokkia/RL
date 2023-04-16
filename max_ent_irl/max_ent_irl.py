import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import random

# selflibrary 
sys.path.append("../dp")
import value_iteration

class max_ent_irl:
    def __init__(self, states, actions, env, max_steps, fig=None, ax=None):
        self.states = states
        self.actions = actions
        self.ns = len(states)
        self.na = len(actions)
        self.env = env
        self.max_steps = max_steps
        self.phi = np.ones([self.ns])
        self.theta = np.random.rand(self.ns)
        self.gamma = 0.95
        self.R = np.zeros([self.ns])

        # dp
        self.dp = value_iteration.value_iteration(states, actions, env, max_steps, fig, ax)

        # gradient descent
        self.alpha = 0.1

        # fig
        self.fig = fig
        self.ax = ax

        return
    
    # state visitation frequency
    def set_expert_trajectory(self, trajs):
        self.trajs = trajs
        self.ntraj = len(trajs)
        return

    def expect_svf(self):
        nstep = len(self.trajs[0])
        mu = np.ones([self.ns])
        for j in range(nstep):
            for state in self.states:
                action = self.dp.get_action(state)
                r, next_state, fin = self.env.reward(state, action)
                mu[next_state] += 1
        svf = mu / nstep
        return svf

    def learn(self):
        # for expert data
        feature_expert = np.zeros([self.ns])
        for traj in self.trajs:
            for step in traj:
                feature = np.zeros([self.ns])
                feature[step] = self.phi[step]
                feature_expert += feature
        feature_expert = feature_expert / len(self.trajs)

        # for agent
        for i in range(self.max_steps):
            # update reward
            r = self.theta * self.phi
            self.env.set_reward(r)
            self.dp.set_env(self.env)
            # get policy
            self.dp.learn()
            # train
            svf = self.expect_svf()
            grad = - feature_expert + svf * self.phi
            self.theta = self.theta - self.alpha * grad

        self.R = self.theta * self.phi
        return

    def plot_reward(self, x_range, y_range):
        if self.fig == None:
            return
        self.ax.cla()
        plt.pause(0.1)
        p = self.ax.imshow(np.reshape(self.R, (x_range, y_range)))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(p, cax=cax)
        plt.pause(0.1)
        return
        
