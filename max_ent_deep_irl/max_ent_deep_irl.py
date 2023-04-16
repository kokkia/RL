import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import random

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# selflibrary 
sys.path.append("../dp")
import value_iteration


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)

class reward_net(nn.Module):
  def __init__(self,ns):
      super(reward_net, self).__init__()
      self.fc1 = nn.Linear(ns, 50)
      self.fc2 = nn.Linear(50, 50)
      self.fc3 = nn.Linear(50, 50)
      self.fc4 = nn.Linear(50,1)# state value

      init_w = 3e-4
      self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      self.fc3.weight.data = init_weight(self.fc2.weight.data.size())
      self.fc4.weight.data.uniform_(-init_w, init_w)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

class max_ent_deep_irl:
    def __init__(self, states, actions, env, max_steps, reward_net, device, fig=None, ax=None):
        self.states = states
        self.actions = actions
        self.ns = len(states)
        self.na = len(actions)
        self.env = env
        self.max_steps = max_steps
        self.gamma = 0.95
        self.phi = np.ones([self.ns])
        self.R = np.zeros([self.ns])

        # dp
        self.dp = value_iteration.value_iteration(states, actions, env, max_steps, fig, ax)

        # network
        self.device = device
        self.lr = 1e-4
        self.reward_net = reward_net
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(),lr=self.lr,weight_decay=0.01)

        # gradient descent
        self.alpha = 0.1

        # log
        self.writer = SummaryWriter(log_dir="./logs")

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
            r = self.create_reward_map()
            self.env.set_reward(r)
            self.dp.set_env(self.env)
            # get policy
            self.dp.learn()
            # train
            svf = self.expect_svf()
            grad = - feature_expert + svf * self.phi
            t_grad = torch.from_numpy(grad.astype(np.float32)).to(self.device)
            # train
            self.reward_net.train()
            reward_sum = 0
            for s in self.states:
                ts = np.array([s])
                ts = torch.from_numpy(ts.astype(np.float32)).clone()
                reward_sum += self.reward_net.forward(ts)
            reward_loss = reward_sum
            self.reward_optimizer.zero_grad()
            reward_loss.backward(torch.sum(t_grad))
            self.reward_optimizer.step()

        self.R = self.create_reward_map()
        return

    def create_reward_map(self):
        R = np.zeros([self.ns])
        for s in self.states:
            ts = np.array([s])
            ts = torch.from_numpy(ts.astype(np.float32)).clone()
            r = self.reward_net.forward(ts)
            R[int(s)] = r
        return R

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
        
