
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class actor_net(nn.Module):
  def __init__(self,ns,na):
      super(actor_net, self).__init__()
      self.fc1 = nn.Linear(ns, 30)
      self.dropout1 = nn.Dropout2d()
      self.fc2 = nn.Linear(30, 30)
      self.dropout2 = nn.Dropout2d()
      self.fc3 = nn.Linear(30,na) # mu, log(sigma**2)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      # x = self.dropout1(x)
      x = F.relu(self.fc2(x))
      # x = self.dropout2(x)
      x = self.fc3(x)
      return x

class critic_net(nn.Module):
  def __init__(self,ns,na):
      super(critic_net, self).__init__()
      self.fc1 = nn.Linear(ns, 30)
      self.dropout1 = nn.Dropout2d()
      self.fc2 = nn.Linear(30, 30)
      self.dropout2 = nn.Dropout2d()
      self.fc3 = nn.Linear(30,1)# state value

  def forward(self, x):
      x = F.relu(self.fc1(x))
      # x = self.dropout1(x)
      x = F.relu(self.fc2(x))
      # x = self.dropout2(x)
      x = self.fc3(x)
      return x

class REINFORCE:
  def __init__(self,dstates, dactions, s0, reward, nepisode,actor_net,critic_net):
    # self.states = states
    # self.actions = actions
    self.ns = dstates
    self.na = dactions
    self.max_episode_step = nepisode
    self.r = reward
    self.s0 = s0
    self.alpha = 0.5
    self.gamma = 0.99#0.99
    
    self.history = []
    self.memory = []
    # actor
    self.actor_net = actor_net
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=0.001)
    self.actor_criterion = nn.MSELoss()
    # critic
    self.critic_net = critic_net
    self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=0.001)
    self.critic_criterion = nn.MSELoss()

  def learn(self):
    iter = []
    l = []
    for i in range(300):
      s = copy.deepcopy(self.s0)
      # print(self.s0,s)

      # 1episode計算
      steps = 0
      self.actor_net.eval()
      while True:
        steps += 1    
        # 行動決定
        ts = torch.from_numpy(s.astype(np.float32)).clone()
        a = self.actor_net.forward(ts)
        a = a.to('cpu').detach().numpy().copy()
        mu = a[:self.na]
        logvar = a[self.na:]
        var = np.exp(logvar)
        action = np.random.normal(loc=mu, scale=np.sqrt(var))
        # print(s,a)
        r, s_, fin = self.r(s, action)  # 行動の結果、rewardと状態が帰ってくる
        # print(s,a,s_,r)
        # addmemory
        self.memory.append((s,a,action,s_,r))
        # update
        s = copy.deepcopy(s_)
        if fin==1:
            # print("\n episode end episode:",i," step:",step,"\n")
            break
        if steps > self.max_episode_step:
            break

      # 学習
      advanteges = []
      state_values = []
      states = []
      actions = []
      for t, step in enumerate(self.memory):
        s,a,action,s_,r = step
        # 割引報酬計算
        Gt = 0
        for j, mem in enumerate(self.memory[t:]):
          rs, ra, raction, rs_, rr = copy.deepcopy(mem)
          Gt += self.gamma ** j * rr
        Gt = torch.from_numpy(np.array([Gt]).astype(np.float32)).clone()
        # 状態価値計算
        self.critic_net.eval()
        ts = torch.from_numpy(s.astype(np.float32)).clone()
        state_value = self.critic_net.forward(ts)
        # advantege計算
        advantege = Gt - state_value
        # 保存
        advanteges.append(advantege)
        state_values.append(state_value)
        states.append(s)
        actions.append(a)

        # network更新
        self.actor_net.train()
        self.critic_net.train()
        # 方策勾配計算
        mu = ra[:self.na]
        logvar = ra[self.na:]
        action = raction
        var = np.exp(logvar)
        logpi = -0.5 * logvar - 0.5 * (action - mu)**2 / var# 次元注意
        logpi = torch.from_numpy(logpi.astype(np.float32)).clone()
        actor_loss = -logpi * advantege
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # critic
        critic_loss = F.smooth_l1_loss(Gt,state_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
      print("episode", i, "reward", r, "al", actor_loss, "cl", critic_loss)

      if (i + 1) % 50 == 0:
        torch.save(self.actor_net.state_dict(), "out_RF/dnn" + str(i + 1) +".pt")
        print("loss=",actor_loss)
        
      iter.append(i)
      l.append(Gt)
    plt.plot(iter, l, '-k')
    plt.show()

  def reward(self, s,a):
    return self.r(s,a)

  def action(self,s):
    # s = np.array([s])
    ts = torch.from_numpy(s.astype(np.float32)).clone()
    ts = ts.unsqueeze(dim=0)
    ts = torch.tensor(ts, dtype=torch.float)
    self.Q = self.targetnet.forward(ts)
    self.Q = self.Q.to('cpu').detach().numpy().copy()
    # print(self.Q)

    a = np.argmax(self.Q[0,:])
    return a
