
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


class experience_replay:
  def __init__(self,batch_size):
    self.batch_size = batch_size
    self.memory = []
    self.memory_size = 1000
    
  def add(self,s,a,s_,r):
    self.memory.append((s, a, s_, r))
    if len(self.memory) > self.memory_size:
        self.memory= self.memory[-self.memory_size:]

  def sample(self):
    exps = random.sample(self.memory,self.batch_size)
    return exps

class DDPG:
  def __init__(self,dstates, dactions, s0, reward, nepisode, actor_net, critic_net):
    # self.states = states
    self.actions = dactions
    self.ns = dstates
    self.na = dactions
    # self.epsilon = epsilon
    self.nepisode = nepisode
    self.r = reward
    self.s0 = s0
    self.alpha = 0.5
    self.gamma = 0.99
    self.tau = 0.1
    
    self.history=[]
    # actor
    self.actor_net = actor_net
    self.actor_target = copy.deepcopy(actor_net)
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=0.001)
    self.actor_criterion = nn.MSELoss()
    # critic
    self.critic_net = critic_net
    self.critic_target = copy.deepcopy(critic_net)
    self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=0.001)
    self.critic_criterion = nn.MSELoss()

    self.batch_size = 32
    self.exp = experience_replay(self.batch_size)

  def learn(self):
    t = []
    l = []
    for i in range(300):
      s = copy.deepcopy(self.s0)
    #   s= env.reset()
      print(self.s0,s)
      for j in range(self.nepisode):
        # 行動決定
        ts = torch.from_numpy(s.astype(np.float32)).clone()
        a = self.actor_net.forward(ts)
        a = a.to('cpu').detach().numpy().copy()
        a = a + np.random.rand(self.na)
        # print(s,a)
        r, s_, fin = self.r(s, a)  # 行動の結果、rewardと状態が帰ってくる
        print(s,a,s_,r)
        # addmemory
        self.exp.add(s,a,s_,r)
        # replay
        if len(self.exp.memory) < self.batch_size:
          continue
        batches = self.exp.sample()
        for k, batch in enumerate(batches, 0):
          rs,ra,rs_,rr = batch
        #   print(rs,ra,rs_,rr)
          # Q値を計算 
          trs_ = torch.from_numpy(rs_.astype(np.float32)).clone()
          tra_ = self.actor_target.forward(trs_)
          tx_ = torch.cat([trs_, tra_], dim=0)
          Q_ = self.critic_target.forward(tx_)
          trr = torch.from_numpy(np.array([rr]).astype(np.float32)).clone()
          ty = trr + Q_
          trs = torch.from_numpy(rs.astype(np.float32)).clone()
          tra = torch.from_numpy(ra.astype(np.float32)).clone()
          tx = torch.cat([trs, tra], dim=0)
          Q = self.critic_net.forward(tx)
          # network更新
          self.actor_net.train()
          self.critic_net.train()
          # critic
          critic_loss = F.smooth_l1_loss(ty,Q)
          self.critic_optimizer.zero_grad()
          critic_loss.backward()
          self.critic_optimizer.step()
          # actor
          actor_loss = - self.critic_net.forward(tx).mean()
          self.actor_optimizer.zero_grad()
          actor_loss.backward(retain_graph=True)
          self.actor_optimizer.step()

          for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
          for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
              target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # train
        # self.Q[s,a] += self.alpha*(r+self.gamma*np.max(self.Q[s_,:])-self.Q[s,a])# TD学習
        s = copy.deepcopy(s_)
        self.history.append(s)
        if fin==1:
          print("\n episode end epidode:",i,"j=",j,"\n")
        #   plt.pause(1)
          break
        # print(self.Q)
      if (i + 1) % 50 == 0:
        torch.save(self.actor_target.state_dict(), "out/dnn" + str(i + 1) +".pt")
        print("loss=",actor_loss)
        
      t.append(i)
      l.append(actor_loss)
    plt.plot(t, l, '-k')
    plt.show()

  def reward(self, s,a):
    return self.r(s,a)

  def action(self,s):
    # s = np.array([s])
    ts = torch.from_numpy(s.astype(np.float32)).clone()
    ts = ts.unsqueeze(dim=0)
    ts = torch.tensor(ts, dtype=torch.float)
    ta = self.actor_target.forward(ts)
    a = ta.to('cpu').detach().numpy().copy()
    return a

