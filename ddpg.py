
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
      self.fc1 = nn.Linear(ns, 16)
      # self.dropout1 = nn.Dropout2d()
      self.fc2 = nn.Linear(16, 16)
      self.fc3 = nn.Linear(16, 16)
      # self.dropout2 = nn.Dropout2d()
      self.fc4 = nn.Linear(16,na) # mu, log(sigma**2)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

class critic_net(nn.Module):
  def __init__(self,ns,na):
      super(critic_net, self).__init__()
      self.fc1 = nn.Linear(ns, 32)
      # self.dropout1 = nn.Dropout2d()
      self.fc2 = nn.Linear(32, 32)
      # self.dropout2 = nn.Dropout2d()
      self.fc3 = nn.Linear(32, 32)
      # self.dropout2 = nn.Dropout2d()
      self.fc4 = nn.Linear(32,1)# state value

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
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
    self.gamma = 0.99
    self.tau = 0.001
    
    self.history=[]
    # actor
    self.actor_net = actor_net
    self.actor_target = copy.deepcopy(actor_net)
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=0.0001)
    self.actor_criterion = nn.MSELoss()
    # critic
    self.critic_net = critic_net
    self.critic_target = copy.deepcopy(critic_net)
    self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=0.001, weight_decay=0.01)
    self.critic_criterion = nn.MSELoss()

    self.batch_size = 64
    self.exp = experience_replay(self.batch_size)

  def learn(self):
    t = []
    l = []
    for i in range(300):
      s = copy.deepcopy(self.s0)
    #   s= env.reset()
      print(self.s0,s)
      total_reward = 0
      for j in range(self.nepisode):
        # 行動決定
        ts = torch.from_numpy(s.astype(np.float32)).clone()
        a = self.actor_net.forward(ts)
        a = a.to('cpu').detach().numpy().copy()
        a = a + np.random.randn(self.na)*0.5
        # print(s,a)
        r, s_, fin = self.r(s, a)  # 行動の結果、rewardと状態が帰ってくる
        total_reward += r
        # print(s,a,s_,r)
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
          critic_loss = F.mse_loss(ty,Q)
          self.critic_optimizer.zero_grad()
          critic_loss.backward()
          self.critic_optimizer.step()
          # actor
          actor_loss = - self.critic_net.forward(tx).mean()
          self.actor_optimizer.zero_grad()
          actor_loss.backward()
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
          break
      print("total_reward", total_reward)
      if (i + 1) % 10 == 0:
        torch.save(self.actor_target.state_dict(), "out_RF/dnn" + str(i + 1) +".pt")
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
    # ts = ts.unsqueeze(dim=0)
    # ts = torch.tensor(ts, dtype=torch.float)
    ta = self.actor_target.forward(ts)
    a = ta.to('cpu').detach().numpy().copy()
    return a

