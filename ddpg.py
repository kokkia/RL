
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ornstein_uhlenbeck_process import *


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)

class actor_net(nn.Module):
  def __init__(self,ns,na):
      super(actor_net, self).__init__()
      self.fc1 = nn.Linear(ns, 400)
      self.fc2 = nn.Linear(400, 300)
      self.fc3 = nn.Linear(300, na)
      init_w = 3e-3
      self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      self.fc3.weight.data.uniform_(-init_w, init_w)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = torch.tanh(self.fc3(x))*2
      return x

class critic_net(nn.Module):
  def __init__(self,ns,na):
      super(critic_net, self).__init__()
      self.fc1 = nn.Linear(ns, 400)
      self.fc2 = nn.Linear(400+na, 300)
      self.fc3 = nn.Linear(300,1)# state value

      init_w = 3e-4
      self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      self.fc3.weight.data.uniform_(-init_w, init_w)

  def forward(self, x, action):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(torch.cat([x,action],dim=1)))
      x = self.fc3(x)
      return x


class experience_replay:
  def __init__(self,batch_size):
    self.batch_size = batch_size
    self.memory = []
    self.memory_size = 1000000
    
  def add(self,s,a,s_,r):
    self.memory.append((s, a, s_, r))
    if len(self.memory) > self.memory_size:
        self.memory= self.memory[-self.memory_size:]

  def sample(self):
    exps = random.sample(self.memory,self.batch_size)
    return exps

  def reset(self):
    self.memory.clear()

class DDPG:
  def __init__(self,dstates, dactions, s0, env, max_steps, max_episodes, actor_net, critic_net, device):
    self.ns = dstates
    self.na = dactions
    self.max_steps = max_steps
    self.max_episodes = max_episodes
    self.env = env
    self.s0 = s0
    self.gamma = 0.99# 割引率
    self.tau = 0.001# ターゲット更新率
    self.batch_size = 64
    self.lr_actor = 1e-4
    self.lr_critic = 1e-3
    self.epsilon = 1.0

    self.device = device
    self.random_process = OrnsteinUhlenbeckProcess(size=self.na)
    
    # actor
    self.actor_net = actor_net
    self.actor_target = copy.deepcopy(actor_net)
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=self.lr_actor)
    self.actor_criterion = nn.MSELoss()
    # critic
    self.critic_net = critic_net
    self.critic_target = copy.deepcopy(critic_net)
    self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=self.lr_critic, weight_decay=0.01)
    self.critic_criterion = nn.MSELoss()

    # experimental replay
    self.exp = experience_replay(self.batch_size)

    # log
    self.writer = SummaryWriter(log_dir="./logs")

    self.history=[]

  
  def get_action(self, s, greedy=False):
      ts = torch.from_numpy(s.astype(np.float32)).clone()
      ts = ts.to(self.device)
      a = self.actor_net.forward(ts)
      if not greedy:
        a = a + self.epsilon*torch.tensor(self.random_process.sample(), dtype=torch.float, device=self.device)
      a = a.to(device=self.device).detach().cpu().numpy().copy()
      return a

  def learn(self):
    t = []
    l = []
    for episode in range(self.max_episodes):
      s= self.env.reset()
      print("initial_state",s)
      total_reward = 0
      for step in range(self.max_steps):
        # 行動決定
        a = self.get_action(s)
        # print(s,a)
        r, s_, fin = self.env.reward(s, a)  # 行動の結果、rewardと状態が帰ってくる
        total_reward += r
        print(s,a,s_,r)
        # addmemory
        self.exp.add(s,a,s_,r)
        # replay
        if len(self.exp.memory) > self.batch_size:
          batches = self.exp.sample()
          for k, batch in enumerate(batches, 0):
            rs,ra,rs_,rr = batch
          #   print(rs,ra,rs_,rr)
            # Q値を計算 
            trs_ = torch.from_numpy(rs_.astype(np.float32)).clone()
            trs_ = trs_.to(self.device)
            tra_ = self.actor_target.forward(trs_)
            Q_ = self.critic_target.forward(trs_,tra_)
            trr = torch.from_numpy(np.array([rr]).astype(np.float32)).clone()
            trr = trr.to(self.device)
            ty = trr + self.gamma*Q_
            trs = torch.from_numpy(rs.astype(np.float32)).clone()
            trs = trs.to(self.device)
            tra = torch.from_numpy(ra.astype(np.float32)).clone()
            tra = tra.to(self.device)
            Q = self.critic_net.forward(trs,tra)
            # network更新
            self.actor_net.train()
            self.critic_net.train()
            # critic
            critic_loss = F.mse_loss(ty,Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # actor
            actor_loss = - self.critic_net.forward(trs,tra).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
                  target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        else:
          pass
        self.history.append(s)
        s = copy.deepcopy(s_)
        if fin==1:
          print("\n episode end epidode:",episode,"step=",step,"\n")
          break
      print("total_reward", total_reward)
      if (episode + 1) % 10 == 0:
        torch.save(self.actor_target.state_dict(), "out_DDPG/dnn" + str(episode + 1) +".pt")
        print("critic loss=",critic_loss)
      self.writer.add_scalar("total reward", total_reward,episode)
      self.writer.add_scalar("critic_loss", critic_loss, episode)
      self.writer.add_scalar("actor_loss", actor_loss, episode)
        
      t.append(episode)
      l.append(critic_loss)
    plt.plot(t, l, '-k')
    plt.show()
    self.writer.close()

  def reward(self, s,a):
    return self.r(s,a)

  # def action(self,s):
  #   # s = np.array([s])
  #   ts = torch.from_numpy(s.astype(np.float32)).clone()
  #   # ts = ts.unsqueeze(dim=0)
  #   # ts = torch.tensor(ts, dtype=torch.float)
  #   ta = self.actor_target.forward(ts)
  #   a = ta.to('cpu').detach().numpy().copy()
  #   return a

