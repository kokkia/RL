
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)

class actor_net(nn.Module):
  def __init__(self,ns,na):
      super(actor_net, self).__init__()
      self.na = na
      self.ns = ns
      self.fc1 = nn.Linear(ns, 50)
      self.fc2 = nn.Linear(50, 50)
      self.fc3 = nn.Linear(50, 50)
      self.fmu = nn.Linear(50, na)
      self.fvar = nn.Linear(50, na)
      init_w = 3e-3
      self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      self.fmu.weight.data.uniform_(-init_w, init_w)
      self.fvar.weight.data.uniform_(-init_w, init_w)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      mu = torch.tanh(self.fmu(x))
      logvar = torch.tanh(self.fvar(x))
      # mu = self.fmu(x)
      # logvar = self.fvar(x)
      return mu, logvar

class critic_net(nn.Module):
  def __init__(self,ns):
      super(critic_net, self).__init__()
      # q1
      self.fc11 = nn.Linear(ns, 50)
      self.fc12 = nn.Linear(50, 50)
      self.fc13 = nn.Linear(50, 50)
      self.fc14 = nn.Linear(50,1)# state value
      # q2
      self.fc21 = nn.Linear(ns, 50)
      self.fc22 = nn.Linear(50, 50)
      self.fc23 = nn.Linear(50, 50)
      self.fc24 = nn.Linear(50,1)# state value

      # init_w = 3e-4
      # self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      # self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      # self.fc3.weight.data = init_weight(self.fc2.weight.data.size())
      # self.fc4.weight.data.uniform_(-init_w, init_w)

  def forward(self, x):
      x1 = F.relu(self.fc11(x))
      x1 = F.relu(self.fc12(x1))
      x1 = F.relu(self.fc13(x1))
      q1 = self.fc14(x1)
      x2 = F.relu(self.fc21(x))
      x2 = F.relu(self.fc22(x2))
      x2 = F.relu(self.fc23(x2))
      q2 = self.fc24(x2)
      return q1, q2

class experience_replay:
  def __init__(self,batch_size):
    self.batch_size = batch_size
    self.memory = []
    self.memory_size = 1e6
    
  def add(self,s,a,s_,r):
    self.memory.append((s, a, s_, r))
    if len(self.memory) > self.memory_size:
        self.memory= self.memory[-self.memory_size:]

  def sample(self):
    exps = random.sample(self.memory,self.batch_size)
    return exps

  def reset(self):
    self.memory.clear()

class SAC:
  def __init__(self,dstates, dactions, env, max_steps, max_episodes, actor_net, critic_net, device):
    self.ns = dstates
    self.na = dactions
    self.max_steps = max_steps
    self.max_episodes = max_episodes
    self.env = env
    self.gamma = 0.99# 割引率
    self.batch_size = 32
    # 学習率
    self.lr_actor_initial = 3e-4
    self.lr_critic_initial = 3e-4
    self.lr_max_steps = 5000

    self.device = device
    
    # actor
    self.actor_net = actor_net
    self.actor_target = copy.deepcopy(actor_net)
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=self.lr_actor_initial,weight_decay=0.01)
    self.actor_criterion = nn.MSELoss()
    # critic
    self.critic_net = critic_net
    self.critic_target = copy.deepcopy(critic_net)
    self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=self.lr_critic_initial)
    self.critic_criterion = nn.MSELoss()

    # experience
    self.exp = experience_replay(self.batch_size)
    self.alpha = 0.2
    self.tau = 0.005

    # log
    self.writer = SummaryWriter(log_dir="./logs")

    self.history=[]

  def get_action(self, s):
      ts = torch.from_numpy(s.astype(np.float32)).clone().to(self.device)
      tmu, tlogvar = self.actor_target.forward(ts)
      tvar = torch.exp(tlogvar)
      tz = torch.randn(self.na).to(self.device)
      # reparameterization trick
      ta = tmu + torch.mul(tvar,tz)
      action = ta.to(device=self.device).detach().cpu().numpy().copy()
      return action

  def get_action_logprob(self, s):
      ts = torch.from_numpy(s.astype(np.float32)).clone().to(self.device)
      tmu, tlogvar = self.actor_target.forward(ts)
      tvar = torch.exp(tlogvar)
      tz = torch.randn(self.na).to(self.device)
      # reparameterization trick
      ta = tmu + torch.mul(tvar,tz)
      logprob = -0.5 * tlogvar
      logprob += -0.5 * (ta - tmu)**2 / (torch.exp(tlogvar) + 1e-8)
      return logprob

  def learn(self):
    t = []
    l = []
    for episode in range(self.max_episodes):
      s = self.env.reset()
      print("initial_state",s)
      total_reward = 0
      for step in range(self.max_steps):
        # 行動決定
        a = self.get_action(s)
        r, s_, fin = self.env.reward(s, a)  # 行動の結果、rewardと状態が帰ってくる
        total_reward += r
        # add memory
        self.exp.add(s,a,s_,r)
        # train
        if len(self.exp.memory) > self.batch_size:
          batches = self.exp.sample()
          for k, batch in enumerate(batches, 0):
            rs,ra,rs_,rr = batch
            trr = torch.from_numpy(np.array([rr]).astype(np.float32)).clone().to(self.device)
            tlogpi_ = self.get_action_logprob(rs_)
            tQ1_, tQ2_ =  self.critic_target.forward(torch.from_numpy(rs_.astype(np.float32)).to(self.device))
            tQ1, tQ2 =  self.critic_net.forward(torch.from_numpy(rs.astype(np.float32)).to(self.device))
            tQtarget = trr + self.gamma * (-tlogpi_*self.alpha + torch.minimum(tQ1_,tQ2_))
            # critic
            self.critic_net.train()
            critic_loss1 = F.mse_loss(tQtarget, tQ1)
            critic_loss2 = F.mse_loss(tQtarget, tQ2)
            critic_loss = critic_loss1 + critic_loss2
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # actor
            tlogpi = self.get_action_logprob(rs)
            tQ1, tQ2 =  self.critic_net.forward(torch.from_numpy(rs.astype(np.float32)).to(self.device))
            tQ = torch.minimum(tQ1, tQ2)
            self.actor_net.train()
            actor_loss = (self.alpha*tlogpi - tQ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update
            for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
                  target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        else:
           pass

        s = copy.deepcopy(s_)
        if fin==1:
          print("\n episode end epidode:",episode,"step=",step,"\n")
          break
      print("total_reward", total_reward)
      if (episode + 1) % 10 == 0:
        torch.save(self.actor_target.state_dict(), "out_SAC/dnn" + str(episode + 1) +".pt")
        print("critic loss=",critic_loss)
      self.writer.add_scalar("total reward", total_reward,episode)
      self.writer.add_scalar("critic_loss", critic_loss, episode)
      self.writer.add_scalar("actor_loss", actor_loss, episode)
        
    #   t.append(episode)
    #   l.append(critic_loss)
    # plt.plot(t, l, '-k')
    # plt.show()
    self.writer.close()

