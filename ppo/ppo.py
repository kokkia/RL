
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random
import scipy

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
      self.fc1 = nn.Linear(ns, 50)
      self.fc2 = nn.Linear(50, 50)
      # self.fc3 = nn.Linear(50, na*2) # mu1, mu2, sigma1, sigma2
      self.fmu = nn.Linear(50, na)
      self.fvar = nn.Linear(50, na)
      init_w = 3e-3
      self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      # self.fc3.weight.data.uniform_(-init_w, init_w)
      self.fmu.weight.data.uniform_(-init_w, init_w)
      self.fvar.weight.data.uniform_(-init_w, init_w)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      # x = torch.tanh(self.fc3(x))
      mu = torch.tanh(self.fmu(x))
      # mu = self.fmu(x)
      logvar = self.fvar(x)
      return mu, logvar

class critic_net(nn.Module):
  def __init__(self,ns):
      super(critic_net, self).__init__()
      self.fc1 = nn.Linear(ns, 50)
      self.fc2 = nn.Linear(50, 50)
      self.fc3 = nn.Linear(50,1)# state value

      init_w = 3e-4
      self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
      self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
      self.fc3.weight.data.uniform_(-init_w, init_w)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x

class PPO:
  def __init__(self,dstates, dactions, env, max_steps, max_episodes, actor_net, critic_net, device):
    self.ns = dstates
    self.na = dactions
    self.max_steps = max_steps
    self.max_episodes = max_episodes
    self.env = env
    self.gamma = 0.99# 割引率
    # self.tau = 0.001# ターゲット更新率
    self.batch_size = 32
    self.lr_actor = 1e-4
    self.lr_critic = 1e-4
    # self.epsilon = 1.0

    self.device = device
    
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

    # experience
    self.experiences = []

    # log
    self.writer = SummaryWriter(log_dir="./logs")

    self.history=[]

  def get_action(self, s):
      ts = torch.from_numpy(s.astype(np.float32)).clone()
      ts = ts.to(self.device)
      tmu, tlogvar = self.actor_net.forward(ts)
      mu = tmu.to(device=self.device).detach().cpu().numpy().copy()
      logvar = tlogvar.to(device=self.device).detach().cpu().numpy().copy()
      log_cov = np.diag(logvar)
      cov = np.exp(log_cov)
      action = np.random.multivariate_normal(mean=mu, cov=cov, size=self.na).reshape(-1)
      return action

  def get_action_logprob(self, s, a):
      ta = torch.from_numpy(a.astype(np.float32)).clone()
      ta = ta.to(self.device)
      ts = torch.from_numpy(s.astype(np.float32)).clone()
      ts = ts.to(self.device)
      tmu, tlogvar = self.actor_net.forward(ts)
      logprob = -0.5 * tlogvar
      logprob += -0.5 * (ta - tmu)**2 / torch.exp(tlogvar)
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
        # print("s,a,s_,r = ",s,a,s_,r)
        # add experience
        self.experiences.append((s,a,s_,r,fin))
        # train
        if len(self.experiences) == self.batch_size or fin == True:
          batch_size = len(self.experiences)
          if fin == True:
            R = 0
          else:
            s_ = self.experiences[-1][2]
            R = self.critic_net.forward(torch.from_numpy(s_.astype(np.float32)).to(self.device)).detach().mean()
          for i in reversed(range(batch_size)):
            # 割引報酬計算
            r_i = self.experiences[i][3]
            r_i = torch.from_numpy(np.array([r_i]).astype(np.float32)).to(self.device).mean()
            R = r_i + self.gamma * R
            s_i = self.experiences[i][0]
            a_i = self.experiences[i][1]
            logprob = self.get_action_logprob(s_i, a_i)
            V = self.critic_net.forward(torch.from_numpy(s_i.astype(np.float32)).to(self.device)).mean()
            # network更新
            self.actor_net.train()
            self.critic_net.train()
            # critic
            critic_loss = F.mse_loss(R, V)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # actor
            advantage = R - V
            actor_loss = - logprob * advantage.detach()
            actor_loss = actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
          self.experiences = []
        else:
          pass

        s = copy.deepcopy(s_)
        if fin==1:
          print("\n episode end epidode:",episode,"step=",step,"\n")
          break
      print("total_reward", total_reward)
      if (episode + 1) % 10 == 0:
        torch.save(self.actor_target.state_dict(), "out_PPO/dnn" + str(episode + 1) +".pt")
        print("critic loss=",critic_loss)
      self.writer.add_scalar("total reward", total_reward,episode)
      self.writer.add_scalar("critic_loss", critic_loss, episode)
      self.writer.add_scalar("actor_loss", actor_loss, episode)
        
    #   t.append(episode)
    #   l.append(critic_loss)
    # plt.plot(t, l, '-k')
    # plt.show()
    self.writer.close()

