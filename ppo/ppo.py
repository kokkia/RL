
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
      self.fc1 = nn.Linear(ns, 50)
      self.fc2 = nn.Linear(50, 50)
      # self.fc3 = nn.Linear(50, na*2) # mu1, mu2, sigma1, sigma2
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
      # x = torch.tanh(self.fc3(x))
      # mu = torch.tanh(self.fmu(x))
      mu = self.fmu(x)
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
    self.batch_size = 32
    # 学習率
    self.lr_actor_initial = 1e-4
    self.lr_actor_final = 0.1e-4
    self.lr_critic_initial = 1e-4
    self.lr_critic_final = 1e-4
    self.lr_max_steps = 500

    self.device = device
    
    # actor
    self.actor_net = actor_net
    self.actor_target = copy.deepcopy(actor_net)
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=self.lr_actor_initial)
    self.actor_criterion = nn.MSELoss()
    # critic
    self.critic_net = critic_net
    self.critic_target = copy.deepcopy(critic_net)
    self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=self.lr_critic_initial)
    self.critic_criterion = nn.MSELoss()

    # advantage 
    self.advantage_steps = 5
    self.reward_que = deque(maxlen=self.advantage_steps)
    self.state_que = deque(maxlen=self.advantage_steps)
    self.td_err_que = deque(maxlen=self.advantage_steps)

    self.clip_range = 0.2
    self.lmd = 0.95

    # experience
    self.experiences = []

    # log
    self.writer = SummaryWriter(log_dir="./logs")

    self.history=[]

  def get_action(self, s):
      ts = torch.from_numpy(s.astype(np.float32)).clone()
      ts = ts.to(self.device)
      tmu, tlogvar = self.actor_net.forward(ts)
      tvar = torch.exp(tlogvar)
      ta = torch.normal(tmu, tvar)
      action = ta.to(device=self.device).detach().cpu().numpy().copy()
      return action

  def get_action_logprob(self, s, a, actor_net):
      ta = torch.from_numpy(a.astype(np.float32)).clone()
      ta = ta.to(self.device)
      ts = torch.from_numpy(s.astype(np.float32)).clone()
      ts = ts.to(self.device)
      tmu, tlogvar = actor_net.forward(ts)
      logprob = -0.5 * tlogvar
      logprob += -0.5 * (ta - tmu)**2 / (torch.exp(tlogvar) + 1e-8)
      return logprob

  def scheduling_adam_lr(self, cnt):
      if cnt >= self.lr_max_steps:
        lr_actor = self.lr_actor_final
        lr_critic = self.lr_critic_final
      else:
        lr_actor = self.lr_actor_initial + (self.lr_actor_final - self.lr_actor_initial)*float(cnt)/self.lr_max_steps
        lr_critic = self.lr_actor_initial + (self.lr_actor_final - self.lr_actor_initial)*float(cnt)/self.lr_max_steps
      self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
      self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=lr_critic)

  def learn(self):
    t = []
    l = []
    for episode in range(self.max_episodes):
      s = self.env.reset()
      print("initial_state",s)
      total_reward = 0
      rewards = []
      for step in range(self.max_steps):
        # 行動決定
        a = self.get_action(s)
        r, s_, fin = self.env.reward(s, a)  # 行動の結果、rewardと状態が帰ってくる
        total_reward += r
        rewards.append(r)
        # print("s,a,s_,r = ",s,a,s_,r)
        # add experience
        self.experiences.append((s,a,s_,r,fin))
        # train
        if len(self.experiences) == self.batch_size or fin == True:
          batch_size = len(self.experiences)
          # deque reset
          self.reward_que = deque(maxlen=self.advantage_steps)
          self.state_que = deque(maxlen=self.advantage_steps)
          # 初期値計算
          if fin == True:
            R = 0
          else:
            s_ = self.experiences[-1][2]
            R = self.critic_net.forward(torch.from_numpy(s_.astype(np.float32)).to(self.device)).detach().mean()
          # rewardsの標準偏差計算
          reward_std = np.std(np.array(rewards))
          # 過去の方策を保存
          actor_net_old = copy.deepcopy(self.actor_net)
          for i in reversed(range(batch_size)):
            # batchから取得
            s_i = self.experiences[i][0]
            a_i = self.experiences[i][1]
            s_i_ = self.experiences[i][2]
            r_i = self.experiences[i][3]
            
            # reward_scaling
            r_i = r_i/(reward_std+1e-8)

            # queに追加
            r_i = torch.from_numpy(np.array([r_i]).astype(np.float32)).to(self.device).mean()
            self.reward_que.appendleft(r_i)
            self.state_que.appendleft(s_i)
            
            # TD
            V_st = self.critic_net.forward(torch.from_numpy(s_i.astype(np.float32)).to(self.device)).mean()
            V_st_ = self.critic_net.forward(torch.from_numpy(s_i_.astype(np.float32)).to(self.device)).mean()
            td_err = r_i + self.gamma * V_st_ - V_st
            self.td_err_que.appendleft(td_err)
            
            # steps未満であれば学習開始しない
            if len(self.reward_que) < self.advantage_steps:
               continue
            
            # TD(n)errorの計算
            s_ik = self.state_que[-1]
            V_target = 0.0
            for step in range(self.advantage_steps):
               V_target += pow(self.gamma,step) * self.reward_que[step]
            V_target += pow(self.gamma, self.advantage_steps)*self.critic_net.forward(torch.from_numpy(s_ik.astype(np.float32)).to(self.device)).mean()
            
            # Generalized Advantage Estimation
            advantage = 0.0
            for step in range(self.advantage_steps):
              advantage += pow((self.lmd * self.gamma), step) * self.td_err_que[step]
            
            # cliped surrogate objectives
            logprob = self.get_action_logprob(s_i, a_i, self.actor_net)
            logprob_old = self.get_action_logprob(s_i, a_i, actor_net_old).detach()
            ratio = torch.exp(logprob-logprob_old)
            ratio_clipped = torch.clip(ratio,1-self.clip_range, 1+self.clip_range)
            
            # critic
            V = self.critic_net.forward(torch.from_numpy(s_i.astype(np.float32)).to(self.device)).mean()
            self.critic_net.train()
            critic_loss = F.mse_loss(V_target, V)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # actor
            self.actor_net.train()
            actor_loss = - torch.minimum(ratio*advantage.detach(),ratio_clipped*advantage.detach()).mean()
            actor_loss = actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
          self.experiences = []
          rewards = []
        else:
          pass

        s = copy.deepcopy(s_)
        if fin==1:
          print("\n episode end epidode:",episode,"step=",step,"\n")
          break
      self.scheduling_adam_lr(episode)
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

