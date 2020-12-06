
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
  def __init__(self,ns,na):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(ns, 30)
      self.dropout1 = nn.Dropout2d()
      self.fc2 = nn.Linear(30, 30)
      self.dropout2 = nn.Dropout2d()
      self.fc3 = nn.Linear(30,na)

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

# 目標：stateとactionとネットワークとrewardを設定すればどんな環境でも強化学習できるクラスをつくる
# @todo:actionの次元を複数次元に対応する
# @todo:現状うまく学習してない
class DDQN:
  def __init__(self,dstates, actions, s0, reward, epsilon, nepisode,net):
    # self.states = states
    self.actions = actions
    self.ns = dstates
    self.na = len(actions)
    self.epsilon = epsilon
    self.nepisode = nepisode
    self.r = reward
    self.s0 = s0
    self.alpha = 0.5
    self.gamma = 0.90#0.99
    
    self.history=[]
    self.mainnet = net
    self.targetnet = copy.deepcopy(net)
    self.optimizer = optim.Adam(self.mainnet.parameters(),lr=0.00001)
    self.criterion = nn.MSELoss()

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
        if np.random.uniform(0,1)<self.epsilon:
          a = np.random.choice(self.actions)
        else:
          ts = torch.from_numpy(s.astype(np.float32)).clone()
          self.Q = self.targetnet.forward(ts)
          self.Q = self.Q.to('cpu').detach().numpy().copy()
          a = np.argmax(self.Q)
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
          self.mainnet.eval()
          self.targetnet.eval()
          ts = torch.from_numpy(rs.astype(np.float32)).clone()
          ts_ = torch.from_numpy(rs_.astype(np.float32)).clone()
          Qs = self.mainnet.forward(ts)
          Qs_ = self.targetnet.forward(ts_).detach()
          Qr = rr+self.gamma*Qs_.max()
          # loss = self.criterion(Qr,Qs[ra])
          # 学習
          self.mainnet.train()
          loss = F.smooth_l1_loss(Qr,Qs[ra])
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
        if j%20==0:# taget networkを更新
          # self.targetnet=copy.deepcopy(self.mainnet)
          self.targetnet.load_state_dict(self.mainnet.state_dict())
          
        # train
        # self.Q[s,a] += self.alpha*(r+self.gamma*np.max(self.Q[s_,:])-self.Q[s,a])# TD学習
        s = copy.deepcopy(s_)
        self.history.append(s)
        if fin==1:
          print("\n episode end epidode:",i,"j=",j,"\n")
        #   plt.pause(1)
          break
        # print(self.Q)
      if (i + 1) % 100 == 0:
        torch.save(self.targetnet.state_dict(), "out/dnn" + str(i + 1) +".pt")
        print("loss=",loss)
        
      t.append(i)
      l.append(loss)
    plt.plot(t, l, '-k')
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

# i_epi = 0

# env = gym.make('CartPole-v0')
# # env = gym.make('MountainCar-v0')
# def reward(s, a):
#     global i_epi
#     env.render()
#     i_epi += 1
#     s_, r, fin, inf0 = env.step(a)
#     r += (1 - abs(s[2]))
#     r += 1 - abs(s[0])
#     r+=float(i_epi)/10

#     # print(np.array(s_))
#     if fin == 1:
#       i_epi = 0
#       plt.pause(0.01)
#       s= env.reset()
#     return r, np.array(s_), fin
    
# Qnet = Net(4,2)
# td = DDQN(1,np.array([0,1]),np.array(env.reset()),reward,0.2, 200,Qnet)
# td.learn()
# print(td.Q)

# env.close()


