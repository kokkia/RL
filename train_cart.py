import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ddqn 

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')


i_epi = 0
def reward(s, a):
    global i_epi
    env.render()
    i_epi += 1
    s_, r, fin, info = env.step(a)
    r += (1 - abs(s[2]))
    r += 1 - abs(s[0])
    r+=float(i_epi)/50
    

    # print(np.array(s_))
    if fin == 1:
      if i_epi < 100:
          r += -1
      i_epi = 0
      plt.pause(0.01)
      s= env.reset()
    return r, np.array(s_), fin
    
def reward2(s, a):
    global i_epi
    env.render()
    i_epi += 1
    s_, r, fin, info = env.step(a)
    r += (1 - abs(s[2]))
    r += 1 - abs(s[0])
    # r+=float(i_epi)/200
    r = (r-3.0)*5

    # print(np.array(s_))
    if fin == 1:
      if i_epi < 100:
          r += -1
      i_epi = 0
      plt.pause(0.01)
      s= env.reset()
    return r, np.array(s_), fin

Qnet = ddqn.Net(4,2)
td = ddqn.DDQN(4,np.array([0,1]),np.array(env.reset()),reward2,0.2, 200,Qnet)
td.learn()
print(td.Q)

env.close()


