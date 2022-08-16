import sys
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

def reward2():
      return

num = sys.argv[1]

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')

s = env.reset()
Qnet = ddqn.Net(4,2)
Qnet.load_state_dict(torch.load("out/dnn"+str(num)+".pt"))
td = ddqn.DDQN(4, np.array([0, 1]), np.array(env.reset()), reward2, 0.2, 200, Qnet)

for i in range(1000):
    a = td.action(s)
    s_, r, fin, info = env.step(a)
    s = copy.deepcopy(s_)
    env.render()
print(s)

env.close()


