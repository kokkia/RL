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
import ddpg

def reward(s,a):
    return

num = sys.argv[1]

# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
env = gym.make("Pendulum-v0")
env.reset()

s = env.reset()
a_net = ddpg.actor_net(3,1)
c_net = ddpg.critic_net(4,1)
a_net.load_state_dict(torch.load("out_RF/dnn"+str(num)+".pt"))
rl = ddpg.DDPG(3,1,np.array(env.reset()),reward,100,a_net,c_net)

for i in range(1000):
    a = rl.action(s)
    s_, r, fin, info = env.step(a)
    s = copy.deepcopy(s_)
    env.render()
print(s)

env.close()


