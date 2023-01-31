import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ppo
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class environment:
  def __init__(self):
    self.env = gym.make("Pendulum-v1")
    self.env.reset()
    self.max_steps = self.env.spec.max_episode_steps

  def reset(self):
    s = self.env.reset()[0]
    return s

  def reward(self, s, a):
      # env.render()
      a = a * 2# 正規化を戻す
      s_, r, terminated, truncated, info = self.env.step(a)
      fin = terminated or truncated
      return r, np.array(s_), fin

env = environment()
max_steps = env.max_steps
max_episodes = 2000
a_net = ppo.actor_net(3,1).to(device)
c_net = ppo.critic_net(3).to(device)
rl = ppo.PPO(3,1,env,max_steps, max_episodes,a_net,c_net,device)
rl.learn()
# print(td.a)

env.close()
