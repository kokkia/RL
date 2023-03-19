import sys
import numpy as np
import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sac

num = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
make_video = False

class environment:
  def __init__(self):
    if make_video:
        self.env = RecordVideo(gym.make("Pendulum-v1",render_mode="rgb_array"),"video/pendulumn"+str(num))
    else:
        self.env = gym.make('Pendulum-v1',render_mode="human")
    self.env.reset()
    self.max_steps = self.env.spec.max_episode_steps

  def render(self):
    self.env.render()

  def close(self):
    self.env.close()

  def reset(self):
    s = self.env.reset()[0]
    return s

  def reward(self, s, a):
      # env.render()
      a = a*2
      s_, r, terminated, truncated, info = self.env.step(a)
      fin = terminated or truncated
      return r, np.array(s_), fin

# 環境初期化
env = environment()
max_steps = env.max_steps
max_episodes = 200
s = env.reset()

# network読み込み
a_net = sac.actor_net(3,1).to(device)
c_net = sac.critic_net(3).to(device)
a_net.load_state_dict(torch.load("out_SAC/dnn"+str(num)+".pt"))
rl = sac.SAC(3,1,env,max_steps, max_episodes,a_net,c_net,device)

# play
total_reward = 0
for i in range(max_steps):
    # a = rl.get_action(s, greedy=True)
    a = rl.get_action(s)
    r, s_, fin = env.reward(s,a)
    s = copy.deepcopy(s_)
    if not make_video:
        env.render()
    total_reward += r
print(s, total_reward)

env.close()


