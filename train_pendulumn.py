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

import ddpg
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
      s_, r, terminated, truncated, info = self.env.step(a)
      fin = terminated or truncated
      return r, np.array(s_), fin

env = environment()
max_steps = env.max_steps
max_episodes = 200
s0 = env.reset()
a_net = ddpg.actor_net(3,1).to(device)
c_net = ddpg.critic_net(3,1).to(device)
rl = ddpg.DDPG(3,1,s0,env,max_steps, max_episodes,a_net,c_net,device)
rl.learn()
# print(td.a)

env.close()


# is_finish = False   # 終了判定

# nb_step = 1
# while(1):
#     env.render()
#     random_action = env.action_space.sample()   # ランダムで行動を選択。0, 1, 2のどれか
#     obs, reward, is_finish, _ = env.step(action=random_action)  
#     print("nb_step:{}, action:{}, obs:{}, reward:{}, is_finish:{}".format(nb_step, random_action, obs, reward, is_finish))
#     nb_step += 1
#     if is_finish == True:   # 終了したら、whileを抜ける
#         break