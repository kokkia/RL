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

import reinforce
import gym

env = gym.make("Pendulum-v0")
env.reset()

i_epi = 0
def reward(s, a):
    global i_epi
    # env.render()
    i_epi += 1
    s_, r, fin, info = env.step(action=a)
    # r += s_[1]
    # r += - abs(s_[0])/32.0
    r = r/10

    # print(np.array(s_))
    if fin == 1:
      # if i_epi < 50:
      #     r += -1
      # i_epi = 0
      # plt.pause(0.01)
      s= env.reset()
    return r, np.array(s_), fin


a_net = reinforce.actor_net(3,2)
c_net = reinforce.critic_net(3,1)
rl = reinforce.REINFORCE(3,1,np.array(env.reset()),reward,100,a_net,c_net)
rl.learn()
print(td.Q)

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