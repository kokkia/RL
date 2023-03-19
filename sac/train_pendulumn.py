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

# selflibrary 
import sys
sys.path.append("../env")
import pendulum_env
import sac

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

env = pendulum_env.pendulum_env()
max_steps = env.max_steps
max_episodes = 500
a_net = sac.actor_net(3,1).to(device)
c_net = sac.critic_net(3).to(device)
rl = sac.SAC(3,1,env,max_steps, max_episodes,a_net,c_net,device)
rl.learn()
# print(td.a)

env.close()
