import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# selflibrary 
sys.path.append("../env")
import simple_grid_env
sys.path.append("../dp")
import value_iteration
import max_ent_deep_irl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fig
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
size = 10 
env = simple_grid_env.simple_grid_env(simple_grid_env.RENDER, size, fig, ax1)
env.draw_reward()
states = copy.deepcopy(env.states)
actions = [0, 1, 2, 3]

# dp
dp = value_iteration.value_iteration(states, actions, env, 10, fig, ax2)
dp.learn()
print("value", dp.V)
print("Q", dp.Q)
dp.plot_state_value(size,size)

# collect data
trajs = []
traj = []
fin = 0
# env.render(state)
for state in states:
    traj.append(state)
    for i in range(20):
        action = dp.get_action(state)
        # print(action)
        r, next_state, fin = env.reward(state, action)
        state = next_state
        # env.render(state)
        traj.append(state)
        if fin == 1:
            break
    trajs.append(traj)
print("fin", fin)

# net
r_net = max_ent_deep_irl.reward_net(1).to(device)
# IRL
irl = max_ent_deep_irl.max_ent_deep_irl(states, actions, env, 40, r_net, device, fig, ax2)
irl.set_expert_trajectory(trajs)
irl.learn()
irl.plot_reward(size, size)

plt.show()