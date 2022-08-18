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

import gym
import pybulletgym.envs

env = gym.make('Walker2DPyBulletEnv-v0')
for _ in range(1):
    state = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state_new, r, done, info = env.step(action)
        print("reward: ", r)
        if done:
            print('episode done')
            break
