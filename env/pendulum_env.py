import sys
import numpy as np
import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import copy
import random

TRAINING = 0
VIDEO = 1
RENDER = 2

class pendulum_env:
  def __init__(self, mode=TRAINING, num=0):
    if mode == TRAINING:
        self.env = gym.make('Pendulum-v1')
    elif mode == VIDEO:
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
