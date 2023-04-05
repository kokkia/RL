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

class frozen_lake_env:
  def __init__(self, mode=TRAINING, num=0):
    map=["SFFF", "FFFF", "FFFF", "FFFG"]
    if mode == TRAINING:
        self.env = gym.make('FrozenLake-v1', desc=map)
    elif mode == VIDEO:
        self.env = RecordVideo(gym.make("FrozenLake-v1", desc=map, render_mode="rgb_array"),"video/lake"+str(num))
    else:
        self.env = gym.make('FrozenLake-v1',desc=map,render_mode="human")
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
    #   self.env.render()
      s_, r, terminated, truncated, info = self.env.step(int(a))
      print(s,a,s_,r)
      fin = terminated or truncated
      return r, s_, fin