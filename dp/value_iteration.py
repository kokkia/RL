import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# selflibrary 
import sys
sys.path.append("../env")
import maze_env

class value_iteration:
    def __init__(self, dstates, dactions, env, max_steps):
        self.ns = dstates
        self.na = dactions
        self.env = env
        self.max_steps = max_steps
        self.V = np.zeros([self.ns])
        self.Q = np.zeros([self.ns, self.na])
        return 

    def learn(self):
        for step in range(self.max_steps):
            for state in self.env.states:
                Q_max = -1e6
                for action in state.actions:
                    R, next_state, fin = self.env.reward(state, action)
                    Q_tmp = R + self.V[next_state]
                    self.Q[state,action] = Q_tmp
                    if Q_tmp > Q_tmp:
                        Q_max = Q_tmp
                self.V[state] = Q_max
        return

    def get_action(self, state):
        ret_action = 0
        V_max = -1e6
        for action in self.actions:
            R, next_state, fin = self.env.reward(state, action)
            if self.V[next_state] > V_max:
                V_max = copy.deepcopy(self.V[next_state])
                ret_action = action 
        return ret_action
    
env = maze_env.maze_env(maze_env.RENDER)
for i in range(5):
    env.render(i)
plt.show()