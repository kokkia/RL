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
    def __init__(self, states, actions, env, max_steps):
        self.states = states
        self.actions = actions
        self.ns = len(states)
        self.na = len(actions)
        self.env = env
        self.max_steps = max_steps
        self.V = np.zeros([self.ns])
        self.Q = np.zeros([self.ns, self.na])
        self.gamma = 0.95
        return 

    def learn(self):
        for step in range(self.max_steps):
            for state in self.states:
                Q_max = -1e6
                for action in self.actions:
                    R, next_state, fin = self.env.reward(state, action)
                    # print(R)
                    Q_tmp = R + self.gamma * self.V[next_state]
                    self.Q[state,action] = Q_tmp
                    if Q_tmp > Q_max:
                        Q_max = Q_tmp
                self.V[state] = Q_max
        return

    def get_action(self, state):
        ret_action = 0
        ret_action = np.argmax(self.Q[state,:])
        # V_max = -1e6
        # for action in self.actions:
        #     R, next_state, fin = self.env.reward(state, action)
        #     if self.V[next_state] > V_max:
        #         V_max = copy.deepcopy(self.V[next_state])
        #         ret_action = action 
        return ret_action
    
env = maze_env.maze_env(maze_env.RENDER)
states = copy.deepcopy(env.states)
actions = copy.deepcopy(env.actions)
dp = value_iteration(states, actions, env, 30)
dp.learn()
print("value", dp.V)
print("Q", dp.Q)

fin = 0
state = 6
env.render(state)
for i in range(30):
    action = dp.get_action(state)
    print(action)
    r, next_state, fin = env.reward(state, action)
    state = next_state
    env.render(state)
    if fin == 1:
        break
print("fin", fin)

plt.show()