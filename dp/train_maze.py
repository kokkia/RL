import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# selflibrary 
sys.path.append("../env")
import maze_env
import value_iteration

env = maze_env.maze_env(maze_env.RENDER)
states = copy.deepcopy(env.states)
actions = copy.deepcopy(env.actions)
dp = value_iteration.value_iteration(states, actions, env, 10, True)
dp.learn()
print("value", dp.V)
print("Q", dp.Q)

fin = 0
state =0
# env.render(state)
for i in range(30):
    action = dp.get_action(state)
    print(action)
    r, next_state, fin = env.reward(state, action)
    state = next_state
    # env.render(state)
    if fin == 1:
        break
print("fin", fin)
dp.plot_state_value(3,3)
plt.show()