import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# selflibrary 
sys.path.append("../env")
import simple_grid_env
import value_iteration

# fig
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
size =10 
env = simple_grid_env.simple_grid_env(simple_grid_env.RENDER, size, fig, ax1)
# env.draw_reward()
states = copy.deepcopy(env.states)
actions = [0, 1, 2, 3]
dp = value_iteration.value_iteration(states, actions, env, 40, fig, ax2)
dp.learn()
print("value", dp.V)
print("Q", dp.Q)

dp.plot_state_value(size,size)
fin = 0
state =0
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