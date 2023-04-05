import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# selflibrary 
sys.path.append("../env")
import frozen_lake_env
import value_iteration

env = frozen_lake_env.frozen_lake_env(frozen_lake_env.RENDER)
states = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
actions = [0,1,2,3]
dp = value_iteration.value_iteration(states, actions, env, 30)
dp.learn()
print("value", dp.V)
print("Q", dp.Q)

# fin = 0
# state = env.reset()
# for i in range(30):
#     action = dp.get_action(state)
#     # print(action)
#     r, next_state, fin = env.reward(state, action)
#     state = next_state
#     env.render()
#     if fin == 1:
#         break
# print("fin", fin)

# plt.show()