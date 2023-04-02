import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

class salary_man_env:
    def __init__(self):
        return
    
    def reward(self, s, a):
        # 報酬期待値の設定
        r = np.zeros((3, 3, 2))
        r[0, 1, 0] = 1.0
        r[0, 2, 0] = 2.0
        r[0, 0, 1] = 0.0
        r[1, 0, 0] = 1.0
        r[1, 2, 0] = 2.0
        r[1, 1, 1] = 1.0
        r[2, 0, 0] = 1.0
        r[2, 1, 0] = 0.0
        r[2, 2, 1] = -1.0
        return 
    
    def reset(self):
        return