import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# 初期位置での迷路の様子

# 図を描く大きさと、図の変数名を宣言
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

# 赤い壁を描く
plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)

# 状態を示す文字S0～S8を描く
plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

# 描画範囲の設定と目盛りを消す設定
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')

# 現在地S0に緑丸を描画する
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
plt.show()

class environment:
    def __init__(self):
        # 行は状態0～7、列は移動方向で↑、→、↓、←を表す
        self.maze = np.array([[0, 1, 1, 0],  # s0
                        [0, 1, 0, 1],  # s1
                        [0, 0, 1, 1],  # s2
                        [1, 1, 1, 0],  # s3
                        [0, 0, 1, 1],  # s4
                        [1, 0, 0, 0],  # s5
                        [1, 0, 0, 0],  # s6
                        [1, 1, 0, 0],  # s7、※s8はゴールなので、方策はなし
                        ])
        self.direction = ["up", "right", "down", "left"]
        self.env.reset()
        self.max_steps = 50
        self.states = [0,1,2,3,4,5,6,7,8] 
        self.actions = [0,1,2,3]

    def render(self):
        return

    def close(self):
        return

    def reset(self):
        s = 0
        return s

    def reward(self, s, a):
        fin = 0
        next_direction = self.direction[a]
        r=0
        if self.maze[s,a]==0:
            s_next = s+0
            r=-20
            fin=1
        elif next_direction == "up":
            s_next = s - 3  # 上に移動するときは状態の数字が3小さくなる
        elif next_direction == "right":
            s_next = s + 1  # 右に移動するときは状態の数字が1大きくなる
        elif next_direction == "down":
            s_next = s + 3  # 下に移動するときは状態の数字が3大きくなる
        elif next_direction == "left":
            s_next = s - 1  # 左に移動するときは状態の数字が1小さくなる
        
        if s_next == 8:
            r = 40
            fin = 1

        return r, s_next, fin