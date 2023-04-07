import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

TRAINING = 0
VIDEO = 1
RENDER = 2

class maze_env:
    def __init__(self, mode=TRAINING, fig=None, ax=None):
        # 行は状態0～7、列は移動方向で↑、→、↓、←を表す
        self.maze = np.array([[0, 1, 1, 0, 0],  # s0
                        [0, 1, 0, 1, 0],  # s1
                        [0, 0, 1, 1, 0],  # s2
                        [1, 1, 1, 0, 0],  # s3
                        [0, 0, 1, 1, 0],  # s4
                        [1, 0, 0, 0, 0],  # s5
                        [1, 0, 0, 0, 0],  # s6
                        [1, 1, 0, 0, 0],  # s7、※s8はゴールなので、方策はなし
                        [0, 0, 0, 1, 0],  # s7、※s8はゴールなので、方策はなし
                        ])
        self.direction = ["up", "right", "down", "left", "stay"]
        self.reset()
        self.max_steps = 50
        self.states = [0,1,2,3,4,5,6,7,8] 
        self.actions = [0,1,2,3,4]
        self.ns = 9
        self.na = 4

        # fig
        self.mode = mode
        self.fig = fig
        self.ax = ax
        self.setup_figure()

    def render(self, state):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        self.setup_figure(state)
        plt.pause(0.1)
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
            fin=-1
        elif next_direction == "up":
            s_next = s - 3  # 上に移動するときは状態の数字が3小さくなる
        elif next_direction == "right":
            s_next = s + 1  # 右に移動するときは状態の数字が1大きくなる
        elif next_direction == "down":
            s_next = s + 3  # 下に移動するときは状態の数字が3大きくなる
        elif next_direction == "left":
            s_next = s - 1  # 左に移動するときは状態の数字が1小さくなる
        elif next_direction == "stay":
            s_next = s 
        
        if s_next == 8:
            r = 40
            fin = 1

        return r, s_next, fin

    def setup_figure(self, state=0):
        # 図を描く大きさと、図の変数名を宣言
        # self.ax = plt.gca()
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
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 3)
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        labelbottom='off', right='off', left='off', labelleft='off')

        # 現在地S0に緑丸を描画する
        x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
        y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
        # line, = self.ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
        line, = self.ax.plot(x, y, marker="o", color='g', markersize=60)
        return


