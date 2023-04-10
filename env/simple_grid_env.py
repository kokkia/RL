import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import random

TRAINING = 0
VIDEO = 1
RENDER = 2

class simple_grid_env:
    def __init__(self, mode=TRAINING, size=4, fig=None, ax=None):
        # 行は状態0～7、列は移動方向で↑、→、↓、←を表す
        self.max_steps = 50
        self.states = np.arange(size**2)
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.size = size
        self.ns = size**2
        self.na = len(self.actions)
        self.reset()

        self.s_goal = self.ns-1
        self.R = np.zeros([self.ns])
        self.R[-21] = -400
        self.R[-32] = -400
        self.R[self.s_goal] = 40

        # fig
        self.mode = mode
        self.fig = fig
        self.ax = ax

    def set_reward(self, R):
        self.R = R
        return

    def render(self, state):
        self.ax.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        self.draw_state(state)
        plt.pause(0.1)
        return

    def close(self):
        return

    def reset(self):
        s = 0
        return s

    def reward(self, s, a):
        fin = 0
        s_next = self.state_index_transition(s, a)
        r = self.R[s_next]
        if s_next == self.s_goal:
            fin = 1

        return r, s_next, fin
    
    def draw_reward(self):
        if self.fig == None:
            return
        p = self.ax.imshow(np.reshape(self.R, (self.size, self.size)))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(p, cax=cax)
        plt.pause(1.0)
        return

    def draw_state(self, state):
        if self.fig == None:
            return
        states = np.zeros([self.ns]) 
        states[state] = 1
        p = self.ax.imshow(np.reshape(states, (self.size, self.size)))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(p, cax=cax)
        return



    def state_index_to_point(self, state):
        """
        Convert a state index to the coordinate representing it.
        Args:
            state: Integer representing the state.
        Returns:
            The coordinate as tuple of integers representing the same state
            as the index.
        """
        return state % self.size, state // self.size

    def state_point_to_index(self, state):
        """
        Convert a state coordinate to the index representing it.
        Note:
            Does not check if coordinates lie outside of the world.
        Args:
            state: Tuple of integers representing the state.
        Returns:
            The index as integer representing the same state as the given
            coordinate.
        """
        return state[1] * self.size + state[0]

    def state_point_to_index_clipped(self, state):
        """
        Convert a state coordinate to the index representing it, while also
        handling coordinates that would lie outside of this world.
        Coordinates that are outside of the world will be clipped to the
        world, i.e. projected onto to the nearest coordinate that lies
        inside this world.
        Useful for handling transitions that could go over an edge.
        Args:
            state: The tuple of integers representing the state.
        Returns:
            The index as integer representing the same state as the given
            coordinate if the coordinate lies inside this world, or the
            index to the closest state that lies inside the world.
        """
        s = (max(0, min(self.size - 1, state[0])), max(0, min(self.size - 1, state[1])))
        return self.state_point_to_index(s)

    def state_index_transition(self, s, a):
        """
        Perform action `a` at state `s` and return the intended next state.
        Does not take into account the transition probabilities. Instead it
        just returns the intended outcome of the given action taken at the
        given state, i.e. the outcome in case the action succeeds.
        Args:
            s: The state at which the action should be taken.
            a: The action that should be taken.
        Returns:
            The next state as implied by the given action and state.
        """
        s = self.state_index_to_point(s)
        s = s[0] + self.actions[a][0], s[1] + self.actions[a][1]
        return self.state_point_to_index_clipped(s)



