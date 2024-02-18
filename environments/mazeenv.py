#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:24:24 2024

@author: brendandevlin-hill
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeEnv(gym.Env):

    def __init__(self,
                 size,
                 goal,
                 walls,
                 render_mode=None):

        self.size = size
        self.goal = goal
        self.walls = walls
        self.render_mode = render_mode
        self.initial_state = {"x" : 0, "y" : 0}
        self.state = self.initial_state.copy()

        obDict = {}
        # obDict["north_blocked"] = spaces.Discrete(1)
        # obDict["west_blocked"] = spaces.Discrete(1)
        # obDict["south_blocked"] = spaces.Discrete(1)
        # obDict["east_blocked"] = spaces.Discrete(1)
        obDict["x"] = spaces.Discrete(self.size)
        obDict["y"] = spaces.Discrete(self.size)
        self.observation_space = spaces.Dict(obDict)

        self.action_space = spaces.Discrete(4)  # move in each cardinal direction

    def reset(self, options=None):
        if(options):
            if(options["type"] == "state"):
                # assert options["state"]
                # print(options["state"])
                self.state = options["state"].copy()
        else:
            self.state = self.initial_state.copy()
        self.elapsed_steps = 0

        return self.state.copy(), self.get_info()

    def get_info(self):
        info = {"elapsed steps" : self.elapsed_steps, "pseudoreward" : self.pseudoreward_function()}
        return info

    def pseudoreward_function(self):
        return -((self.goal[0] - self.state["x"])**2 + (self.goal[0] - self.state["y"]))**2

    def step(self, action):

        self.elapsed_steps += 1
        reward = 0

        new_state = self.state.copy()

        if action == 0:  # north
            if([self.state["x"], self.state["y"] + 1] in self.walls):
                reward = -1
            else:
                new_state["y"] += 1
        if action == 1:  # east
            if([self.state["x"] + 1, self.state["y"]] in self.walls):
                reward = -1
            else:
                new_state["x"] += 1
        if action == 2:  # south
            if([self.state["x"], self.state["y"] - 1] in self.walls):
                reward = -1
            else:
                new_state["y"] -= 1
        if action == 3:  # west
            if([self.state["x"] - 1, self.state["y"]] in self.walls):
                reward = -1
            else:
                new_state["x"] -= 1

        if (new_state["x"] < 0 or new_state["x"] > self.size or new_state["y"] < 0 or new_state["y"] > self.size):
            reward = -1
        else:
            self.state = new_state.copy()

        terminated = False
        if (list(self.state.values()) == self.goal):
            terminated = True
            reward = 100

        info = self.get_info()

        return self.state.copy(), reward, terminated, False, info
