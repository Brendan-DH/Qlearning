#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:34:49 2023

@author: brendandevlin-hill
"""

import numpy as np
import gymnasium as gym
import system_logic.hybrid_system_tensor_logic
from gymnasium import spaces
import pygame

import torch

class TokamakEnvMA1(gym.Env):
    metadata = {"render_modes": [], "render_fps": 30}

    # def set_parameters(size, num_active, num_goals, goal_locations):
    #     return None

    def set_rendering(self, rendering):
        self.render = rendering

    def __init__(self,
                 system_parameters,
                 transition_model,
                 blocked_model,
                 reward_model,
                 pseudoreward_function=None,
                 initial_state_logic=None,
                 training=True,
                 render=False):

        # operational parameters

        initial_state_dict = {}

        self.transition_model = transition_model
        self.reward_model = reward_model
        self.blocked_model = blocked_model
        self.pseudoreward_function = pseudoreward_function
        self.initial_state_logic = initial_state_logic
        self.runstates = []
        self.statetree = []
        self.render = render

        self.frame_counter = 0

        initial_state_dict["clock"] = 0  # the clock is the index of the robot that is currently active
        for i in range(len(system_parameters.robot_locations)):
            initial_state_dict[f"robot{i} location"] = system_parameters.robot_locations[i]
        for i in range(len(system_parameters.goal_locations)):
            initial_state_dict[f"goal{i} location"] = system_parameters.goal_locations[i]
            initial_state_dict[f"goal{i} active"] = system_parameters.goal_activations[i]  # 1 = goal should be engaged
            initial_state_dict[f"goal{i} checked"] = system_parameters.goal_checked[i]  # 1 = goal location has been visited
            initial_state_dict[f"goal{i} discovery probability"] = system_parameters.goal_discovery_probabilities[i]  # p that goal is present
            initial_state_dict[f"goal{i} completion probability"] = system_parameters.goal_completion_probabilities[i]  # p that goal is completed in 1 attempt

        self.state_dict = initial_state_dict.copy()
        self.initial_state_dict = initial_state_dict.copy()

        self.size = system_parameters.size
        self.elapsed_steps = 0
        self.start_locations = np.array(system_parameters.robot_locations.copy())
        self.num_goals = len(system_parameters.goal_locations)
        self.num_robots = len(system_parameters.robot_locations)

        if initial_state_logic:
            print("Taking care of initial state transitions")
            self.state_dict = initial_state_logic(self, self.state_dict).copy()

        # internal/environmental parameters
        self.training = training
        self.window_size = 700  # The size of the PyGame window
        self.num_actions = 4  # this can be changed for dev purposes
        self.most_recent_actions = np.empty((self.num_actions), np.dtype('U100'))

        self.action_labels = [  # this is used mostly for pygame rendering
            "r0 ccw",
            "r0 cw",
            "r0 engage",
            "r0 wait",
            "r1 ccw",
            "r1 cw",
            "r1 engage",
            "r1 wait",
            "r2 ccw",
            "r2 cw",
            "r2 engage",
            "r2 wait",
        ]

        # self.observation_space =  #TensorSpace(shape=(self.num_robots * 2 + self.num_goals * 5,), dtype=np.float32)

        # Define discrete ranges (adjust bounds as needed)
        obs_space = {}
        obs_space["my location"] = spaces.Discrete(self.size)
        # obs_space["current goal difficulty"] = spaces.Box(low=-0.0, high=1.0, shape=(1,), dtype=np.float64)
        for i in range(1, self.num_robots):
            obs_space[f"teammate{i} location"] = spaces.Discrete(self.size)

        for i in range(self.num_goals):
            obs_space[f"goal{i} active"] = spaces.Discrete(2)
            obs_space[f"goal{i} checked"] = spaces.Discrete(2)
            
        self.observation_space = spaces.Dict(obs_space)

        # actions that the robots can carry out
        # move clockwise/anticlockwise, engage
        self.action_space = spaces.Discrete(self.num_actions)

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None
        self.pygame_clock = None
        self.elapsed_ticks = 0
        self.reset()

    def state_dict_to_observable(self, state_dict, robot_no):

        obs_dict = {}
        rob_loc = int(state_dict[f"robot{robot_no} location"])
        obs_dict["my location"] = rob_loc

        teammate_num = 1
        for i in range(1, self.num_robots):
            next_robot_index = (robot_no + i) % self.num_robots
            index = next_robot_index
            obs_dict[f"teammate{teammate_num} location"] = int(state_dict[f"robot{index} location"])
            teammate_num += 1

        for i in range(self.num_goals):
            obs_dict[f"goal{i} active"] = int(state_dict[f"goal{i} active"])
            obs_dict[f"goal{i} checked"] = int(state_dict[f"goal{i} checked"])

        return obs_dict
    
    def state_vector_to_dict(self, state_vector):
        state_dict = {}
        state_dict["clock"] = int(state_vector[0])
        for i in range(self.num_robots):
            state_dict[f"robot{i} location"] = int(state_vector[i +1])

        for i in range(self.num_goals):
            state_dict[f"goal{i} location"] = int(state_vector[self.num_robots + i])
            state_dict[f"goal{i} active"] = int(state_vector[[self.num_robots + i + 1]])
            state_dict[f"goal{i} checked"] = int(state_vector[[self.num_robots + i + 2]])
            state_dict[f"goal{i} discovery probability"] = float(state_vector[[self.num_robots + i + 3 ]])
            state_dict[f"goal{i} completion probability"] = float(state_vector[[self.num_robots + i + 4]])

        return state_dict

    def get_obs(self):
        return self.state_dict_to_observable(self.state_dict, self.state_dict["clock"])

    def get_info(self):
        info = {}
        info["elapsed steps"] = self.elapsed_steps
        info["elapsed ticks"] = self.elapsed_ticks
        return info

    def reset(self, random=False, seed=None):
        super().reset(seed=seed)
        init_copy = self.initial_state_dict.copy()

        if(random):
            for i in range(self.num_robots):
                init_copy[f"robot{i} location"] = np.random.randint(0, self.size)

        if self.initial_state_logic:
            init_copy = self.initial_state_logic(self, init_copy.copy())

        self.state_dict = init_copy

        self.elapsed_steps = 0
        self.elapsed_ticks = 0

        info = self.get_info()

        if self.render:
            self.render_frame(self.state_dict, info)

        obs = self.get_obs()
        # print(obs.keys())

        return obs, info

    def pseudoreward_function(self, state_tensor):
        raise NotImplementedError("No pseudoreward function has been supplied.")

    def transition_model(self, state, action_no):
        raise NotImplementedError("The transistion model of this environment is not defined.")

    def reward_model(self, old_state, action, new_state):
        raise NotImplementedError("The rewards model of this environment is not defined.")

    def step(self, action):

        self.elapsed_steps += 1
        old_state_dict = self.state_dict.copy()
        p_array, s_array = self.transition_model(self, old_state_dict, old_state_dict["clock"], action)  # probabilities and states

        # roll dice to detemine resultant state from possibilities
        roll = np.random.random()
        t = 0
        chosen_state = -1
        for i in range(len(p_array)):
            t += p_array[i]
            if (roll < t):
                chosen_state = i
                break

        if (chosen_state < 0):
            print(action, p_array, s_array)
            raise ValueError("Something has gone wrong with choosing the state")

        # get the reward for this transition based on the reward model
        reward = self.reward_model(self, old_state_dict, old_state_dict["clock"], action, s_array[chosen_state])

        # assume the new state
        self.state_dict = s_array[chosen_state]
        if (self.render):
            self.render_frame(self.state_dict, True)

        # set terminated (all goals checked and inactive)
        terminated = True
        for i in range(self.num_goals):
            if (self.state_dict[f"goal{i} active"] == 1 or self.state_dict[f"goal{i} checked"] == 0):
                terminated = False
                break

        info = self.get_info()

        return self.get_obs(), reward, terminated, False, info

    def render_frame(self, state, inEnv=False):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size * 1.3, self.window_size)
            )
        if self.pygame_clock is None:
            self.pygame_clock = pygame.time.Clock()

        font = pygame.font.SysFont('notosans', 25)
        canvas = pygame.Surface((self.window_size * 1.3, self.window_size))

        tokamak_centre = ((self.window_size / 2), self.window_size / 2)

        canvas.fill((255, 255, 255))

        # draw all positions
        angle = - 2 * np.pi / self.size
        tokamak_r = self.window_size / 2 - 40
        for i in range(self.size):
            pygame.draw.circle(canvas, (144, 144, 144), (tokamak_centre[0], tokamak_centre[1]), tokamak_r, width=1)
            # offset the angle so that other objects are displayed between lines rather than on top
            xpos = tokamak_centre[0] + tokamak_r * np.cos((i - 0.5) * angle - np.pi)
            ypos = tokamak_centre[1] + tokamak_r * np.sin((i - 0.5) * angle - np.pi)
            pygame.draw.line(canvas,
                             (144, 144, 144),
                             (tokamak_centre[0], tokamak_centre[1]),
                             (xpos, ypos))

            text = font.render(str(i), True, (0, 0, 0))
            text_width = text.get_rect().width
            text_height = text.get_rect().height
            xpos = tokamak_centre[0] + (tokamak_r * 1.05) * np.cos((i) * angle - np.pi)
            ypos = tokamak_centre[1] + (tokamak_r * 1.05) * np.sin((i) * angle - np.pi)
            canvas.blit(text, (xpos - text_width / 2, ypos - text_height / 2))

        # draw the goals
        for i in range(self.num_goals):
            pos = state[f"goal{i} location"]
            xpos = tokamak_centre[0] + tokamak_r * 3 / 4 * np.cos(angle * pos - np.pi)
            ypos = tokamak_centre[1] + tokamak_r * 3 / 4 * np.sin(angle * pos - np.pi)
            if (state[f"goal{i} checked"] == 0):
                text = font.render("{:.2f}".format(state[f"goal{i} discovery probability"]), True, (255, 255, 255))
                circ_colour = (200 * state[f"goal{i} discovery probability"], 200 * (1 - state[f"goal{i} discovery probability"]), 0)
            elif (state[f"goal{i} active"] == 1):
                text = font.render("{:.2f}".format(state[f"goal{i} completion probability"]), True, (255, 255, 255))
                circ_colour = (0, 200, 0)
            elif (state[f"goal{i} active"] == 0):
                text = font.render("{:.2f}".format(state[f"goal{i} completion probability"]), True, (255, 255, 255))
                circ_colour = (200, 200, 200)

            circ = pygame.draw.circle(canvas, circ_colour, (xpos, ypos), 30)  # maybe make these rects again

            text_width = text.get_rect().width
            text_height = text.get_rect().height
            canvas.blit(source=text, dest=(circ.centerx - text_width / 2, circ.centery - text_height / 2))

        # draw robots
        for i in range(self.num_robots):
            pos = state[f"robot{i} location"]
            if (self.state_dict["clock"] == i):
                colour = (0, 200, 200)
            else:
                colour = (0, 0, 255)
            second_robot_present = False
            for j in range(self.num_robots):
                if j <= i:
                    continue
                else:
                    if state[f"robot{j} location"] == pos:
                        second_robot_present = True
            xpos = tokamak_centre[0] + (tokamak_r / (2 if not second_robot_present else 3)) * np.cos(angle * pos - np.pi)
            ypos = tokamak_centre[1] + (tokamak_r / (2 if not second_robot_present else 3)) * np.sin(angle * pos - np.pi)
            circ = pygame.draw.circle(canvas, colour, (xpos, ypos), 20)
            # if (state[f"robot{i} clock"]):
            #     text = font.render(str(i) + "'", True, (255, 255, 255))
            # else:
            text = font.render(str(i), True, (255, 255, 255))

            text_width = text.get_rect().width
            text_height = text.get_rect().height
            canvas.blit(source=text, dest=(circ.centerx - text_width / 2, circ.centery - text_height / 2))

        circ = pygame.draw.circle(canvas, (255, 255, 255), tokamak_centre, 60)
        circ = pygame.draw.circle(canvas, (144, 144, 144), tokamak_centre, 60, width=1)

        if not inEnv:
            rect = pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect((self.window_size, 40), (40, 40)))
            canvas.blit(font.render("(Trace replay)", True, (0, 0, 0)), rect)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.pygame_clock.tick(self.metadata["render_fps"])

        pygame.image.save(self.window, f"./animation/{self.frame_counter}.png")
        self.frame_counter += 1

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
