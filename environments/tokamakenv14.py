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


class TensorSpace(gym.Space):
    def __init__(self, shape, dtype=torch.float32):
        """
        Custom observation space for PyTorch tensors.

        Args:
            shape (tuple): Shape of the tensor.
            dtype (torch.dtype): Data type of the tensor.
        """
        # Initialize the parent class with shape and dtype
        super().__init__(shape, dtype)

        # Store the shape and dtype as instance attributes
        self._shape = shape
        self._dtype = dtype

    def sample(self):
        """Generate a random tensor."""
        return torch.rand(self._shape, dtype=self._dtype)

    def contains(self, x):
        """Check if `x` is a valid tensor in this space."""
        return isinstance(x, torch.Tensor) and x.shape == self._shape and x.dtype == self._dtype

    def __repr__(self):
        return f"TensorSpace(shape={self._shape}, dtype={self._dtype})"


class TokamakEnv14(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    # def set_parameters(size, num_active, num_goals, goal_locations):
    #     return None

    def set_rendering(self, rendering):
        self.render = rendering

    def __init__(self,
                 system_parameters,
                 transition_model,
                 blocked_model,
                 reward_model,
                 initial_state_logic=None,
                 training=True,
                 render=False,
                 render_ticks_only=True):

        # operational parameters

        state = {}

        self.transition_model = transition_model
        self.reward_model = reward_model
        self.blocked_model = blocked_model
        self.runstates = []
        self.statetree = []
        self.render = render
        self.render_ticks_only = render_ticks_only
        self.frame_counter = 0

        for i in range(len(system_parameters.robot_locations)):
            state[f"robot{i} location"] = system_parameters.robot_locations[i]
            state[f"robot{i} clock"] = 0
        for i in range(len(system_parameters.goal_locations)):
            state[f"goal{i} location"] = system_parameters.goal_locations[i]
            state[f"goal{i} active"] = system_parameters.goal_activations[i]  # 1 = goal should be engaged
            state[f"goal{i} checked"] = system_parameters.goal_checked[i]  # 1 = goal location has been visited
            state[f"goal{i} discovery probability"] = system_parameters.goal_discovery_probabilities[i]  # p that goal is present
            state[f"goal{i} completion probability"] = system_parameters.goal_completion_probabilities[i]  # p that goal is completed in 1 attempt

        # state["elapsed ticks"] = system_parameters.elapsed_ticks

        self.state = state.copy()
        self.initial_state = state.copy()

        self.size = system_parameters.size
        self.elapsed_steps = 0
        self.start_locations = np.array(system_parameters.robot_locations.copy())
        self.num_goals = len(system_parameters.goal_locations)
        self.num_robots = len(system_parameters.robot_locations)

        if initial_state_logic:
            print("Taking care of initial state transitions")
            state_tensor = initial_state_logic(self, self.construct_state_tensor_from_system_parameters(system_parameters, device=torch.device("cpu")))
            self.state_tensor = state_tensor
            self.initial_state_tensor = state_tensor.detach().clone()

        else:
            self.state_tensor = self.construct_state_tensor_from_system_parameters(system_parameters, device=torch.device("cpu"))
            self.initial_state_tensor = self.construct_state_tensor_from_system_parameters(system_parameters, device=torch.device("cpu"))

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

        self.observation_space = TensorSpace(shape=(self.num_robots * 2 + self.num_goals * 5,), dtype=np.float32)

        # Define discrete ranges (adjust bounds as needed)
        obs_space = {}
        for i in range(self.num_robots):
            obs_space[f"robot{i} location"] = spaces.Discrete(self.size)
            obs_space[f"robot{i} clock"] = spaces.Discrete(2)

        for i in range(self.num_goals):
            obs_space[f"goal{i} active"] = spaces.Discrete(2)
            obs_space[f"goal{i} checked"] = spaces.Discrete(2)

        self.observation_space = spaces.Dict(obs_space)

        # actions that the robots can carry out
        # move clockwise/anticlockwise, engage
        self.action_space = spaces.Discrete(self.num_robots * self.num_actions)

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None
        self.clock = None
        self.elapsed_ticks = 0
        self.reset()

    def state_tensor_to_observable(self, state_tensor):

        obs_dict = {}

        for i in range(self.num_robots):
            index = i * 2
            obs_dict[f"robot{i} location"] = int(state_tensor[index].item())
            obs_dict[f"robot{i} clock"] = int(state_tensor[index + 1].item())

        for i in range(self.num_goals):
            index = self.num_robots * 2 + i * 5
            obs_dict[f"goal{i} active"] = int(state_tensor[index + 1].item())
            obs_dict[f"goal{i} checked"] = int(state_tensor[index + 2].item())

        return obs_dict

    def interpret_state_tensor(self, state_tensor):

        # function to take the state tensor and translate it into a state dict

        state_dict = {}

        for i in range(self.num_robots):
            index = i * 2
            state_dict[f"robot{i} location"] = state_tensor[index].item()
            state_dict[f"robot{i} clock"] = state_tensor[index + 1].item()

        for i in range(self.num_goals):
            index = self.num_robots * 2 + i * 5
            state_dict[f"goal{i} location"] = state_tensor[index].item()
            state_dict[f"goal{i} active"] = state_tensor[index + 1].item()
            state_dict[f"goal{i} checked"] = state_tensor[index + 2].item()
            state_dict[f"goal{i} discovery probability"] = state_tensor[index + 3].item()
            state_dict[f"goal{i} completion probability"] = state_tensor[index + 4].item()

        return state_dict

    def construct_state_tensor_from_system_parameters(self, system_parameters, device=torch.device("cpu")):

        tensor_length = self.num_robots * 2 + self.num_goals * 5
        state_tensor = torch.empty((tensor_length), dtype=torch.float32, device=device, requires_grad=False)

        for i in range(len(system_parameters.robot_locations)):
            index = i * 2
            state_tensor[index] = system_parameters.robot_locations[i]
            state_tensor[index + 1] = 0
        for i in range(len(system_parameters.goal_locations)):
            index = self.num_robots * 2 + i * 5
            state_tensor[index] = system_parameters.goal_locations[i]
            state_tensor[index + 1] = system_parameters.goal_activations[i]  # 1 = goal should be engaged
            state_tensor[index + 2] = system_parameters.goal_checked[i]  # 1 = goal location has been visited
            state_tensor[index + 3] = system_parameters.goal_discovery_probabilities[i]  # p that goal is present
            state_tensor[index + 4] = system_parameters.goal_completion_probabilities[i]  # p that goal is completed in 1 attempt

        return state_tensor

    def construct_state_tensor_from_dict(self, state_dict, device=torch.device("cpu")):

        tensor_length = self.num_robots * 2 + self.num_goals * 5
        state_tensor = torch.empty((tensor_length), dtype=torch.float32, device=device, requires_grad=False)

        for i in range(self.num_robots):
            index = i * 2
            state_tensor[index] = state_dict[f"robot{i} location"]
            state_tensor[index + 1] = state_dict[f"robot{i} clock"]
        for i in range(self.num_goals):
            index = self.num_robots * 2 + i * 5
            state_tensor[index] = state_dict[f"goal{i} location"]
            state_tensor[index + 1] = state_dict[f"goal{i} active"]
            state_tensor[index + 2] = state_dict[f"goal{i} checked"]
            state_tensor[index + 3] = state_dict[f"goal{i} discovery probability"]
            state_tensor[index + 4] = state_dict[f"goal{i} completion probability"]

        return state_tensor

    def get_obs(self):

        obs_dict = {}

        for i in range(self.num_robots):
            index = i * 2
            obs_dict[f"robot{i} location"] = int(self.state_tensor[index].item())
            obs_dict[f"robot{i} clock"] = int(self.state_tensor[index + 1].item())

        for i in range(self.num_goals):
            index = self.num_robots * 2 + i * 5
            obs_dict[f"goal{i} active"] = int(self.state_tensor[index + 1].item())
            obs_dict[f"goal{i} checked"] = int(self.state_tensor[index + 2].item())

        return obs_dict

    def get_info(self):
        info = {}
        info["elapsed steps"] = self.elapsed_steps
        info["elapsed ticks"] = self.elapsed_ticks
        # info["pseudoreward"] = self.pseudoreward_function()
        return info

    # def query_state_action_pair(self, state, action):

    def reset(self, seed=None, options=None):
        # print("reset")
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        self.state_tensor = self.initial_state_tensor.detach().clone()

        self.elapsed_steps = 0
        self.elapsed_ticks = 0

        info = self.get_info()

        if self.render:
            self.render_frame(self.state, info)

        return self.get_obs(), info

    def pseudoreward_function(self, state_tensor):
        # defining a pseudoreward function that roughly describes the proximity to the `completed' state
        pr = self.size * self.num_robots * 2  # initialising to a high value
        print("init", pr)
        for i in range(self.num_robots):
            rob_position = state_tensor[i * 2].item()
            goal_min_mod_dist = self.size + 1  # store the mod distance to closest goal
            rob_min_mod_dist = self.size + 1  # store the mod distance to closest robot

            for j in range(self.num_robots):
                if i == j:
                    continue
                other_robot_pos = state_tensor[j * 2].item()
                naive_dist = abs(rob_position - other_robot_pos)  # non-mod distance
                rob_mod_dist = min(naive_dist, self.size - naive_dist)  # to account for cyclical space
                rob_min_mod_dist = min(rob_min_mod_dist, rob_mod_dist)  # update the smaller of the two

            pr += 0.2 * rob_min_mod_dist  # give a small bonus for being farther away from nearest robot
            print(f"robot {i} prox bonus", 0.2 * rob_min_mod_dist)

            for j in range(self.num_goals):
                goal_active = state_tensor[(self.num_robots * 2) + (j * 5) + 1].item()
                goal_checked = state_tensor[(self.num_robots * 2) + (j * 5) + 2].item()
                if (goal_active == 0 and goal_checked == 1):
                    pr += self.size + 2  # bonus for completing a goal; ensures PR always increases when goals completed
                    print(f"robot{i} goal {j} complete bonus", self.size + 2)
                else:
                    goal_position = state_tensor[(self.num_robots * 2) + (j * 5)].item()
                    # have to check here if there is another robot between the two - difficult
                    # for k in range(self.num_robots):
                    #     other_robot_pos = state_tensor[k * 2].item()
                    #     if other_robot_pos in ...
                    naive_dist = abs(rob_position - goal_position)  # non-mod distance
                    goal_mod_dist = min(naive_dist, self.size - naive_dist)  # to account for cyclical space
                    goal_min_mod_dist = min(goal_mod_dist, goal_min_mod_dist)  # update the smaller of the two

            print(f"robot {i} goal penalty", goal_min_mod_dist)
            pr -= goal_min_mod_dist  # subtract the distance 'penalty' from total possible reward

        print("final pr ", pr)

        return pr

    def transition_model(self, state, action_no):
        raise NotImplementedError("The transistion model of this environment is not defined.")

    def reward_model(self, old_state, action, new_state):
        raise NotImplementedError("The rewards model of this environment is not defined.")

    def step(self, action):

        self.elapsed_steps += 1

        old_state = self.state_tensor

        p_tensor, s_tensor = self.transition_model(self, old_state, action)  # probabilities and states

        # roll dice to detemine resultant state from possibilities
        roll = np.random.random()
        t = 0
        chosen_state = -1
        for i in range(len(p_tensor)):
            t += p_tensor[i]
            if (roll < t):
                chosen_state = i
                break

        if (chosen_state < 0):
            print(action, p_tensor, s_tensor)
            raise ValueError("Something has gone wrong with choosing the state")

        # get the reward for this transition based on the reward model
        reward = self.reward_model(self, old_state, action, s_tensor[chosen_state])

        # assume the new state
        self.state_tensor = s_tensor[chosen_state]

        # get the new observed state
        # observed_state_tensor = self.construct_state_tensor_from_dict(self.get_obs(), device=torch.device("cpu"))

        # set terminated (all goals checked and inactive)
        terminated = True
        goal_start_tensor_index = self.num_robots * 2
        for i in range(self.num_goals):
            if (self.state_tensor[goal_start_tensor_index + (i * 5) + 1].item() == 1 or self.state_tensor[goal_start_tensor_index + (i * 5) + 2].item() == 0):
                terminated = False
                break

        info = self.get_info()

        return self.get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self, state, inEnv=False):

        # note: the -np.pi is to keep the segments consistent with the jorek interpreter

        # print(self.blocked_model(self, state))

        if (self.render_ticks_only):
            for i in range(self.num_robots):
                if (state[f"robot{i} clock"] == 1):
                    return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size * 1.3, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

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
            second_robot_present = False
            for j in range(self.num_robots):
                if j <= i:
                    continue
                else:
                    if state[f"robot{j} location"] == pos:
                        second_robot_present = True
            xpos = tokamak_centre[0] + (tokamak_r / (2 if not second_robot_present else 3)) * np.cos(angle * pos - np.pi)
            ypos = tokamak_centre[1] + (tokamak_r / (2 if not second_robot_present else 3)) * np.sin(angle * pos - np.pi)
            circ = pygame.draw.circle(canvas, (0, 0, 255), (xpos, ypos), 20)
            if (state[f"robot{i} clock"]):
                text = font.render(str(i) + "'", True, (255, 255, 255))
            else:
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
        self.clock.tick(self.metadata["render_fps"])

        pygame.image.save(self.window, f"./animation/{self.frame_counter}.png")
        self.frame_counter += 1

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
