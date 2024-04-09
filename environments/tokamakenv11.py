#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:34:49 2023

@author: brendandevlin-hill
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class TokamakEnv11(gym.Env):
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
                 training=True,
                 render=False):

        # operational parameters
        state = {}

        self.transition_model = transition_model
        self.reward_model = reward_model
        self.blocked_model = blocked_model
        self.runstates = []
        self.statetree = []
        self.render = render

        for i in range(len(system_parameters.robot_locations)):
            state[f"robot{i} location"] = system_parameters.robot_locations[i]
            state[f"robot{i} clock"] = 0
        for i in range(len(system_parameters.goal_locations)):
            state[f"goal{i} location"] = system_parameters.goal_locations[i]
            state[f"goal{i} probability"] = system_parameters.goal_probabilities[i]
            state[f"goal{i} active"] = system_parameters.goal_activations[i]

        # state["elapsed ticks"] = system_parameters.elapsed_ticks

        self.state = state.copy()
        self.initial_state = state.copy()

        # non-operational (static/inherited) parameters
        self.size = system_parameters.size
        self.elapsed_steps = 0
        self.start_locations = np.array(system_parameters.robot_locations.copy())
        self.num_goals = len(system_parameters.goal_locations)
        self.num_robots = len(system_parameters.robot_locations)

        # Transition = namedtuple('Transition', (f"robot{self.num_robots}clock"))
        # a = Transition("test")
        # print(a)

        # internal/environmental parameters
        self.training = training
        self.window_size = 700  # The size of the PyGame window
        self.num_actions = 3  # this can be changed for dev purposes
        self.most_recent_actions = np.empty((3), np.dtype('U100'))
        # self.render_mode = render_mode

        # observations actually contain less info than states now
        obDict = {}
        for i in range(0,self.num_robots):
            obDict[f"robot{i} location"] = spaces.Discrete(self.size)
            obDict[f"robot{i} clock"] = spaces.Discrete(2)  # true = cannot move
        for i in range(0,self.num_goals):
            obDict["active goals"] = spaces.Discrete(len(system_parameters.goal_locations))
        # obDict["elapsed ticks"] = spaces.Discrete(100)

        self.action_labels = [
            "r0 ccw",
            "r0 cw",
            "r0 engage",
            "r1 ccw",
            "r1 cw",
            "r1 engage",
            "r2 ccw",
            "r2 cw",
            "r2 engage"
        ]

        self.observation_space = spaces.Dict(obDict)

        # actions that the robots can carry out
        # move clockwise/anticlockwise, engage
        self.action_space = spaces.Discrete(self.num_robots * self.num_actions)

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None
        self.clock = None
        self.reset()

    def get_parameters(self):
        # parameters = {
        #     "size": self.size,
        #     "num_active" : self.num_robots,
        #     "robot_locations" : self.robot_locations,
        #     "goal_locations" : self.goal_locations,
        #     "goal_probabilities" : self.goal_probabilities,
        #     "goal_activations" : self.goal_activations,
        #     # "elapsed" : self.elapsed
        # }
        # return parameters

        raise NotImplementedError()

    def get_obs(self):
        obs = {}
        for i in range(0,self.num_robots):
            obs[f"robot{i} location"] = self.state[f"robot{i} location"]
            obs[f"robot{i} clock"] = self.state[f"robot{i} clock"]
        for i in range(0,self.num_goals):
            obs["active goals"] = np.sum([self.state[f"goal{i} active"] for i in range(0, self.num_goals)])
        return obs

    def get_info(self):
        info = {}
        info["elapsed ticks"] = self.elapsed_ticks  # CURRENTLY ALWAYS 0!
        info["elapsed steps"] = self.elapsed_steps
        info["pseudoreward"] = self.pseudoreward_function()
        return info

    def parse_robot_locations(self):

        robot_locations = self.start_locations[np.argwhere(self.robot_status)].flatten()
        broken_robot_locations = self.start_locations[np.argwhere(self.robot_status - 1)].flatten()

        return robot_locations, broken_robot_locations

    # def query_state_action_pair(self, state, action):

    def reset(self, seed=None, options=None):
        # print("reset")
        super().reset(seed=seed)
        if(options):
            if(options["type"] == "state"):
                # assert options["state"]
                # print(options["state"])
                self.state = options["state"].copy()
        else:
            self.state = self.initial_state.copy()
        self.elapsed_steps = 0
        self.elapsed_ticks = 0

        info = self.get_info()

        if self.render:
            self.render_frame(self.state, info)

        return self.get_obs(), info

    def pseudoreward_function(self):
        tot = 0
        # goal positions
        for i in range(self.num_robots):
            rob_pos = self.state[f"robot{i} location"]
            min_mod_dist = self.size * 2
            for j in range(self.num_goals):
                if self.state[f"goal{j} active"] == 0:  # this goal is already completed
                    pass
                else:
                    goal_pos = self.state[f"goal{j} location"]
                    #calculate average distance of robot from each goal:
                    naive_dist = abs(rob_pos - goal_pos)  # weight by the probability that it is actually there
                    mod_dist = min(naive_dist, self.size - naive_dist)  # to account for cyclical space
                    min_mod_dist = min(mod_dist, min_mod_dist)
            tot += min_mod_dist
        return -tot  # -ve sign so that it should be minimised

    def transition_model(self, state, action_no):
        raise NotImplementedError("The transistion model of this environment is not defined.")

    def reward_model(self, old_state, action, new_state):
        raise NotImplementedError("The rewards model of this environment is not defined.")

    def step(self, action):

        self.elapsed_steps += 1

        old_state = self.state.copy()

        p_array, s_array = self.transition_model(self, old_state, action)
        #implement r_array/reward

        # roll dice to detemine resultant state from possibilities
        roll = np.random.random()
        t = 0
        chosen_state = -1
        for i in range(len(p_array)):
            t += p_array[i]
            # print("probs:", t, p_array)
            if (roll < t):
                chosen_state = i
                break
        if(chosen_state < 0):
            raise ValueError("Something has gone wrong with choosing the state")

        # get the reward for this transition based on the reward model
        reward = self.reward_model(self, old_state, action, s_array[chosen_state])

        # assume the new state
        self.state = s_array[chosen_state]

        # set terminated (all goals checked and not instantiated)
        terminated = True
        for i in range(self.num_goals):
            if (self.state[f"goal{i} active"] == 1):
                terminated = False
                break

        if(s_array[chosen_state] not in self.runstates):
            self.statetree.append(s_array[chosen_state])

        if(terminated):
            for state in self.runstates:
                self.statetree.append(state)

        info = self.get_info()

        if self.render:
            # checking if the robot performed a wait action
            robot_no = int(np.floor(action / self.num_actions))
            if(self.blocked_model(self, old_state)[action]):
                self.most_recent_actions[robot_no] = "wait"
            else:
                self.most_recent_actions[robot_no] = self.action_labels[action]
            self.render_frame(self.state, info)

        obs = self.get_obs()

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self, state, info):
        # note: the -np.pi is to keep the segments consistent with the jorek interpreter

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
        angle = 2 * np.pi / self.size
        tokamak_r = self.window_size / 2 - 40
        for i in range(self.size):
            pygame.draw.circle(canvas, (144,144,144), (tokamak_centre[0], tokamak_centre[1]),tokamak_r, width=1)
            # offset the angle so that other objects are displayed between lines rather than on top
            xpos = tokamak_centre[0] + tokamak_r * np.cos((i - 0.5) * angle - np.pi)
            ypos = tokamak_centre[1] + tokamak_r * np.sin((i - 0.5) * angle - np.pi)
            pygame.draw.line(canvas,
                             (144,144,144),
                             (tokamak_centre[0], tokamak_centre[1]),
                             (xpos, ypos))

            text = font.render(str(i), True, (0,0,0))
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
            if(state[f"goal{i} active"] == 1):
                text = font.render(str(state[f"goal{i} probability"]), True, (255,255,255))
                colour = (0,200,0)
            elif(state[f"goal{i} active"] == 0):
                text = font.render(str(state[f"goal{i} probability"]), True, (255,255,255))
                colour = (100,100,100)

            circ = pygame.draw.circle(canvas, colour, (xpos, ypos), 30)  # maybe make these rects again

            text_width = text.get_rect().width
            text_height = text.get_rect().height
            canvas.blit(source=text, dest=(circ.centerx - text_width / 2, circ.centery - text_height / 2))

        # draw robots
        for i in range(self.num_robots):
            pos = state[f"robot{i} location"]
            xpos = tokamak_centre[0] + tokamak_r / 2 * np.cos(angle * pos - np.pi)
            ypos = tokamak_centre[1] + tokamak_r / 2 * np.sin(angle * pos - np.pi)
            circ = pygame.draw.circle(canvas, (0,0,255), (xpos, ypos), 20)
            if(state[f"robot{i} clock"]):
                text = font.render(str(i) + "'", True, (255,255,255))
            else:
                text = font.render(str(i), True, (255,255,255))

            text_width = text.get_rect().width
            text_height = text.get_rect().height
            canvas.blit(source=text, dest=(circ.centerx - text_width / 2, circ.centery - text_height / 2))

        circ = pygame.draw.circle(canvas, (255,255,255), tokamak_centre, 60)
        circ = pygame.draw.circle(canvas, (144,144,144), tokamak_centre, 60, width=1)

        # print(f"""
        #       epoch: {state["elapsed"]}
        #       robot locations : {self.robot_locations}
        #       actions: {self.most_recent_actions}
        #       blocked actions: {self.get_blocked_actions()}
        #       goal checked : {self.goal_checked}
        #       goal instantiations: {self.goal_instantiations}
        #       """)

        # print(f"""
        #       epoch: {info["elapsed steps"]}
        #       """)

        # # draw tick number
        # rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,0), (40, 40)))
        # canvas.blit(font.render("t=" + str(state["elapsed ticks"]), True, (0,0,0)), rect)
        # most recent actions
        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,40), (40, 40)))
        canvas.blit(font.render("r0: " + str(self.most_recent_actions[0]), True, (0,0,0)), rect)

        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,80), (40, 40)))
        canvas.blit(font.render("r1: " + str(self.most_recent_actions[1]), True, (0,0,0)), rect)

        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,120), (40, 40)))
        canvas.blit(font.render("r2: " + str(self.most_recent_actions[2]), True, (0,0,0)), rect)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
