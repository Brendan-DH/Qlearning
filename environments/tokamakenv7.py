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


class TokamakEnv7(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # def set_parameters(size, num_active, num_goals, goal_locations):
    #     return None

    def __init__(self,
                 system_parameters,
                 training=True,
                 render_mode=None):

        # operational parameters
        self.size = system_parameters.size
        self.robot_status = np.array(system_parameters.robot_status)
        self.elapsed = system_parameters.elapsed
        self.port_locations = system_parameters.port_locations
        self.breakage_probability = system_parameters.breakage_probability
        self.goal_locations = system_parameters.goal_locations
        self.goal_checked = system_parameters.goal_checked
        self.goal_probabilities = system_parameters.goal_probabilities
        self.goal_instantiations = system_parameters.goal_instantiations
        self.num_goals = len(self.goal_locations)
        self.start_locations = np.array(system_parameters.robot_locations.copy())
        self.active_robot_locations = system_parameters.robot_locations.copy()
        self.num_active = len(self.active_robot_locations)

        # non-operational (static/inherited) parameters
        self.goal_resolutions = system_parameters.goal_resolutions

        # internal/environmental parameters
        self.training = training
        self.window_size = 700  # The size of the PyGame window
        self.num_actions = 3  # this can be changed for dev purposes
        self.most_recent_actions = np.empty((3), np.dtype('U100'))
        self.render_mode = render_mode

        obDict = {}

        # positions of the robots
        for i in range(0,self.num_active):
            obDict[f"robot{i} location"] = spaces.Discrete(self.size)
            obDict[f"robot{i} clock"] = spaces.Discrete(2)  # true = cannot move

        # positions of the goals and whether they are complete
        for i in range(0,self.num_goals):
            obDict[f"goal{i} location"] = spaces.Discrete(self.size)
            obDict[f"goal{i} probability"] = spaces.Box(low=0, high=1, shape=[1])  # continuous variable in [0,1]. sample() returns array though.
            obDict[f"goal{i} instantiated"] = spaces.Discrete(2)
            obDict[f"goal{i} checked"] = spaces.Discrete(2)  # records if a goal has been visited yet

        obDict["elapsed_ticks"] = spaces.Discrete(500)

        self.observation_space = spaces.Dict(obDict)

        # actions that the robots can carry out
        # move clockwise/anticlockwise, engage
        self.action_space = spaces.Discrete(self.num_active * self.num_actions)

        # may need an array here for mapping abstract action to
        # function

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None
        self.clock = None
        self.elapsed = 0
        self.reset()

    def get_parameters(self):
        parameters = {
            "size": self.size,
            "num_active" : self.num_active,
            "robot_locations" : self.active_robot_locations,
            "goal_locations" : self.goal_locations,
            "goal_probabilities" : self.goal_probabilities,
            "goal_instantiations" : self.goal_instantiations,
            "goal_checked" : self.goal_checked,
            "elapsed" : self.elapsed
        }

        return parameters

    def get_obs(self):
        obs = {}
        for i in range(0,self.num_active):
            obs[f"robot{i} location"] = self.active_robot_locations[i]
            obs[f"robot{i} clock"] = self.robot_clocks[i]  # true = cannot move
        for i in range(0,self.num_goals):
            obs[f"goal{i} location"] = self.goal_locations[i]
            obs[f"goal{i} probability"] = self.goal_probabilities[i]
            obs[f"goal{i} instantiated"] = self.goal_instantiations[i]
            obs[f"goal{i} checked"] = self.goal_checked[i]
        obs["elapsed_ticks"] = self.elapsed

        return obs

    def get_info(self):
        info = {}
        info["elapsed_ticks"] = self.elapsed
        info["av_dist"] = self.av_dist()
        info["goal_resolutions"] = self.goal_resolutions.copy()
        info["robot_status"] = self.robot_status.copy()
        return info

    def parse_robot_locations(self):

        active_robot_locations = self.start_locations[np.argwhere(self.robot_status)].flatten()
        broken_robot_locations = self.start_locations[np.argwhere(self.robot_status - 1)].flatten()

        return active_robot_locations, broken_robot_locations

    def reset(self, seed=None, options=None):
        # print("reset")
        super().reset(seed=seed)
        self.active_robot_locations = self.start_locations.copy()

        # reset goals
        self.goal_probabilities = self.goal_probabilities.copy()
        self.goal_checked = [0 for i in range(self.num_goals)]
        self.clock_tick()
        self.elapsed = 0  # must be set to 0 as clock_tick increments by 1

        self.goal_resolutions = np.zeros_like(self.goal_probabilities)
        self.goal_instantiated = np.zeros_like(self.goal_probabilities)

        if self.render_mode == "human":
            self.render_frame()

        return self.get_obs(), self.get_info()

    def av_dist(self):
        tot_av = 0
        # goal positions
        for i in range(len(self.active_robot_locations)):
            rob_pos = self.active_robot_locations[i]
            rob_av = 0
            num_active_goals = len(self.goal_locations)
            # num_active_goals = np.sum(np.array(self.goal_probabilities) > 0) # status is True if complete; this calculates num False
            for j in range(len(self.goal_locations)):
                if self.goal_probabilities[j] == 0:  # this goal is already completed
                    continue
                goal_pos = self.goal_locations[j]
                #calculate average distance of robot from each goal:
                dist = abs(rob_pos - goal_pos) * self.goal_probabilities[j]  # weight by the probability that it is actually there
                mod_dist = min((dist, self.size - dist))  # to account for cyclical space
                rob_av += mod_dist / num_active_goals
            # average of average distances
            tot_av += rob_av / len(self.active_robot_locations)
        return tot_av

    def get_counter_cw_blocked(self, robot_no):

        moving_robot_loc = self.active_robot_locations[robot_no]

        for j in range(self.num_active):
            other_robot_loc = self.active_robot_locations[j]
            if(robot_no == j):  # don't need to check robots against themselves
                continue
            if (moving_robot_loc == other_robot_loc):
                raise ValueError(f"Two robots occupy the same location (r{robot_no} & r{j} @ {moving_robot_loc}).")
            if(other_robot_loc == (moving_robot_loc + 1) % self.size):
                return True
        return False

    def get_cw_blocked(self, robot_no):

        moving_robot_loc = self.active_robot_locations[robot_no]

        for j in range(self.num_active):
            other_robot_loc = self.active_robot_locations[j]
            if(robot_no == j):  # don't need to check robots against themselves
                continue
            if (moving_robot_loc == other_robot_loc):
                raise ValueError(f"Two robots occupy the same location (r{robot_no} & r{j} @ {moving_robot_loc}).")
            if(other_robot_loc == (self.size - 1 if moving_robot_loc - 1 < 0 else moving_robot_loc - 1)):
                return True

        return False

    def get_blocked_actions(self):

        # observation  = self.get_obs()
        blocked_actions = np.zeros(self.action_space.n)
        # go per robot
        # actions are left, right, inspect
        # lets say for now that robots cannot occupy the same tile
        for i in range(self.num_active):
            moving_robot_loc = self.active_robot_locations[i]
            if(self.robot_clocks[i]):  # has robot's clock ticked?
                blocked_actions[i * self.num_actions:(i * self.num_actions) + self.num_actions] = 1  # block all actions for this robot
            else:
                for j in range(self.num_active):
                    other_robot_loc = self.active_robot_locations[j]

                    if(i == j):  # don't need to check robots against themselves
                        continue

                    if (moving_robot_loc == other_robot_loc):
                        raise ValueError(f"Two robots occupy the same location (r{i} & r{j} @ {moving_robot_loc}).")

                blocked_actions[(i * self.num_actions)] = self.get_counter_cw_blocked(i)
                blocked_actions[(i * self.num_actions) + 1] = self.get_cw_blocked(i)

                #block inspection if robot is not over known task location:
                block_inspection = 1
                for k in range(len(self.goal_locations)):
                    if (self.goal_locations[k] == moving_robot_loc and self.goal_instantiations[k] == 1):
                        block_inspection = 0  # unblock this engage action
                blocked_actions[(i * self.num_actions) + 2] = block_inspection

        # print(self.active_robot_locations, blocked_actions)
        return blocked_actions

    def clock_tick(self):
        self.robot_clocks = [0 for i in range(self.num_active)]  # set all clocks to false
        self.elapsed += 1

    def step(self, action):

        # determine blocked actions based on the current state
        # blocked actions give a negative reward and don't progress the system
        rel_action = action % self.num_actions  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait
        # by which robot:
        robot_no = int(np.floor(action / self.num_actions))

        blocked_actions = self.get_blocked_actions()
        if(blocked_actions[action]):

            reward = -1
            current_action = "wait"

        else:
            # which action is being taken:

            current_location = self.active_robot_locations[robot_no]
            current_action = ""
            reward = 0

            if(np.sum(self.robot_clocks) == 0):
                self.most_recent_actions = np.empty((3), np.dtype('U100'))

            if(rel_action == 0):  # counter-clockwise movement
                if (current_location < self.size - 1):
                    self.active_robot_locations[robot_no] = current_location + 1
                if (current_location == self.size - 1):  # cycle round
                    self.active_robot_locations[robot_no] = 0
                # reward -= 0.5
                current_action = "move ccw"

            if(rel_action == 1):  # clockwise movement
                if (current_location > 0):
                    self.active_robot_locations[robot_no] = current_location - 1
                if (current_location == 0):  # cycle round
                    self.active_robot_locations[robot_no] = self.size - 1
                # reward -= 0.5
                current_action = "move cw"

            if (rel_action == 2):  # engage robot, complete task
                for i in range(len(self.goal_locations)):  # iterate over locations and mark appropriate goals as done
                    if(self.goal_locations[i] == current_location and self.goal_instantiations[i] == 1):
                        self.goal_instantiations[i] = 0
                        # reward if robots manage to complete a task. Lessens over time.
                        reward += 1000 - ((self.elapsed * self.num_active) / 100) * 500
                current_action = "engage"

            for i in range(len(self.goal_locations)):  # iterate over locations and mark appropriate goals as done
                # print(i, self.goal_locations[i], self.goal_checked, self.active_robot_locations)
                if(self.goal_locations[i] == self.active_robot_locations[robot_no] and self.goal_checked[i] == 0):
                    self.goal_checked[i] = 1  # note that the goal has been visited and checked
                    # resolve the non-determinism:
                    if np.random.rand() < self.goal_probabilities[i]:
                        self.goal_resolutions[i] = 1  # the goal exists
                        self.goal_instantiations[i] = 1
                    else:
                        self.goal_resolutions[i] = 0  # the goal does not
                        self.goal_instantiations[i] = 0

                    reward += 100  # reward robots for discovering tasks

        self.robot_clocks[robot_no] = 1  # lock robot until clock ticks

        # robot breaks, this should alert the overseer module
        # this shouldn't happen in training mode
        if(not self.training):
            if(np.random.random() < self.breakage_probability):
                self.robot_status[robot_no] = 0
                print(f"robot {robot_no} broken")

        if np.sum(self.robot_clocks) == self.num_active:  # check if a tick should happen
            self.clock_tick()

        self.most_recent_actions[robot_no] = current_action

        if(all(self.goal_checked) and not any(self.goal_instantiations)):
            terminated = True
        else:
            terminated = False

        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == "human":
            self.render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        # note: the -np.pi is to keep the segments consistent with the jorek interpreter

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size * 1.3, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
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
            pos = self.goal_locations[i]
            xpos = tokamak_centre[0] + tokamak_r * 3 / 4 * np.cos(angle * pos - np.pi)
            ypos = tokamak_centre[1] + tokamak_r * 3 / 4 * np.sin(angle * pos - np.pi)
            if(self.goal_checked[i] == 0):
                text = font.render(f"{self.goal_probabilities[i]}?", True, (255,255,255))
                colour = (255,0,0)
            elif(self.goal_checked[i] == 1):
                if(self.goal_resolutions[i] == 0):
                    text = font.render("0", True, (255,255,255))
                    colour = (200,200,200)
                elif(self.goal_instantiations[i] == 1):
                    text = font.render("1", True, (255,255,255))
                    colour = (0,200,0)
                elif(self.goal_instantiations[i] == 0):
                    text = font.render("done", True, (255,255,255))
                    colour = (0,200,0)

            circ = pygame.draw.circle(canvas, colour, (xpos, ypos), 30)  # maybe make these rects again

            text_width = text.get_rect().width
            text_height = text.get_rect().height
            canvas.blit(source=text, dest=(circ.centerx - text_width / 2, circ.centery - text_height / 2))

        # draw robots
        for i in range(self.num_active):
            pos = self.active_robot_locations[i]
            xpos = tokamak_centre[0] + tokamak_r / 2 * np.cos(angle * pos - np.pi)
            ypos = tokamak_centre[1] + tokamak_r / 2 * np.sin(angle * pos - np.pi)
            circ = pygame.draw.circle(canvas, (0,0,255), (xpos, ypos), 20)
            if(self.robot_clocks[i]):
                text = font.render(str(i) + "'", True, (255,255,255))
            else:
                text = font.render(str(i), True, (255,255,255))

            text_width = text.get_rect().width
            text_height = text.get_rect().height
            canvas.blit(source=text, dest=(circ.centerx - text_width / 2, circ.centery - text_height / 2))

        circ = pygame.draw.circle(canvas, (255,255,255), tokamak_centre, 60)
        circ = pygame.draw.circle(canvas, (144,144,144), tokamak_centre, 60, width=1)

        print(f"""
              epoch: {self.elapsed}-{np.sum(self.robot_clocks)}
              robot locations : {self.active_robot_locations}
              actions: {self.most_recent_actions}
              blocked actions: {self.get_blocked_actions()}
              goal checked : {self.goal_checked}
              goal instantiations: {self.goal_instantiations}
              """)

        # draw tick number
        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,0), (40, 40)))
        canvas.blit(font.render("t=" + str(self.elapsed), True, (0,0,0)), rect)
        # most recent actions
        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,40), (40, 40)))
        canvas.blit(font.render("r0: " + str(self.most_recent_actions[0]), True, (0,0,0)), rect)

        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,80), (40, 40)))
        canvas.blit(font.render("r1: " + str(self.most_recent_actions[1]), True, (0,0,0)), rect)

        rect = pygame.draw.rect(canvas,(255,255,255), pygame.Rect((self.window_size,120), (40, 40)))
        canvas.blit(font.render("r2: " + str(self.most_recent_actions[2]), True, (0,0,0)), rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
