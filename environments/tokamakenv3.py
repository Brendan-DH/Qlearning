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


class TokamakEnv3(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def set_parameters(size, num_robots, num_goals, goal_locations):
        return None
    
    def __init__(self,
                 size=None,
                 num_robots=None,
                 num_goals=None,
                 goal_locations=None, 
                 render_mode=None):
        
        self.size = size
        self.num_robots = num_robots
        self.num_goals = num_goals
        self._goal_locations = goal_locations
        self.window_size = 512  # The size of the PyGame window
        self.elapsed = 0

        obDict = {}
        
        # positions of the robots
        for i in range(0,num_robots):
            obDict[f"robot{i} location"] = spaces.Discrete(size)
            obDict[f"robot{i} clock"] = spaces.Discrete(2) # true = cannot move
            
        # positions of the goals and whether they are complete
        for i in range(0,num_goals):
            obDict[f"goal{i} location"] = spaces.Discrete(size)
            obDict[f"goal{i} done"] = spaces.Discrete(2)

        self.observation_space = spaces.Dict(obDict)
        
        # actions that the robots can carry out
        # move clockwise/anticlockwise, engage
        self.action_space = spaces.Discrete(num_robots*3)
        
        # may need an array here for mapping abstract action to 
        # function
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.elapsed = 0
        self.reset()
        
    
    def _get_obs(self):
        obs = {}
        for i in range(0,self.num_robots):
            obs[f"robot{i} location"] = self._robot_locations[i]
            obs[f"robot{i} clock"] = self._robot_clocks[i] # true = cannot move
        for i in range(0,self.num_goals):
            obs[f"goal{i} location"] = self._goal_locations[i]
            obs[f"goal{i} done"] = self._goal_status[i]
            
        return obs
    
    def _get_info(self):
        info = {}
        info["elapsed"] = self.elapsed
        info["av_dist"] = self.av_dist()
        return info
        
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        if(options):
            self._robot_locations = options["robot_locations"].copy()
        else:
            # reset robot locations to random:
            self._robot_locations = [self.np_random.integers(0,self.size) 
                                     for i in range(self.num_robots)] 
            
        
        # reset goals
        self._goal_status = [False for i in range(self.num_goals)]
        self.elapsed = 0
        self._robot_clocks = [False for i in range(self.num_robots)] # set all clocks to false
        
        return self._get_obs(), self._get_info()
        
        self.elapsed = 0
        
        if self.render_mode == "human":
            self._render_frame()    
        
        return self._get_obs(), self._get_info()
        
    def av_dist(self):
        tot_av = 0
        # goal positions
        for i in range(len(self._robot_locations)):
            rob_pos = self._robot_locations[i]
            rob_av = 0
            num_active_goals = self.num_goals - np.sum(self._goal_status) # status is True if complete; this calculates num False
            for j in range(len(self._goal_locations)):
                if self._goal_status[j] == True: # this goal is already completed
                    continue
                goal_pos = self._goal_locations[j]
                #calculate average distance of robot from each goal:
                dist = abs(rob_pos - goal_pos)
                mod_dist = min((dist, self.size - dist)) # to account for cyclical space
                rob_av += mod_dist/num_active_goals
            # average of average distances
            tot_av += rob_av/len(self._robot_locations)
        # print("goal status:", self._goal_status, self._goal_locations, self._robot_locations, tot_av)
        return tot_av
                    
    def _get_blocked_actions(self):
        
        # observation  = self._get_obs()
        blocked_actions = np.zeros(self.action_space.n)
        
        # go per robot
        # actions are left, right, inspect
        # lets say for now that robots cannot occupy the same tile
        for i in range(self.num_robots):
            moving_robot_loc = self._robot_locations[i]
            if(self._robot_clocks[i]):
                blocked_actions[i*self.num_robots:(i*self.num_robots)+3] = 1 #block all actions for this robot
                
            for j in range(self.num_robots):
                other_robot_loc = self._robot_locations[j]
                
                if(i==j):
                    continue
                
                if (moving_robot_loc == other_robot_loc):
                    raise ValueError(f"Two robots occupy the same location (r{i} & r{j} @ {moving_robot_loc}).")
                
                # block counter-clockwise movement:
                if((other_robot_loc == moving_robot_loc - 1) or (moving_robot_loc == 0 and other_robot_loc == self.size-1)):
                    blocked_actions[i*self.num_robots] = 1
                    
                #block clockwise movement:
                if((other_robot_loc == moving_robot_loc + 1) or (moving_robot_loc == self.size-1 and other_robot_loc == 0)):
                    blocked_actions[(i*self.num_robots)+1] = 1
                    
        # print(self._robot_locations, blocked_actions)

        return blocked_actions
    
    def _clock_tick(self):
        self._robot_clocks = [False for i in range(self.num_robots)] # set all clocks to false
            
                
    def step(self, action):
    
        # determine blocked actions based on the current state
        # blocked actions give a negative reward and don't progress the system
        
        blocked_actions = self._get_blocked_actions()
        if(blocked_actions[action]):
            
            terminated = False
            reward = -1.0
                    
        else: 
            # which action is being taken:
            rel_action = action % 3 # 0=counter-clockwise, 1=clockwise, 2=engage
            # by which robot:
            robot_no = int(np.floor(action/self.num_robots))
            
            reward = 0.0
            
            current_location = self._robot_locations[robot_no]
            
            # simple cyclical motion; robots can move in either direction
            # and loop around the tokamak. They don't exclude or interfere with
            # each other (yet)
            
            if(rel_action == 0): # counter-clockwise movement
                if (current_location>0):
                    self._robot_locations[robot_no] = current_location - 1
                
                if (current_location==0): # cycle round
                    self._robot_locations[robot_no] = self.size-1
            
            if(rel_action == 1): # clockwise movement
                if (current_location<self.size-1):
                    self._robot_locations[robot_no]  = current_location + 1
                
                if (current_location==self.size-1): # cycle round
                    self._robot_locations[robot_no] = 0
                    
            if (rel_action == 2): # engage robot, complete task
                for i in range(len(self._goal_locations)): # iterate over locations and mark appropriate goals as done
                    if(self._goal_locations[i] == current_location and self._goal_status[i]==False):
                        self._goal_status[i] = True
                        reward += 1.0 # reward if robots manage to complete a task
                    
            self._robot_clocks[robot_no] = True # lock robot until clock ticks
            
            if np.sum(self._robot_clocks) == self.num_robots: # check if a tick should happen
                self._clock_tick()
                self.elapsed += 1
    
    
            terminated = True
            for status in self._goal_status:
                if status == False:        
                    terminated = False # not terminated if any goals are left
                        
    
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        
        font = pygame.font.SysFont('Arial', 25)
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size/4)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
    
        canvas = pygame.Surface((self.window_size, self.window_size/4))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # draw all positions
        for i in range(self.size):
            pygame.draw.rect(canvas,
                             (144,144,144),
                             pygame.Rect((pix_square_size*i ,0),
                                         (pix_square_size, pix_square_size*0.5)))
        
        # draw the goals
        for i in range(self.num_goals):
            pygame.draw.rect(canvas,
                             ((255,0,0) if self._goal_status[i]==False else (0,255,0)),
                             pygame.Rect((pix_square_size * self._goal_locations[i],0),
                                         (pix_square_size, pix_square_size)))
        
        # draw robots
        for i in range(self.num_robots):
            rect = pygame.draw.rect(canvas,
                             (0,0,255),
                             pygame.Rect((pix_square_size * self._robot_locations[i],pix_square_size),
                                         (pix_square_size, 2*pix_square_size)))
            if(self._robot_clocks[i]):
                canvas.blit(font.render(str(i)+ "'", True, (255,255,255)), rect)
            else:
                canvas.blit(font.render(str(i), True, (255,255,255)), rect)
                        
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
            
