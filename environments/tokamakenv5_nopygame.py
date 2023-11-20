#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:34:49 2023

@author: brendandevlin-hill
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TokamakEnv5_nopygame(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # def set_parameters(size, num_robots, num_goals, goal_locations):
    #     return None
    
    def __init__(self,
                 size=None,
                 num_robots=None,
                 goal_locations=None,
                 goal_probabilities = None,
                 render_mode=None):
        
        self.size = size
        self.num_robots = num_robots
        self.num_goals = len(goal_locations)
        self._goal_locations = goal_locations
        self._goal_probabilities = goal_probabilities
        self._original_probabilities = goal_probabilities
        # 1 if goal was there. 0 if goal was not. -1 if goal hasn't been resolved yet
        self._goal_resolutions = np.ones_like(goal_probabilities)*-1 
        self.window_size = 700  # The size of the PyGame window
        self.elapsed = 0
        self.parameters = {
            "size" : size,
            "num_robots" : num_robots,
            "goal_locations" : goal_locations,
            "goal_probabilities" : goal_probabilities
            }
        self.num_actions = 4 # this can be changed for dev purposes
        self.most_recent_actions = np.empty((3), np.dtype('U100'))
        # print(np.shape(self.most_recent_actions))
        
        
        obDict = {}
        
        # positions of the robots
        for i in range(0,num_robots):
            obDict[f"robot{i} location"] = spaces.Discrete(size)
            obDict[f"robot{i} clock"] = spaces.Discrete(2) # true = cannot move
            
        # positions of the goals and whether they are complete
        for i in range(0,self.num_goals):
            obDict[f"goal{i} location"] = spaces.Discrete(size)
            obDict[f"goal{i} probability"] = spaces.Box(low=0, high=1, shape=[1]) # continuous variable in [0,1]. sample() returns array though.

        self.observation_space = spaces.Dict(obDict)
        
        # actions that the robots can carry out
        # move clockwise/anticlockwise, engage
        self.action_space = spaces.Discrete(num_robots*self.num_actions)
        
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
            obs[f"goal{i} probability"] = self._goal_probabilities[i]
            
        return obs
    
    def _get_info(self):
        info = {}
        info["elapsed"] = self.elapsed
        info["av_dist"] = self.av_dist()
        info["goal_resolutions"] = self._goal_resolutions.copy()
        return info
        
    def reset(self, seed=None, options=None):
        # print("reset")
        super().reset(seed=seed)
        if(options):
            self._robot_locations = options["robot_locations"].copy()
        else:
            # reset robot locations to random:
            self._robot_locations = [self.np_random.integers(0,self.size) 
                                     for i in range(self.num_robots)] 
            
        # reset goals
        self._goal_probabilities = self._original_probabilities.copy()
        self.elapsed = 0
        self._robot_clocks = [False for i in range(self.num_robots)] # set all clocks to false
        
        return self._get_obs(), self._get_info()
        
        self.elapsed = 0
        self._goal_resolutions = np.ones_like(self._goal_probabilities) * -1
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), self._get_info()
        
        
    def av_dist(self):
        tot_av = 0
        # goal positions
        for i in range(len(self._robot_locations)):
            rob_pos = self._robot_locations[i]
            rob_av = 0
            num_active_goals = len(self._goal_locations)
            # num_active_goals = np.sum(np.array(self._goal_probabilities) > 0) # status is True if complete; this calculates num False
            for j in range(len(self._goal_locations)):
                if self._goal_probabilities[j] == 0: # this goal is already completed
                    continue
                goal_pos = self._goal_locations[j]
                #calculate average distance of robot from each goal:
                dist = abs(rob_pos - goal_pos) * self._goal_probabilities[j] # weight by the probability that it is actually there
                mod_dist = min((dist, self.size - dist)) # to account for cyclical space
                rob_av += mod_dist/num_active_goals
            # average of average distances
            tot_av += rob_av/len(self._robot_locations)
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
                blocked_actions[i*self.num_actions:(i*self.num_actions)+self.num_actions] = 1 #block all actions for this robot
                
            for j in range(self.num_robots):
                other_robot_loc = self._robot_locations[j]
                
                if(i==j): # don't need to check robots against themselves
                    continue
                
                if (moving_robot_loc == other_robot_loc):
                    raise ValueError(f"Two robots occupy the same location (r{i} & r{j} @ {moving_robot_loc}).")
                
                # block counter-clockwise movement:
                if((other_robot_loc == moving_robot_loc + 1) or (moving_robot_loc == self.size-1 and other_robot_loc == 0)):
                    blocked_actions[(i*self.num_actions)] = 1

                #block clockwise movement:
                if((other_robot_loc == moving_robot_loc - 1) or (moving_robot_loc == 0 and other_robot_loc == self.size-1)):
                    blocked_actions[(i*self.num_actions)+1] = 1
                    
            #block inspection if robot is not over known task location:
            block_inspection = 1
            for k in range(len(self._goal_locations)): 
                if (self._goal_locations[k] == moving_robot_loc and self._goal_probabilities[k] == 1):
                    block_inspection = 0
            blocked_actions[(i*self.num_actions)+2] = block_inspection
                    
        # print(self._robot_locations, blocked_actions)
        return blocked_actions
    
    def _clock_tick(self):
        self._robot_clocks = [False for i in range(self.num_robots)] # set all clocks to false
            
                
    def step(self, action):
    
        # determine blocked actions based on the current state
        # blocked actions give a negative reward and don't progress the system
        rel_action = action % self.num_actions # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait
        # by which robot:
        robot_no = int(np.floor(action/self.num_actions))
        
        blocked_actions = self._get_blocked_actions()
        if(blocked_actions[action]):
            
            terminated = False
            reward = -10
            current_action = "forbidden"
                    
        else: 
            # which action is being taken:
            
            current_location = self._robot_locations[robot_no]
            current_action = ""
            reward = 0.0
            
            if(rel_action == 0): # counter-clockwise movement
                if (current_location<self.size-1):
                    self._robot_locations[robot_no]  = current_location + 1
                if (current_location==self.size-1): # cycle round
                    self._robot_locations[robot_no] = 0
                # reward -= 0.5
                current_action="move ccw"
            
            
            if(rel_action == 1): # clockwise movement
                if (current_location>0):
                    self._robot_locations[robot_no] = current_location - 1                
                if (current_location==0): # cycle round
                    self._robot_locations[robot_no] = self.size-1
                # reward -= 0.5
                current_action="move cw"
                    

                    
            if (rel_action == 2): # engage robot, complete task
                for i in range(len(self._goal_locations)): # iterate over locations and mark appropriate goals as done
                    if(self._goal_locations[i] == current_location and self._goal_probabilities[i]==1):
                        self._goal_probabilities[i] = 0
                        reward += 1000 * 1/(self.elapsed+1) # reward if robots manage to complete a task
                # reward -= 0.5
                current_action="engage"
                
            if (rel_action == 3): # wait; nothing happens, no reward lost
                reward -= 1 # we only want to machine to wait when it has nothing useful to do
                current_action="wait"
                        
            for i in range(len(self._goal_locations)): # iterate over locations and mark appropriate goals as done
                if(self._goal_locations[i] == self._robot_locations[robot_no] and self._goal_probabilities[i] > 0 and self._goal_probabilities[i] < 1):
                    
                    # resolve the non-determinism:
                    if np.random.rand() < self._goal_probabilities[i]:
                        self._goal_probabilities[i] = 1
                        self._goal_resolutions[i] = 1 # feel like there's a more concise way to do this...
                    else:
                        self._goal_probabilities[i] = 0
                        self._goal_resolutions[i] = 0

                    reward += 100 # reward robots for discovering tasks

                        
                    
        self.most_recent_actions[robot_no] = current_action
        # print(current_action, robot_no, self.most_recent_actions, type(self.most_recent_actions[0]))
        self._robot_clocks[robot_no] = True # lock robot until clock ticks
        
        if np.sum(self._robot_clocks) == self.num_robots: # check if a tick should happen
            self._clock_tick()
            self.elapsed += 1

    
        terminated = True
        for prob in self._goal_probabilities:
            if prob > 0:        
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
        pass # cannot render on windows because pygame is not supported
            
