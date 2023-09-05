#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:34:45 2023

@author: brendandevlin-hill
"""

#%%

import pandas as pd
import numpy as np
import os
import random
import Qlearning as ql



# data directory
data_directory = os.getcwd() + "/data/"

# import a pkl of the normalised energy deposition
intQ = pd.read_pickle(data_directory + "rects_id12_integratedQ.pkl")


# create the q table. 
# each row corresponds to a system state.
# each column corresponds to the quality score of the actions
# actions are wait/move cw/move ccw/inspect

#%%

print("Generating states...")

# generate a 2d grid
grid_size = 4
goal_position = np.array([[3,3]])

states = []

for i in range(0, grid_size):
    for j in range(0, grid_size):
        states.append([i,j])

states = np.array(states)

#%%

# define possible state transitions. [state, state, state, state...]
# the row index corresponds to the outgoing state
# the column index corresponds to the action taken
# the contents of [i,j] corresponds to the index of the resultant state

def isAinB(a,b):
    for entry in b:
        print(entry, a)
        if entry[0] == a[0] and entry[1] == a[1]:
            return True
    return False
            

transitions = np.zeros((len(states),4,2))
transition_rewards = np.zeros((len(states),4))

print("Generating transitions and rewards...")

for i in range(len(states)):
    
    available_states = np.ones((4,2))*-1
    available_rewards = np.zeros(4)
    
    if states[i,0]>0: #left 
        new_state = states[i] - [1,0]
        available_states[0] = new_state
        if isAinB(new_state, goal_position):
            available_rewards[0] = 1
        
    if states[i,0]<(grid_size-1): #right 
        new_state = states[i] + [1,0]
        available_states[1] = new_state
        if isAinB(new_state, goal_position):
            available_rewards[1] = 1
        
    if states[i,1]>0: #up 
        new_state = states[i] - [0,1]
        available_states[2] = new_state
        if isAinB(new_state, goal_position):
            available_rewards[2] = 1
        
    if states[i,1]<(grid_size-1): #down 
        new_state = states[i] + [0,1]
        available_states[3] = new_state
        if isAinB(new_state, goal_position):
            available_rewards[3] = 1
        
    transitions[i] = available_states
    transition_rewards[i] = available_rewards
    
print(transitions)
    
#%%

Qtable = np.random.rand(len(states),4)

def AverageDistancePseudReward(system, state, action):
    r_pos, _, d_seg = system.interpret_state(state)
    tot_av = 0
    for i in range(len(r_pos)):
        rob_av = 0
        for j in range(len(d_seg)):
            rob_av += abs(r_pos[i] - d_seg[j])/len(d_seg)
        tot_av += rob_av/len(r_pos)
    return 10/tot_av

def EmptyPseudoReward(system, state, action):
    return 0

agent = ql.Train(Qtable, 
              ql.PrimitiveSystem([1,5,9], states, transitions, transition_rewards, AverageDistancePseudReward),
              [7],
              transitions,
              max_steps = 100,
              n_training_episodes = int(1E+6), #1E+7 and 3E-7 decay rate work together nicely
              min_epsilon = 0.05,
              max_epsilon = 0.5,
              decay_rate = 3E-6,
              gamma = 0.7,
              learning_rate = 0.2)

np.savetxt(os.getcwd() + "/agent.txt", agent)

#%%

# agent = np.loadtxt(os.getcwd() + "/agent.txt")
# transitions = np.loadtxt(os.getcwd() + "/transitions.txt")
# states = np.loadtxt(os.getcwd() +  "/states.txt")

agent = Qtable

ev = ql.Evaluate(agent,
                 ql.PrimitiveSystem([1,5,9], states, transitions, transition_rewards, AverageDistancePseudReward),
                 [7],
                 transitions,
                 10000,
                 400)

ev
















