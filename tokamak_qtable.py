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

num_segments = 24
max_segment = num_segments-1



damaged_segments = [1,5,9]
num_robots = 3

print("Generating states...")

states = []
states_dict = {}
initial_indices = []
absorbing_indices = []
counter = 0

for r1_pos in range(0, num_segments):
    for r2_pos in range(0, num_segments):
        for r3_pos in range(0,num_segments):

            for d1 in [-1, damaged_segments[0]]:
                for d2 in [-1, damaged_segments[1]]:
                    for d3 in [-1, damaged_segments[2]]:

                        state = np.array([r1_pos, r2_pos, r3_pos, d1, d2, d3])
                        states.append(state)
                        states_dict[str(state)] = counter
                        
                        if(d1 == -1 and d2 == -1 and d3 == -1):
                            absorbing_indices.append(counter)
                        elif (d1 >= 0 and d2 >= 0 and d3 >= 0):
                            initial_indices.append(counter)
                        
                        counter+=1
            
                        
print(f"{len(initial_indices)} initial states, {len(absorbing_indices)} absorbing states")

#%%

# define possible state transitions. [state, state, state, state...]
# the row index corresponds to the outgoing state
# the column index corresponds to the action taken
# the contents of [i,j] corresponds to the index of the resultant state



transitions = np.zeros((len(states), num_robots*2+len(damaged_segments))) #two actions per robot, one per damaged segment
transition_rewards = np.zeros((len(states),9))

print("Generating transitions and rewards...")

for i in range(len(states)):
    
    if (i%1000==0):
        print("Transitions from state {: >5g}/{:g} ({: >4.2f}%)".format(i, len(states), i*100/len(states)))
    
    # get the state of the system
    r1_pos, r2_pos, r3_pos, *d_segs = states[i]
    
    # array of states that can be accessed from current state index
    available_states = np.ones(9)*-1
    t_rewards = np.zeros(9)
    
    
    # robot 1 actions:
    if (r1_pos<max_segment):
        available_states[0] = states_dict[str(np.hstack(([r1_pos+1, r2_pos, r3_pos], d_segs)))]
        t_rewards[0] = 0
    if (r1_pos>0):    
        available_states[1] = states_dict[str(np.hstack(([r1_pos-1, r2_pos, r3_pos], d_segs)))]
        t_rewards[1] = 0
    if (r1_pos in d_segs):
        ds = [-1 if (a==r1_pos) else a for a in d_segs]
        available_states[2] = states_dict[str(np.hstack(([r1_pos, r2_pos, r3_pos], ds)))]
        t_rewards[2] = len(np.argwhere(np.array(ds) < 0))**2
    
    # robot 2 actions:
    if (r2_pos<max_segment):
        available_states[3] = states_dict[str(np.hstack(([r1_pos, r2_pos+1, r3_pos], d_segs)))]
        t_rewards[3] = 0
    if (r2_pos>0):
        available_states[4] = states_dict[str(np.hstack(([r1_pos, r2_pos-1, r3_pos], d_segs)))]
        t_rewards[4] = 0
    if (r2_pos in d_segs):
        ds = [-1 if (a==r2_pos) else a for a in d_segs]
        available_states[5] = states_dict[str(np.hstack(([r1_pos, r2_pos, r3_pos], ds)))]
        t_rewards[5] = len(np.argwhere(np.array(ds) < 0))**2

    # robot 3 actions:
    if (r3_pos<max_segment ):
        available_states[6] = states_dict[str(np.hstack(([r1_pos, r2_pos, r3_pos+1], d_segs)))]
        t_rewards[6] = 0
    if (r3_pos>0):
        available_states[7] = states_dict[str(np.hstack(([r1_pos, r2_pos, r3_pos-1], d_segs)))]
        t_rewards[7] = 0
    if (r3_pos in d_segs):
        ds = [-1 if (a==r3_pos) else a for a in d_segs]
        available_states[8] = states_dict[str(np.hstack(([r1_pos, r2_pos, r3_pos], ds)))]
        t_rewards[8] = len(np.argwhere(np.array(ds) < 0))**2

    transitions[i] = available_states
    transition_rewards[i] = t_rewards        

states = np.array(states)

#%%

Qtable = np.random.rand(*np.shape(transitions))
Qtable = Qtable + (transitions < 0) * -1e9

def AverageDistance(system, state, action):
    r_pos, d_seg = system.interpret_state(state)
    tot_av = 0
    for i in range(len(r_pos)):
        rob_av = 0
        for j in range(len(d_seg)):
            rob_av += abs(r_pos[i] - d_seg[j])/len(d_seg)
        tot_av += rob_av/len(r_pos)
    return (10/tot_av)

def EmptyRewards(system, state, action):
    return 0
    
def InterpretState(system, state):
    
    r_pos = state[:3]
    d_segs = state[3:]
        
    return r_pos, d_segs

system = ql.GeneralisedSystem(states, initial_indices, absorbing_indices, transitions, transition_rewards, AverageDistance, InterpretState)

agent = ql.TrainGeneral(Qtable, 
                        system,
                        max_steps = 500,
                        n_training_episodes = int(1E+6), #1E+7 and 3E-7 decay rate work together nicely
                        min_epsilon = 0.05,
                        max_epsilon = 0.5,
                        decay_rate = 3E-6,
                        gamma = 0.7,
                        learning_rate = 0.5,
                        verbosity=5000)

np.savetxt(os.getcwd() + "/agent.txt", agent)

#%%

agent = np.loadtxt(os.getcwd() + "/agent.txt")

ev = ql.EvaluateGeneral(qtable = agent,
                        system = system,                     
                        n_eval_episodes = 10000,
                        max_steps = 400,
                        gamma = 0.7)

ev
















