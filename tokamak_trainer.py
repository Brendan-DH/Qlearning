#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:12:23 2023

@author: brendandevlin-hill
"""

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import DQN
import os
import numpy as np
# from abc import ABC, abstractmethod
env_to_use = "Tokamak-v8"


starting_parameters = DQN.system_parameters(
    size=12,
    robot_status=[1,1,1],
    robot_locations=[1,5,6],
    breakage_probability=0.0001,
    goal_locations=[11,5,2,10,9,8],
    goal_probabilities=[0.49, 0.9, 0.7, 0.7, 0.4, 0.7],
    goal_instantiations=[0,1,1,1,0,0],
    goal_resolutions=[0,1,1,1,0,1],
    goal_checked=[1,1,1,1,1,1,0],
    port_locations=[0,11],
    elapsed=0,
)

env = gym.make(env_to_use,
               system_parameters=starting_parameters,
               training=True,
               render_mode=None)

state, info = env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_weights_name = "saved_weights_182634"
scenario_id = 108186


class Action():

    def __init__(self, name, environment, effect_function, execute_function):
        self.name = name  # human-readable name of the action
        self.environment = environment  # environment to which the action is applied
        self.effect_function = effect_function  # dict of probability / state describing result on 'state'
        self.execute_function = execute_function  # lambda function describing effect of action on environment

    def execute(self):
        self.effect_function(self.environment)

    def effect(self, state):
        effect_dict = self.effect_function(self.environment, state)
        return effect_dict


# how do I deal with the action number that is needed for this logic??
# a generic effect function should not need a action number. template functions can take one
# the effect function is the function which describes the effect of an action
def template_move(env, state, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    # the effect function for moving robot 1 ccw
    ef_dict = {}
    new_state = state.copy()
    robot_no = int(np.floor(action_no / env.num_actions))
    current_location = new_state[f"robot{robot_no} location"]
    robot_no = int(np.floor(action_no / env.num_actions))
    rel_action = action_no % env.num_actions

    # deterministic part of the result:
    if(rel_action == 0):
        # counter-clockwise
        if (current_location < env.size - 1):
            new_state[f"robot{robot_no} location"] = current_location + 1
        if (current_location == env.size - 1):  # cycle round
            new_state[f"robot{robot_no} location"] = 0
    elif(rel_action == 1):
        # clockwise
        if (current_location > 0):
            new_state[f"robot{robot_no} location"] = current_location - 1
        if (current_location == 0):  # cycle round
            new_state[f"robot{robot_no} location"] = env.size - 1
    else:
        raise ValueError("Error: invalid action number for movement effect function!")

    # if there is no goal here:
    if(new_state[f"robot{robot_no} location"] not in env.goal_locations):
        ef_dict = {"1" : new_state}
        return ef_dict

    # if there is a goal here, get the index (i.e. which goal it is)
    for i, value in enumerate(env.goal_locations):
        if value == new_state[f"robot{robot_no} location"]:
            goal_index = i
            break

    # check if this goal has already been resolved
    if (state[f"goal{goal_index} checked"] == 1):
        ef_dict = {"1" : new_state}
        return ef_dict

    # if a goal needs to be revealed:
    else:
        # this goal is now checked
        new_state[f"goal{goal_index} checked"] = 1

        # this goal exists
        prob1 = env.goal_probabilities[goal_index]
        state1 = new_state.copy()
        state1[f"goal{goal_index} instantiated"] = 1
        ef_dict[str(prob1)] = state1

        # this goal does not exist
        prob2 = round(1 - env.goal_probabilities[goal_index], 5)  # this may cause problems!!!!!
        state2 = new_state.copy()
        state2[f"goal{goal_index} instantiated"] = 0
        ef_dict[str(prob2)] = state2

        return ef_dict


# then these functions are specific effect functions for certain robots moving in certain directions:
def r0_ccw(env, state):
    return template_move(env, state, 0)


def r0_cw(env, state):
    return template_move(env, state, 1)


def r1_ccw(env, state):
    return template_move(env, state, 3)


def r1_cw(env, state):
    return template_move(env, state, 4)


def r2_ccw(env, state):
    return template_move(env, state, 6)


def r2_cw(env, state):
    return template_move(env, state, 7)


r1_ccw(env, state)
r2_cw(env, state)

# move_1_ccw = Action("move 1 ccw", env, ef_func, ex_func)


#%%
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
try:
    assert saved_weights_name
    print(f"Loading from '/outputs/{saved_weights_name}'")
    policy_net.load_state_dict(torch.load(os.getcwd() + "/outputs/" + saved_weights_name))
except NameError:
    print("No saved weights defined, starting from scratch")
    target_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    trained_dqn, dur, re, eps = DQN.train_model(env,
                                                policy_net,
                                                target_net,
                                                None,
                                                alpha=1e-3,
                                                gamma=0.5,
                                                num_episodes=1500,
                                                min_epsilon_time=500,
                                                epsilon_min=0.05,
                                                usePseudorewards=False,
                                                batch_size=256)

    filename = f"saved_weights_{int(np.random.rand()*1e6)}"
    print(f"Saving as {filename}")
    torch.save(trained_dqn.state_dict(), f"./outputs/{filename}")

#%%


s, a, times, ts = DQN.evaluate_model(dqn=policy_net,
                                     num_episodes=100,
                                     system_parameters=starting_parameters,
                                     env_name=env_to_use,
                                     render=False)

plt.figure(figsize=(10,7))
plt.hist(x=ts, rwidth=0.95)
plt.xlabel("Total env steps")
