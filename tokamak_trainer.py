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

env_to_use = "Tokamak-v8"


starting_parameters = DQN.system_parameters(
    size=12,
    robot_status=[1,1,1],
    robot_locations=[1,5,6],
    breakage_probability=0.0001,
    goal_locations=[11,5,2,10,9,8],
    goal_probabilities=[0.5, 0.9, 0.7, 0.7, 0.4, 0.7],
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


# how do I deal with the action number that is needed for this logic???
def ef_func(env, action_no, state): 
    # the effect function for moving robot 1 ccw
    ef_dict = {}
    new_state = state.copy()
    robot_no = int(np.floor(action_no / env.num_actions))
    current_location = state["robot_locations"][robot_no]
    robot_no = int(np.floor(action_no / env.num_actions))

    # deterministic part of the result:
    if (current_location < env.size - 1):
        new_state["robot_locations"][robot_no] = current_location + 1
    if (current_location == env.size - 1):  # cycle round
        new_state["robot_locations"][robot_no] = 0

    # if there is no goal here:
    if(new_state["robot_locations"][robot_no] not in env.goal_locations):
        ef_dict = {"1" : new_state}
        return ef_dict

    # if there is a goal but it's already done:
    goal_index = env.goal_locations == new_state["robot_locations"][robot_no]
    if (state["goal_checked"][goal_index] == 0):
        ef_dict = {"1" : new_state}
        return ef_dict

    # if a goal needs to be revealed:
    else:
        new_state["goal_checked"][goal_index] = 1
        prob1 = env.goal_probabilities[goal_index]
        state1 = new_state.copy()
        state1["goal_instantiations"][goal_index] = 1
        ef_dict[str(prob1)] = state1

        prob2 = 1 - env.goal_probabilities[goal_index]
        state2 = new_state.copy()
        state2["goal_instantiations"][goal_index] = 0
        ef_dict[str(prob2)] = state2

        return ef_dict


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
