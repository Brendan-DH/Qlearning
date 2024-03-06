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
import transition_model as mdpt
# from abc import ABC, abstractmethod
env_to_use = "Tokamak-v10"


starting_parameters = DQN.system_parameters(
    size=12,
    robot_status=[1,1,1],
    robot_locations=[1,5,6],
    goal_locations=[11,3,5,2,8,7],
    goal_probabilities=[0.7,0.7,0.7, 0.7, 0.7,0.7],
    goal_activations=[1,1,1,1,1,1],
    elapsed_ticks=0,
)

env = gym.make(env_to_use,
               system_parameters=starting_parameters,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               training=True,
               render_mode=None)


state, info = env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# saved_weights_name = "saved_weights_23876"  # "saved_weights_182634"
# scenario_id = 108186


def decay_function(ep, e_max, e_min, num_eps):
    return DQN.linear_epsilon_decay(episode=ep, epsilon_max=e_max, epsilon_min=e_min, num_episodes=num_eps, max_epsilon_time=50, min_epsilon_time=50)


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
                                                epsilon_decay_function=decay_function,
                                                # epsilon_min=0,
                                                alpha=1e-3,
                                                gamma=0.2,
                                                # reset_options={"type": "statetree"},
                                                num_episodes=3000,
                                                usePseudorewards=True,
                                                plot_frequency=200,
                                                tree_prune_frequency=1e9,
                                                state_tree_capacity=300,
                                                batch_size=256)

    filename = f"saved_weights_{int(np.random.rand()*1e6)}"
    print(f"Saving as {filename}")
    torch.save(trained_dqn.state_dict(), f"./outputs/{filename}")

#%%


s, a, steps = DQN.evaluate_model(dqn=policy_net,
                                 num_episodes=1000,
                                 env=env,
                                 render=False)

plt.figure(figsize=(10,7))
plt.hist(x=steps, rwidth=0.95)
plt.xlabel("Total env steps")


# plt.figure(figsize=(10,7))
# plt.hist(x=ticks, rwidth=0.95)

# plt.xlabel("Total env ticks")

# print(ticks)
