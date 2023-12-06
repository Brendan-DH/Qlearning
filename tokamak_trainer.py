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


env_to_use = "Tokamak-v7"


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

# saved_weights_name = "policy_weights_735539594"
scenario_id = 108186

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


#%%
trained_dqn, dur, re, eps = DQN.train_model(env,
                                            policy_net,
                                            target_net,
                                            None,
                                            alpha=1e-3,
                                            gamma=0.5,
                                            num_episodes=3000,
                                            min_epsilon_time=1000,
                                            epsilon_min=0.05,
                                            usePseudorewards=False,
                                            batch_size=256)

filename = f"op_size{starting_parameters.size}_active{len(starting_parameters.robot_status)}_{scenario_id}"
print(f"Saving as {filename}")
torch.save(trained_dqn.state_dict(), f"./outputs/{filename}")

#%%


s, a, times, ts = DQN.evaluate_model(dqn=policy_net,
                                     num_episodes=1000,
                                     system_parameters=starting_parameters,
                                     env_name=env_to_use,
                                     render=True)

plt.plot(ts)
