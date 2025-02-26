#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:12:23 2023

@author: brendandevlin-hill
"""
import importlib
from os.path import split
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import DQN
import os
import numpy as np
import handle_input
from mdp_translation import GenerateDTMCFile
import subprocess
import sys
import scenarios
import select

# use a non-display backend. no, i don't know what this means.
matplotlib.use('Agg')
sys.stdout.flush()

input_dict = handle_input.get_input_dict()

scenario = getattr(scenarios, input_dict["scenario"], None)
try:
    # mdpt = importlib.import_module(f"system_logic.{input_dict['system_logic']}")
    mdpt = importlib.import_module(f"system_logic.hybrid_system_tensor_logic")
except ModuleNotFoundError:
    print(f"System logic {input_dict['system_logic']} was not found")
    sys.exit(1)

if not scenario:
    print(f"Scenario {input_dict['scenario']} was not found.")
    sys.exit(1)

env_to_use = input_dict["environment"]

env = gym.make(env_to_use,
               system_parameters=scenario,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               training=True,
               render=False)

nodes_per_layer = int(input_dict["nodes_per_layer"])  # default 128

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

n_actions = env.action_space.n
state_tensor, info = env.reset()
n_observations = len(state_tensor)

policy_net = DQN.DeepQNetwork(n_observations, n_actions, nodes_per_layer)

saved_weights_name = input_dict["saved_weights_file"]

target_net = DQN.DeepQNetwork(n_observations, n_actions, nodes_per_layer)
target_net.load_state_dict(policy_net.state_dict())
trained_dqn, dur, re, eps = DQN.train_model(env,
                                            policy_net,
                                            target_net,
                                            epsilon_decay_function=lambda ep, e_max, e_min, num_eps: DQN.exponential_epsilon_decay(episode=ep, epsilon_max=e_max, epsilon_min=e_min,
                                                                                                                                   num_episodes=num_eps,
                                                                                                                                   max_epsilon_time=0, min_epsilon_time=0),
                                            epsilon_min=float(input_dict["epsilon_min"]),
                                            alpha=float(input_dict["alpha"]),
                                            gamma=float(input_dict["gamma"]),
                                            num_episodes=int(input_dict["num_episodes"]),
                                            tau=float(input_dict["tau"]),
                                            usePseudorewards=False,
                                            plot_frequency=int(input_dict["plot_frequency"]),
                                            memory_sort_frequency=int(input_dict["memory_sort_frequency"]),
                                            max_steps=int(input_dict["max_steps"]),
                                            buffer_size=int(input_dict["buffer_size"]),
                                            checkpoint_frequency=int(input_dict["checkpoint_frequency"]),
                                            batch_size=int(input_dict["batch_size"]),
                                            )

random_identifier = int(np.random.rand() * 1e6)
new_file_name = f"saved_weights_{random_identifier}"

if saved_weights_name is None:
    print(f"Saving weights as {new_file_name}")
    output_name = new_file_name
elif (saved_weights_name in os.listdir(os.getcwd() + "/outputs")):
    print(f"File {saved_weights_name} already exists. Saving as {new_file_name} instead")
    output_name = new_file_name
else:
    print(f"Saving weights as f{saved_weights_name}")
    output_name = saved_weights_name

torch.save(trained_dqn.state_dict(), f"./outputs/{output_name}")

sys.exit(1)