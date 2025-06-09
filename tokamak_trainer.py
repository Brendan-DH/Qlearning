#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:12:23 2023

@author: brendandevlin-hill
"""
import importlib
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import handle_input
import sys
import scenarios

import dqn.multiagent_training as ma_training
import dqn.training as training
from dqn.dqn import DeepQNetwork
from dqn.decay_functions import linear_epsilon_decay, exponential_epsilon_decay

# use a non-display backend. Honestly not sure of the purpose
matplotlib.use('Agg')
sys.stdout.flush()

input_dict = handle_input.get_input_dict()

scenario = getattr(scenarios, input_dict["scenario"], None)

try:
    mdpt = importlib.import_module(f"system_logic.{input_dict['system_logic']}")
except ModuleNotFoundError:
    print(f"System logic {input_dict['system_logic']} was not found")
    sys.exit(1)

if not scenario:
    print(f"Scenario {input_dict['scenario']} was not found.")
    sys.exit(1)

if (input_dict['epsilon_decay_type'] == "exponential"):
    epsilon_function = exponential_epsilon_decay
elif (input_dict['epsilon_decay_type'] == "linear"):
    epsilon_function = linear_epsilon_decay
else:
    print(f"Epsilon decay type '{input_dict['epsilon_decay_type']}' not recognised. Exiting.")
    sys.exit(1)

env_to_use = input_dict["environment"]

env = gym.make(env_to_use,
               system_parameters=scenario,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               pseudoreward_function=mdpt.pseudoreward_function,
               initial_state_logic=mdpt.initial_state_logic,
               training=True,
               render=input_dict["render_training"].lower() == "y",
        )

nodes_per_layer = int(input_dict["nodes_per_layer"])  # default 128
num_hidden_layers = int(input_dict["num_hidden_layers"])

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

n_actions = env.action_space.n
obs_state, info = env.reset()
n_observations = len(obs_state)


def sinusoidal_epsilon(episode, base_epsilon):
    period = 100
    epsilon_range = 0.15
    epsilon = base_epsilon + np.sin(episode * (np.pi / period)) * (epsilon_range / 2)
    if epsilon < 0:
        epsilon = 0
    return epsilon


run_id = input_dict["run_id"]
overwrite = input_dict["overwrite_saved_weights"].lower() == "y"

multiagent = input_dict["multiagent"] == "y"
if multiagent:
    print("Multiagent training")


    def block_illegal_actions(action_utilities):
        blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_dict, env.unwrapped.clock)
        x = torch.where(blocked, 0, action_utilities)
        print("devices: ", action_utilities.device, blocked.device)
        return x
        # return action_utilities


    policy_net = DeepQNetwork(n_observations + 2, n_actions, num_hidden_layers, nodes_per_layer, block_illegal_actions)
    target_net = DeepQNetwork(n_observations + 2, n_actions, num_hidden_layers, nodes_per_layer, block_illegal_actions)
    target_net.load_state_dict(policy_net.state_dict())

    trained_dqn, dur, re, eps = ma_training.train_model(env,
                                                        policy_net,
                                                        target_net,
                                                        epsilon_decay_function=lambda ep, e_max, e_min, num_eps: epsilon_function(episode=ep,
                                                                                                                                  epsilon_max=e_max,
                                                                                                                                  epsilon_min=e_min,
                                                                                                                                  num_episodes=num_eps,
                                                                                                                                  max_epsilon_time=float(input_dict["max_epsilon_time"]),
                                                                                                                                  min_epsilon_time=float(input_dict["min_epsilon_time"])),
                                                        # epsilon_decay_function=lambda ep, e_max, e_min, num_eps: sinusoidal_epsilon(episode=ep, base_epsilon=e_min),
                                                        epsilon_max=float(input_dict["epsilon_max"]),
                                                        epsilon_min=float(input_dict["epsilon_min"]),
                                                        optimisation_frequency=int(input_dict["optimisation_frequency"]),
                                                        alpha=float(input_dict["alpha"]),
                                                        gamma=float(input_dict["gamma"]),
                                                        num_episodes=int(input_dict["num_training_episodes"]),
                                                        tau=float(input_dict["tau"]),
                                                        use_pseudorewards=input_dict["use_pseudorewards"].lower() == "y",
                                                        plot_frequency=int(input_dict["plot_frequency"]),
                                                        memory_sort_frequency=int(input_dict["memory_sort_frequency"]),
                                                        max_steps=int(input_dict["max_steps"]),
                                                        buffer_size=int(input_dict["buffer_size"]),
                                                        checkpoint_frequency=int(input_dict["checkpoint_frequency"]),
                                                        batch_size=int(input_dict["batch_size"]),
                                                        reward_sharing_coefficient=float(input_dict["reward_sharing_coefficient"]),
                                                        run_id=run_id
                                                        )
else:
    if not torch.cuda.is_available():
        # def block_illegal_actions(action_utilities):
        #     blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_dict, env.unwrapped.state_dict["clock"])
        #     x = torch.where(blocked, 0, action_utilities)
        #     return x
        def block_illegal_actions(action_utilities):
            return action_utilities
        
    else:
        # we can't use the blocked_model on the GPU because it would sabotage performance with device transfers
        def block_illegal_actions(action_utilities):
            return action_utilities


    policy_net = DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer, block_illegal_actions)
    policy_net_gpu = DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer, block_illegal_actions)
    target_net = DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer, block_illegal_actions)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net_gpu.load_state_dict(policy_net.state_dict())

    trained_dqn, dur, re, eps = training.train_model(env,
                                                     policy_net,
                                                     target_net,
                                                     policy_net_gpu,
                                                     epsilon_decay_function=lambda ep, e_max, e_min, num_eps: epsilon_function(episode=ep,
                                                                                                                               epsilon_max=e_max,
                                                                                                                               epsilon_min=e_min,
                                                                                                                               num_episodes=num_eps,
                                                                                                                               max_epsilon_time=float(input_dict["max_epsilon_time"]),
                                                                                                                               min_epsilon_time=float(input_dict["min_epsilon_time"])),
                                                     epsilon_max=float(input_dict["epsilon_max"]),
                                                     epsilon_min=float(input_dict["epsilon_min"]),
                                                     optimisation_frequency=int(input_dict["optimisation_frequency"]),
                                                     alpha=float(input_dict["alpha"]),
                                                     gamma=float(input_dict["gamma"]),
                                                     num_episodes=int(input_dict["num_training_episodes"]),
                                                     tau=float(input_dict["tau"]),
                                                     use_pseudorewards=input_dict["use_pseudorewards"].lower() == "y",
                                                     plot_frequency=int(input_dict["plot_frequency"]),
                                                     memory_sort_frequency=int(input_dict["memory_sort_frequency"]),
                                                     max_steps=int(input_dict["max_steps"]),
                                                     buffer_size=int(input_dict["buffer_size"]),
                                                     checkpoint_frequency=int(input_dict["checkpoint_frequency"]),
                                                     batch_size=int(input_dict["batch_size"]),
                                                     run_id=run_id
                                                     )

random_id = int(np.random.rand() * 1e6)
if run_id is None:
    print("No run_id was specified in the input file, so one has been assigned: ", random_id)
    run_id = random_id

file_name = f"weights_{run_id}"

if (file_name in os.listdir(os.getcwd() + "/outputs/saved_weights") and not overwrite):
    print(f"File {file_name} already exists. Saving as {file_name}_{random_id} instead")
    output_name = f"{file_name}_{random_id}"
else:
    print(f"Saving weights as weights_{run_id}")
    output_name = file_name

torch.save(trained_dqn.state_dict(), f"./outputs/saved_weights/{output_name}")

sys.exit(0)
