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
import system_logic.hybrid_system as mdpt
from mdp_translation import GenerateDTMCFile
# from abc import ABC, abstractmethod
env_to_use = "Tokamak-v13"
env_size = 12

# large_case = DQN.system_parameters(
#     size=env_size,
#     robot_status=[1,1,1],
#     robot_locations=[1,2,3],
#     goal_locations=[i for i in range(env_size)],
#     goal_probabilities=[0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95],
#     goal_activations=[1 for i in range(0, env_size)],
#     elapsed_ticks=0,
# )

small_case1 = DQN.system_parameters(
    size=env_size,
    robot_status=[1,1,1],
    robot_locations=[1,2,3],
    goal_locations=[11, 5, 7],
    goal_discovery_probabilities=[0.95, 0.95, 0.95],
    goal_completion_probabilities=[0.95, 0.95, 0.95],
    goal_checked=[0,0,0],
    goal_activations=[0,0,0],
    elapsed_ticks=0,
)

# small_case2 = DQN.system_parameters(
#     size=env_size,
#     robot_status=[1,1,1],
#     robot_locations=[1, 5, 6],
#     goal_locations=[11, 3, 5],
#     goal_probabilities=[0.95, 0.95, 0.95],
#     goal_activations=[1,1,1],
#     elapsed_ticks=0,
# )

case_5goals = DQN.system_parameters(
    size=env_size,
    robot_status=[1,1,1],
    robot_locations=[1, 2, 7],
    goal_locations=[11, 3, 5, 4, 6],
    goal_discovery_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7],
    goal_completion_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7],
    goal_activations=[0,0,0,0,0],
    goal_checked=[0,0,0,0,0],
    elapsed_ticks=0,
)

case_7goals = DQN.system_parameters(
    size=env_size,
    robot_status=[1,1,1],
    robot_locations=[1, 5, 6],
    goal_locations=[11, 3, 5, 4, 10, 9, 7],
    goal_discovery_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    goal_completion_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    goal_activations=[0 for i in range(7)],
    goal_checked=[0 for i in range(7)],
    elapsed_ticks=0,
)

# large_case_flat = DQN.system_parameters(
#     size=env_size,
#     robot_status=[1,1,1],
#     robot_locations=[1,5,6],
#     goal_locations=[i for i in range(env_size)],
#     goal_probabilities=[0.95 for i in range(env_size)],
#     goal_activations=[1 for i in range(env_size)],
#     elapsed_ticks=0,
# )

# large_case_peaked = DQN.system_parameters(
#     size=env_size,
#     robot_status=[1,1,1],
#     robot_locations=[1,5,6],
#     goal_locations=[i for i in range(env_size)],
#     goal_completion_probabilities=[0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95],
#     goal_discovery_probabilities=[0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95],
#     goal_activations=[0 for i in range(env_size)],
#     goal_checked=[0 for i in range(env_size)],
#     elapsed_ticks=0,
# )

# case_0goals = DQN.system_parameters(
#     size=env_size,
#     robot_status=[1,1,1],
#     robot_locations=[1,5,6],
#     goal_locations=[env_size + 1],  # will never be completed
#     goal_probabilities=[1],
#     goal_activations=[1 for i in range(env_size)],
#     elapsed_ticks=0,
# )

env = gym.make(env_to_use,
               system_parameters=case_7goals,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               training=True,
               render=False,
               render_ticks_only=True)

nodes_per_layer = 128 * 2  # default 128

state, info = env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# saved_weights_name = "saved_weights_770884"
# scenario_id = 108186


def decay_function(ep, e_max, e_min, num_eps):
    return DQN.exponential_epsilon_decay(episode=ep, epsilon_max=e_max, epsilon_min=e_min, num_episodes=num_eps, max_epsilon_time=0, min_epsilon_time=0)


#%%
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN.DeepQNetwork(n_observations, n_actions, nodes_per_layer).to(device)
try:
    assert saved_weights_name
    print(f"Loading from '/outputs/{saved_weights_name}'")
    policy_net.load_state_dict(torch.load(os.getcwd() + "/outputs/" + saved_weights_name))
except NameError:
    print("No saved weights defined, starting from scratch")
    target_net = DQN.DeepQNetwork(n_observations, n_actions, nodes_per_layer).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    trained_dqn, dur, re, eps = DQN.train_model(env,
                                                policy_net,
                                                target_net,
                                                epsilon_decay_function=decay_function,
                                                epsilon_min=0.05,
                                                alpha=1e-3,
                                                gamma=0.3,
                                                num_episodes=500,
                                                tau=0.005,
                                                usePseudorewards=False,   # something wrong with these. investigate noisy rewards.
                                                plot_frequency=10,
                                                memory_sort_frequency=1,
                                                max_steps=200,
                                                buffer_size=5000,
                                                checkpoint_frequency=50,
                                                batch_size=64)

    filename = f"saved_weights_{int(np.random.rand()*1e6)}"
    print(f"Saving as {filename}")
    torch.save(trained_dqn.state_dict(), f"./outputs/{filename}")
    print("Generate DTMC file...")
    GenerateDTMCFile(os.getcwd() + "/outputs/" + filename, env)

#%%


print("\nEvaluation by trail...")
s, a, steps, deadlock_traces = DQN.evaluate_model(dqn=policy_net,
                                                  num_episodes=10000,
                                                  env=env,
                                                  max_steps=100,
                                                  render=False)

plt.figure(figsize=(10,7))
plt.hist(x=steps, rwidth=0.95)
plt.xlabel("Total env steps")

#%%
print(len(deadlock_traces))

trace = deadlock_traces[0]
print(len(trace))
for i in range(len(trace)):
    print(i)
    env.render_frame(trace[i])


# plt.figure(figsize=(10,7))
# plt.hist(x=ticks, rwidth=0.95)

# plt.xlabel("Total env ticks")

# print(ticks)
