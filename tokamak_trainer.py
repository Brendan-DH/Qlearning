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
import system_logic.hybrid_system_tensor_logic as mdpt
from mdp_translation import GenerateDTMCFile
import subprocess
import sys
import scenarios

# use a non-display backend. no, i don't know what this means.
matplotlib.use('Agg')
sys.stdout.flush()

env_to_use = "Tokamak-v14"
saved_weights_name = "" #"saved_weights_999862"
env = gym.make(env_to_use,
               system_parameters=scenarios.case_5goals,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               training=True,
               render=False,
               render_ticks_only=True)

nodes_per_layer = 128 * 2  # default 128

state_tensor, info = env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on '{device}'")

n_actions = env.action_space.n
state_tensor, info = env.reset()
n_observations = len(state_tensor)
print("State is of length", n_observations)

policy_net = DQN.DeepQNetwork(
    n_observations, n_actions, nodes_per_layer).to(device)
if (saved_weights_name != ""):
    print(f"Loading from '/outputs/{saved_weights_name}")
    policy_net.load_state_dict(torch.load(
        os.getcwd() + "/outputs/" + saved_weights_name))
else:
    print("No saved weights defined, starting from scratch")
    target_net = DQN.DeepQNetwork(
        n_observations, n_actions, nodes_per_layer).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    trained_dqn, dur, re, eps = DQN.train_model(env,
                                                policy_net,
                                                target_net,
                                                epsilon_decay_function=lambda ep, e_max, e_min, num_eps: DQN.exponential_epsilon_decay(episode=ep, epsilon_max=e_max, epsilon_min=e_min, num_episodes=num_eps,
                                                                                                                                       max_epsilon_time=0, min_epsilon_time=0),
                                                epsilon_min=0.05,
                                                alpha=1e-3,
                                                gamma=0.5,
                                                num_episodes=300,
                                                tau=0.005,
                                                # something wrong with these. investigate noisy rewards.
                                                usePseudorewards=False,
                                                plot_frequency=20,
                                                memory_sort_frequency=5,
                                                max_steps=200,
                                                buffer_size=50000,
                                                checkpoint_frequency=50,
                                                batch_size=128 * 2)

    random_identifier = int(np.random.rand() * 1e6)
    saved_weights_name = f"saved_weights_{random_identifier}"
    print(f"Saving as {saved_weights_name}")
    torch.save(trained_dqn.state_dict(), f"./outputs/{saved_weights_name}")

print("\nEvaluation by trail...")
s, a, steps, deadlock_traces = DQN.evaluate_model(dqn=policy_net,
                                                  num_episodes=200,
                                                  env=env,
                                                  max_steps=300,
                                                  render=False)

plt.figure(figsize=(10, 7))
plt.hist(x=steps, rwidth=0.95)
plt.xlabel("Total env steps")

print("Generate DTMC file...")
GenerateDTMCFile(os.getcwd() + "/outputs/" + saved_weights_name,
                 env, f"dtmc_of_{saved_weights_name}")


# this makes slurm think that the job was nice :) (a lie)
# don't know if actually needed
# sys.exit(os.EX_OK)

verification_property = "Rmax=?[F \"done\"]"

subprocess.run(["storm",
                "--explicit",
                f"outputs/dtmc_of_{saved_weights_name}.tra",
                f"outputs/dtmc_of_{saved_weights_name}.lab",
                "--transrew",
                f"outputs/dtmc_of_{saved_weights_name}.transrew",
                "--prop",
                verification_property])

# %%
print(len(deadlock_traces))



for i in range(len(deadlock_traces)):
    trace = deadlock_traces[i]
    print(len(trace))
    for j in range(len(trace)):
        print(trace[j]["robot0 clock"])
        env.render_frame(trace[j])


# plt.figure(figsize=(10, 7))
# plt.hist(x=ticks, rwidth=0.95)
#
# plt.xlabel("Total env ticks")
#
# print(ticks)
