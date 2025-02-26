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
from mdp_translation import GenerateDTMCFile
import subprocess
import sys
import scenarios
import select
import handle_input

# use a non-display backend. no, i don't know what this means.
matplotlib.use('Agg')
sys.stdout.flush()

input_dict = handle_input.get_input_dict()

saved_weights = input_dict["saved_weights_file"]
if (not saved_weights):
    print("No weights file provided, exiting.")
    sys.exit(1)

render = input_dict["render_evaluation"].lower() == "true"

scenario = getattr(scenarios, input_dict["scenario"], None)
try:
    mdpt = importlib.import_module(f"system_logic.{input_dict['system_logic']}")
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
print("State is of length", n_observations)

policy_net = DQN.DeepQNetwork(n_observations, n_actions, nodes_per_layer)



print(f"Loading from '/outputs/{saved_weights}")
policy_net.load_state_dict(torch.load(os.getcwd() + "/outputs/" + saved_weights))

print("\nEvaluation by trail...")
s, a, steps, deadlock_traces = DQN.evaluate_model(dqn=policy_net,
                                                  num_episodes=int(input_dict["num_evaluation_episodes"]),
                                                  env=env,
                                                  max_steps=int(input_dict["max_steps"]),
                                                  render=render)

plt.figure(figsize=(10, 7))
plt.hist(x=steps, rwidth=0.95)
plt.xlabel("Total env steps")
plt.savefig(f"outputs/trial_${saved_weights.replace('/', '_')}.svg")

print("Generate DTMC file...")
GenerateDTMCFile(os.getcwd() + "/outputs/" + saved_weights, env, f"dtmc_of_{saved_weights}")

verification_property = "Rmax=?[F \"done\"]"

print("Running STORM")
subprocess.run(["storm",
                "--explicit",
                f"outputs/dtmc_of_{saved_weights}.tra",
                f"outputs/dtmc_of_{saved_weights}.lab",
                "--transrew",
                f"outputs/dtmc_of_{saved_weights}.transrew",
                "--prop",
                verification_property])

sys.exit(0)