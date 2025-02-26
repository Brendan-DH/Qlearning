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

# use a non-display backend. no, i don't know what this means.
matplotlib.use('Agg')
sys.stdout.flush()

input_dict = {
    "environment": "Tokamak-v14",
    "scenario": None,
    "system_logic": None,
    "nodes_per_layer": 256,
    "epsilon_min": 0.05,
    "alpha": 1e-3,
    "gamma": 0.5,
    "num_episodes": 300,
    "tau": 0.005,
    "usePseudorewards": False,  # something wrong with these. investigate noisy rewards.
    "plot_frequency": 20,
    "memory_sort_frequency": 5,
    "max_steps": 200,
    "buffer_size": 50000,
    "checkpoint_frequency": 50,
    "batch_size": 256
}

if ("default_inputs.in" not in os.listdir(os.getcwd())):
    print(f"Saving default inputs to {os.getcwd()}'/default_inputs.in'")
    with open("default_inputs.in", "w") as file:
        file.write("# default input parameters for tokamak_trainer.py\n")
        for key, value in input_dict.items():
            file.write(f"{key}={value}\n")
    file.close()

# get input from stdin
print("Attempting to read input file.")
stdin = sys.stdin
if (select.select([sys.stdin,],[],[],0.0)[0]):
    for line in stdin:
        if (line[0] == "#"):
            continue
        key, value = line.strip().split("=")
        if (key in input_dict.keys()):
            input_dict[key] = value
        else:
            print(f"Warning: input variable '{key}' not understood, skipping.")
else:
    print("No input file specified, exiting.")
    sys.exit(1)


dict_string = '\n'.join('{0}: {1}'.format(k, v)  for k,v in input_dict.items())
print(f"Input dictionary:\n----\n{dict_string}\n----\n")

scenario = getattr(scenarios, input_dict["scenario"], None)
try:
    mdpt = importlib.import_module(f"system_logic.{input_dict['system_logic']}")
except ModuleNotFoundError:
    print(f"System logic {input_dict['system_logic']} was not found")
    sys.exit(1)

if not scenario:
    print(f"Scenario {input_dict['scenario']} was not found.")
    sys.exit(1)

render = os.environ.get("RENDER_TOKAMAK", "0") == "1"
if (render):
    print("RENDERING IS TURNED ON")
else:
    print("RENDERING IS TURNED OFF")

env_to_use = input_dict["environment"]
saved_weights_name = ""
env = gym.make(env_to_use,
               system_parameters=scenarios.rects_id19_case_peaked,
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

if (saved_weights_name != ""):
    print(f"Loading from '/outputs/{saved_weights_name}")
    policy_net.load_state_dict(torch.load(
        os.getcwd() + "/outputs/" + saved_weights_name))
else:
    print("No saved weights defined, starting from scratch")
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
    saved_weights_name = f"saved_weights_{random_identifier}"
    print(f"Saving as {saved_weights_name}")
    torch.save(trained_dqn.state_dict(), f"./outputs/{saved_weights_name}")

print("\nEvaluation by trail...")
s, a, steps, deadlock_traces = DQN.evaluate_model(dqn=policy_net,
                                                  num_episodes=10,
                                                  env=env,
                                                  max_steps=300,
                                                  render=render)

plt.figure(figsize=(10, 7))
plt.hist(x=steps, rwidth=0.95)
plt.xlabel("Total env steps")

# this makes slurm think that the job was nice :) (a lie)
sys.exit(os.EX_OK)

print("Generate DTMC file...")
GenerateDTMCFile(os.getcwd() + "/outputs/" + saved_weights_name,
                 env, f"dtmc_of_{saved_weights_name}")


verification_property = "Rmax=?[F \"done\"]"

subprocess.run(["storm",
                "--explicit",
                f"outputs/dtmc_of_{saved_weights_name}.tra",
                f"outputs/dtmc_of_{saved_weights_name}.lab",
                "--transrew",
                f"outputs/dtmc_of_{saved_weights_name}.transrew",
                "--prop",
                verification_property])
