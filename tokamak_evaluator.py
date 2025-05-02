#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:12:23 2023

@author: brendandevlin-hill
"""

import subprocess
import importlib
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import handle_input
import sys
import scenarios

from dqn.dqn import DeepQNetwork
from dqn.evaluation import evaluate_model_by_trial, generate_dtmc_file, generate_mdp_file
from system_logic.hybrid_system_tensor_logic import pseudoreward_function

# use a non-display backend. no, i don't know what this means.
matplotlib.use('Agg')
sys.stdout.flush()

input_dict = handle_input.get_input_dict()

load_weights_file = input_dict["evaluation_weights_file"]
if (not load_weights_file):
    print("No weights file provided, exiting.")
    sys.exit(1)

render = input_dict["render_evaluation"].lower() == "y"

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
               initial_state_logic=mdpt.initial_state_logic,
               training=False,
               render=False)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

n_actions = env.action_space.n
state_tensor, info = env.reset()
n_observations = len(state_tensor)
print(f"Looking for /inputs/{load_weights_file}")
loaded_weights = torch.load(os.getcwd() + "/inputs/" + load_weights_file)
nodes_per_layer = len(loaded_weights["hidden_layers.0.weight"])
num_hidden_layers = int((len(loaded_weights.keys()) - 4) / 2)  # -4 accounts for input and output weights and biases


def block_illegal_actions(action_utilities):
    blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_tensor)
    x = torch.where(blocked, -100000, action_utilities)
    return x


policy_net = DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer, block_illegal_actions)

print(f"Loading from /inputs/{load_weights_file}")
policy_net.load_state_dict(loaded_weights)

if int(input_dict["num_evaluation_episodes"]) > 0:
    print("\nEvaluation by trail...")
    s, a, steps, deadlock_traces = evaluate_model_by_trial(dqn=policy_net,
                                                           num_episodes=int(input_dict["num_evaluation_episodes"]),
                                                           env=env,
                                                           max_steps=int(input_dict["max_steps"]),
                                                           render=render)

    plt.figure(figsize=(10, 7))
    plt.hist(x=steps, rwidth=0.95)
    plt.xlabel("Total env steps")
    plt.savefig(f"outputs/trial_{load_weights_file.replace('/', '_')}.svg")

storm_dir_contents = os.listdir(os.getcwd() + "/outputs/storm_files")

if (input_dict["evaluation_type"] == "mdp"):
    output_name = f"mdp_of_{load_weights_file}"
    verification_property = "Rmin=?[F \"done\"]"
    if (output_name + ".tra" not in storm_dir_contents
            or output_name + ".lab" not in storm_dir_contents
            or output_name + ".transrew" not in storm_dir_contents):
        print("Generating MDP file")
        generate_mdp_file(os.getcwd() + "/inputs/" + load_weights_file, env, mdpt, output_name)
    else:
        print(f"Found {output_name} files in outputs/storm_files. Will not generate a new one.")
elif (input_dict["evaluation_type"] == "dtmc"):
    output_name = f"dtmc_of_{load_weights_file}"
    verification_property = "Rmax=?[F \"done\"]"
    if (output_name + ".tra" not in storm_dir_contents
            or output_name + ".lab" not in storm_dir_contents
            or output_name + ".transrew" not in storm_dir_contents):
        print("Generating DTMC file")
        generate_dtmc_file(os.getcwd() + "/inputs/" + load_weights_file, env, mdpt, output_name)
    else:
        print(f"Found {output_name} files in outputs/storm_files. Will not generation a new one.")

print("Running STORM")
subprocess.run(["storm",
                "--explicit",
                f"outputs/storm_files/{output_name}.tra",
                f"outputs/storm_files/{output_name}.lab",
                "--transrew",
                f"outputs/storm_files/{output_name}.transrew",
                "--prop",
                verification_property])

sys.exit(0)
