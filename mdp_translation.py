#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:29:40 2024

@author: brendandevlin-hill
"""
import sys
# import gymnasium as gym
from queue import Queue
import torch
import DQN
import os
import numpy as np
from dtmc_checker import CheckDTMC


def GenerateDTMCFile(saved_weights_file, env, system_logic, output_name="dtmc"):

    # load the DQN

    n_actions = env.action_space.n
    state_tensor, info = env.reset()
    initial_state_tensor = state_tensor.detach().clone()
    initial_state_dict = env.interpret_state_tensor(initial_state_tensor)
    n_observations = len(state_tensor)

    if(not saved_weights_file):
        print("No weights file specified, exiting.")
        sys.exit(1)

    try:
        loaded_weights = torch.load(saved_weights_file)
        nodes_per_layer = len(loaded_weights["layer1.weight"])
        num_hidden_layers = len(loaded_weights["hidden_layers"])  # not sure if this will work
        print(f"Loading policy from '{saved_weights_file}'")
    except FileNotFoundError:
        print(f"Weights file {saved_weights_file} not found, exiting.")
        sys.exit(1)

    policy_net = DQN.DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer)
    policy_net.load_state_dict(loaded_weights)

    """Create the explicit DTMC representation"""

    new_id = 0  # an unencountered state will get this id, after which it will be incremented

    states_id_dict = {str(initial_state_dict) : 0}  # dictionary of state dicts to id
    labels_set = {"0 init\n"}  # set of state labels ([id] [label] )

    exploration_queue = Queue()

    new_id += 1
    transitions_array = []
    # states_array = []  # array of state dicts
    # labels_array = []  # array of label strings "[state number] [label name]"
    rewards_array = []

    # ask the DQN what action should be taken here
    init_state, info = env.reset()
    exploration_queue.put(init_state)
    # states_array.append(init_state)

    while(not exploration_queue.empty()):

        # print("\r-----------------------------------------------------------------", end="")
        print(f"\rStates in exploration queue: {' ' * ( 10 - len(str(exploration_queue.qsize())))}{exploration_queue.qsize()}", end="")

        state_tensor = exploration_queue.get().detach().clone()  # what is the differnce between state and stateT
        state_dict = env.interpret_state_tensor(state_tensor)

        action_utilities = policy_net.forward(state_tensor.unsqueeze(0))[0]  # why is this indexed?
        blocked = env.blocked_model(env, state_tensor)
        action_utilities = torch.where(blocked, -1000, action_utilities)
        action = torch.argmax(action_utilities).item()

        # get the result of the action from the transition model
        result = system_logic.t_model(env, state_tensor, action)

        # label end states
        all_done = system_logic.state_is_final(env, state_tensor)
        if (all_done):
            labels_set.add(f"{states_id_dict[str(state_dict)]} done\n")  # label end states
            # end states loop to themselves (formality):
            transitions_array.append(f"{states_id_dict[str(state_dict)]} {states_id_dict[str(state_dict)]} 1")
            continue  # continue as we don't care about other transitions from end states

        # iterate over result states:
        for i in range(len(result[0])):

            prob = result[0][i]
            result_state_tensor = result[1][i]
            result_state_dict = env.interpret_state_tensor(result_state_tensor)

            # print(prob)

            # register newly discovered states
            if (str(result_state_dict) not in list(states_id_dict.keys())):
                states_id_dict[str(result_state_dict)] = new_id
                exploration_queue.put(result_state_tensor)
                new_id += 1

            # assign awards to clock ticks
            # all s' will lead to a clock tick if robots-1 clocks are ticked in s (provided blocked actions are impossible)
            if (np.sum([result_state_dict[f"robot{i} clock"] for i in range(env.num_robots)]) == 0):
                rewards_array.append(f"{states_id_dict[str(state_dict)]} {states_id_dict[str(result_state_dict)]} 1")

            # write the transitions into the file/array
            transitions_array.append(f"{states_id_dict[str(state_dict)]} {states_id_dict[str(result_state_dict)]} {prob}")


    """Write the DTMC file"""
    print(f"Writing file to {os.getcwd()}/{output_name}.tra, {output_name}.lab, {output_name}.transrew")

    f = open(os.getcwd() + f"/outputs/{output_name}.tra", "w")  # create DTMC file .tra
    f.write("dtmc\n")
    for i in range(len(transitions_array)):
        f.write(transitions_array[i] + "\n")
    f.close()

    f = open(os.getcwd() + f"/outputs/{output_name}.lab", "w")  # create labels file .lab
    f.write("""
    #DECLARATION
    init done
    #END
    """)
    labels_list = list(labels_set)
    labels_list.sort(key=lambda x: int(x.split()[0]))  # label file must list states in numerical order
    for i in range(len(labels_list)):
        f.write(labels_list[i])
    f.close()

    f = open(os.getcwd() + f"/outputs/{output_name}.transrew", "w")  # rewards file .transrew
    for i in range(len(rewards_array)):
        f.write(rewards_array[i] + "\n")
    f.close()

    print(f"Saved policy DTMC as {output_name}.")

    # check DTMC for invalid states
    p_problem_states, unacknowledged_states = CheckDTMC(os.getcwd() + f"/outputs/{output_name}.tra")

    if (len(p_problem_states) == 0):
        print("Success: all probabilities sum to 1")
    else:
        print("Error! Some outgoing probabilities do not sum to 1\nstate | total p")
        for i in range(len(p_problem_states)):
            print(f"{p_problem_states[i][0]} | {p_problem_states[i][1]}")

    if(len(unacknowledged_states) == 0):
        print("Success: all states included in transition structure")
    else:
        print("Error! Some encountered states have no outgoing transitions!\nStates:")
        for i in range(len(unacknowledged_states)):
            print(unacknowledged_states[i])