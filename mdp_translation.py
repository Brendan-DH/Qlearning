#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:29:40 2024

@author: brendandevlin-hill
"""


import system_logic.probabilstic_completion as mdpt
import gymnasium as gym
from queue import Queue
import torch
import DQN
import os
import numpy as np
from dtmc_checker import CheckDTMC
import system_logic.hybrid_system as mdpt



#%% 

# define the initial state
env_size = 12

starting_parameters = DQN.system_parameters(
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

# create the environment
env_to_use = "Tokamak-v13"
env = gym.make(env_to_use,
               system_parameters=starting_parameters,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               training=True)

# load the DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_weights_name = "large_peaked_weights"

n_actions = env.action_space.n
state, info = env.reset()
initial_state = state.copy()
n_observations = len(state)

policy_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
try:
    assert saved_weights_name
    print(f"Loading policy from '/outputs/{saved_weights_name}'")
    policy_net.load_state_dict(torch.load(os.getcwd() + "/outputs/" + saved_weights_name))
except NameError:
    print("No saved weights defined, starting from scratch")

#%%

new_id = 0  # an unencountered state will get this id, after which it will be incremented

states_id_dict = {str(initial_state) : 0}
labels_set = set(["0 init\n"])

exploration_queue = Queue()

new_id += 1
transitions_array = []
states_array = []  # array of state dicts
labels_array = []  # array of label strings "[state number] [label name]"
rewards_array = []

# ask the DQN what action should be taken here
init_state, info = env.reset()
exploration_queue.put(init_state)
states_array.append(init_state)

while(not exploration_queue.empty()):

    state = exploration_queue.get().copy()
    state_tensor = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

    action = policy_net.forward(state_tensor).max(1)[1].view(1, 1).item()

    # based on the action chosen, apply the correct effect function
    # yes, this is very messy and restricts the functionality to the same action space
    # (i.e. number of robots). could possibly be made dynamic later.
    if(action == 0):
        result = r0_ccw(env, state)
    elif(action == 1):
        result = r0_cw(env,state)
    elif(action == 2):
        result = r0_inspect(env, state)
    elif(action == 3):
        result = r1_ccw(env, state)
    elif(action == 4):
        result = r1_cw(env,state)
    elif(action == 5):
        result = r1_inspect(env, state)
    elif(action == 6):
        result = r2_ccw(env, state)
    elif(action == 7):
        result = r2_cw(env,state)
    elif(action == 8):
        result = r2_inspect(env, state)

    # check if all goals are done, and if so mark this state as a 'done' state
    all_done = True
    for i in range(env.num_goals):
        # iterate over goals in state
        if(state[f"goal{i} checked"] and not state[f"goal{i} instantiated"]):
            pass
        else:
            all_done = False
            break

    # handle clock tick rewards
    clock_value = 0  # how many robots have ticked
    for i in range(env.num_active):
        clock_value += state[f"robot{i} clock"]

    # handle end states
    if (all_done):
        labels_set.add(f"{states_id_dict[str(state)]} done\n")
        # end states loop to themselves (formality):
        # transitions_array.append(f"{states_id_dict[str(state)]} {states_id_dict[str(state)]} 1")

    # iterate over result states:
    result_list = list(result.items())
    for i in range(len(result_list)):

        # add states to states dictionary with IDs if needed
        prob = result_list[i][0]
        result_state = result_list[i][1]
        if (str(result_state) not in list(states_id_dict.keys())):  # a newly discovered state
            states_id_dict[str(result_state)] = new_id
            states_array.append(result_state)
            exploration_queue.put(result_state)
            new_id += 1

        # assign clock tick rewards
        if (clock_value == env.num_active - 1):
            new_clock_value = 0
            for i in range(env.num_active):
                new_clock_value += result_state[f"robot{i} clock"]
            if(new_clock_value == 0):
                rewards_array.append(f"{states_id_dict[str(state)]} {states_id_dict[str(result_state)]} 1")

        # write the transitions into the file/array
        transitions_array.append(f"{states_id_dict[str(state)]} {states_id_dict[str(result_state)]} {prob}")

f = open(os.getcwd() + "/outputs/dtmc.tra", "w")
f.write("dtmc\n")
for i in range(len(transitions_array)):
    f.write(transitions_array[i] + "\n")
f.close()

f = open(os.getcwd() + "/outputs/dtmc.lab", "w")
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

f = open(os.getcwd() + "/outputs/dtmc.transrew", "w")
for i in range(len(rewards_array)):
    f.write(rewards_array[i] + "\n")
f.close()

p_problem_states, unacknowledged_states = CheckDTMC(os.getcwd() + "/outputs/dtmc.tra")

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
