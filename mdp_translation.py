#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:29:40 2024

@author: brendandevlin-hill
"""


import gymnasium as gym
from queue import Queue
import torch
import DQN
import os
import numpy as np


def template_move(env, state, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    # the effect function for moving robot 1 ccw
    new_state = state.copy()
    robot_no = int(np.floor(action_no / env.num_actions))
    current_location = new_state[f"robot{robot_no} location"]
    rel_action = action_no % env.num_actions

    # deterministic part of the result:
    if(rel_action == 0):
        # counter-clockwise
        if (current_location < env.size - 1):
            new_state[f"robot{robot_no} location"] = current_location + 1
        if (current_location == env.size - 1):  # cycle round
            new_state[f"robot{robot_no} location"] = 0
    elif(rel_action == 1):
        # clockwise
        if (current_location > 0):
            new_state[f"robot{robot_no} location"] = current_location - 1
        if (current_location == 0):  # cycle round
            new_state[f"robot{robot_no} location"] = env.size - 1
    else:
        raise ValueError("Error: invalid action number for movement effect function!")

    new_state = clock_effect(env, new_state, robot_no)  # advance clocks

    ef_dict = discovery_effect(env, new_state, robot_no)

    return ef_dict


def template_inspect(env, state, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    robot_no = int(np.floor(action_no / env.num_actions))

    new_state = state.copy()

    for i in range(env.num_goals):
        # goal_loc = state[f"goal{i} location"]
        # robot_loc = state[f"robot{robot_no} location"]
        # print(f"goal num: {i}, goal loc: {goal_loc }, robot no: {robot_no}, robot loc: {robot_loc}")
        if (state[f"goal{i} location"] == state[f"robot{robot_no} location"] and state[f"goal{i} instantiated"] == 1):
            new_state[f"goal{i} instantiated"] = 0

    new_state = clock_effect(env, new_state, robot_no)  # at this point, goals have been inspected

    ef_dict = discovery_effect(env, new_state, robot_no)

    return ef_dict


def discovery_effect(env, state, robot_no):
    """
    Deals with discovery of goals.
    The discovery of goals is, at present, the only probabilistic property of the
    system. Therefore this takes the state as an input and returns a ef_dict --
    a description of the possible states to come out of the input state/action pair

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        robot_no - number of the robot that is moving

    Outputs:

        ef_dict - dict of probabilities labelled with resultant states


    """
    new_state = state.copy()
    ef_dict = {}

    # deal with the discovery of goals in this location:
    # if there is a goal here, get the index (i.e. which goal it is)
    goal_index = -1
    for i, value in enumerate(env.goal_locations):
        if value == new_state[f"robot{robot_no} location"]:
            goal_index = i
            break

    if (goal_index == -1):  # no goals here; return the original state dict
        return {"1": new_state}

    # if there is a goal here, has it already been checked?
    if (state[f"goal{goal_index} checked"] == 1):
        ef_dict = {"1" : new_state}
        return ef_dict

    # if a goal needs to be revealed:
    else:
        # this goal is now checked
        new_state[f"goal{goal_index} checked"] = 1

        # this goal exists
        prob1 = env.goal_probabilities[goal_index]
        state1 = new_state.copy()
        state1[f"goal{goal_index} instantiated"] = 1
        ef_dict[str(prob1)] = state1

        # this goal does not exist
        prob2 = round(1 - env.goal_probabilities[goal_index], 5)  # this rounding may cause problems!!!!!
        state2 = new_state.copy()
        state2[f"goal{goal_index} instantiated"] = 0
        ef_dict[str(prob2)] = state2

    return ef_dict


def clock_effect(env, state, robot_no):
    """

    Deals with the clock variables after an action has taken place

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:

        new_state - the state dict after the clocks have been advanced

    """

    new_state = state.copy()
    new_state[f"robot{robot_no} clock"] = 1

    for i in range(env.num_active):
        if new_state[f"robot{i} clock"] == 0:
            return new_state  # if any clocks are not ticked, return

    # else if all clocks are ticked:
    for i in range(env.num_active):
        new_state[f"robot{i} clock"] = 0  # set all clocks to 0

    return new_state.copy()


# then these functions are specific effect functions for certain robots moving in certain directions:
def r0_ccw(env, state):
    return template_move(env, state, 0)


def r0_cw(env, state):
    return template_move(env, state, 1)


def r0_inspect(env, state):
    return template_inspect(env, state, 2)


def r1_ccw(env, state):
    return template_move(env, state, 3)


def r1_cw(env, state):
    return template_move(env, state, 4)


def r1_inspect(env, state):
    return template_inspect(env, state, 5)


def r2_ccw(env, state):
    return template_move(env, state, 6)


def r2_cw(env, state):
    return template_move(env, state, 7)


def r2_inspect(env, state):
    return template_inspect(env, state, 8)


#%%

# define the initial state
starting_parameters = DQN.system_parameters(
    size=12,
    robot_status=[1,1,1],
    robot_locations=[1,5,6],
    breakage_probability=0.0001,
    goal_locations=[11,5,2,10,9,8],
    goal_probabilities=[0.49, 0.9, 0.7, 0.7, 0.4, 0.7],
    goal_instantiations=[0,1,1,1,0,0],
    goal_resolutions=[0,1,1,1,0,1],
    goal_checked=[1,1,1,1,1,1,0],
    port_locations=[0,11],
    elapsed=0,
)


# create the environment
env_to_use = "Tokamak-v8"
env = gym.make(env_to_use,
               system_parameters=starting_parameters,
               training=True,
               render_mode=None)


# load the DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_weights_name = "saved_weights_182634"

n_actions = env.action_space.n
state, info = env.reset()
initial_state = state.copy()
n_observations = len(state)

policy_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
try:
    assert saved_weights_name
    print(f"Loading from '/outputs/{saved_weights_name}'")
    policy_net.load_state_dict(torch.load(os.getcwd() + "/outputs/" + saved_weights_name))
except NameError:
    print("No saved weights defined, starting from scratch")


#%%


new_id = 0  # an unencountered state will get this id, after which it will be incremented

states_id_dict = {str(initial_state) : 0}

exploration_queue = Queue()

new_id += 1
transitions_array = []
states_array = []

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

    # now we have state, probabilities, resultant states, register resultant states:

    # iterate over result states:
    result_list = list(result.items())
    for i in range(len(result_list)):

        new_state_unexplored = False

        # add states to states dictionary with IDs if needed
        prob = result_list[i][0]
        result_state = result_list[i][1]
        if (str(result_state) not in list(states_id_dict.keys())):
            states_id_dict[str(result_state)] = new_id
            states_array.append(result_state)
            exploration_queue.put(result_state)
            new_id += 1

        # write the transitions into the file/array
        transitions_array.append(f"{states_id_dict[str(state)]} {states_id_dict[str(result_state)]} {prob}")

f = open("dtmc.tra", "w")
f.write("dtmc\n")
for i in range(len(transitions_array)):
    f.write(transitions_array[i] + "\n")
f.close()
