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
from dtmc_checker import CheckDTMC


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

    p, s = discovery_effect(env, new_state, robot_no)

    return p, s


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
    print("inspect", robot_no)

    for i in range(env.num_goals):
        if (state[f"goal{i} location"] == state[f"robot{robot_no} location"] and state[f"goal{i} instantiated"] == 1):
            new_state[f"goal{i} instantiated"] = 0

    new_state = clock_effect(env, new_state, robot_no)  # at this point, goals have been inspected

    p, s = discovery_effect(env, new_state, robot_no)

    return p, s


def discovery_effect(env, state, robot_no):
    """
    Deals with discovery of goals.
    The discovery of goals is, at present, the only probabilistic property of the
    system. Therefore this takes the state as an input and returns p_array and s_array  --
    a description of the possible states to come out of the input state/action pair

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        robot_no - number of the robot that is moving

    Outputs:

        p_array - array of probabilities for resultant states
        s_array - array of resultant states


    """
    new_state = state.copy()
    p_array = []
    s_array = []

    # deal with the discovery of goals in this location:
    # if there is a goal here, get the index (i.e. which goal it is)
    goal_index = -1
    for i in range(env.num_goals):
        loc = state[f"goal{i} location"]
        if loc == new_state[f"robot{robot_no} location"]:
            goal_index = i
            break

    if (goal_index == -1):  # no goals here; return the original state dict
        p_array = [1]
        s_array = [new_state]
        return p_array, s_array

    # if there is a goal here, has it already been checked?
    if (state[f"goal{goal_index} checked"] == 1):
        p_array = [1]
        s_array = [new_state]
        return p_array, s_array

    # if a goal needs to be revealed:
    else:
        # this goal is now checked
        new_state[f"goal{goal_index} checked"] = 1

        # this goal exists
        prob1 = state[f"goal{goal_index} probability"]
        state1 = new_state.copy()
        state1[f"goal{goal_index} instantiated"] = 1
        p_array.append(prob1)
        s_array.append(state1)

        # this goal does not exist
        prob2 = round(1 - state[f"goal{goal_index} probability"], 5)  # this rounding may cause problems!!!!!
        state2 = new_state.copy()
        state2[f"goal{goal_index} instantiated"] = 0
        p_array.append(prob2)
        s_array.append(state2)

    return p_array, s_array


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

    for i in range(env.num_robots):
        if new_state[f"robot{i} clock"] == 0:
            return new_state  # if any clocks are not ticked, return

    # else if all clocks are ticked:
    for i in range(env.num_robots):
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


def t_model(env, state, action_no):
    """
    Get complete PDF of possible resultant states
    """

    # based on the action_no chosen, apply the correct effect function
    # yes, this is very messy and restricts the functionality to the same action_no space
    # (i.e. number of robots). could possibly be made dynamic later.

    if(env.blocked_model(env, state)[action_no] == 1):
        if(action_no == 2 or action_no == 5 or action_no == 8):
            print("blocked:", action_no, [state[f"robot{i} clock"] for i in range(env.num_robots)],
                  [state[f"goal{i} instantiated"] for i in range(env.num_goals)],
                  [state[f"robot{i} location"] for i in range(env.num_robots)])
        robot_no = int(np.floor(action_no / env.num_actions))
        new_state = clock_effect(env, state, robot_no)
        p = [1]
        s = [new_state]
        return p, s

    if(action_no == 0):
        p, s = r0_ccw(env, state)
    elif(action_no == 1):
        p, s = r0_cw(env,state)
    elif(action_no == 2):
        p, s = r0_inspect(env, state)
    elif(action_no == 3):
        p, s = r1_ccw(env, state)
    elif(action_no == 4):
        p, s = r1_cw(env,state)
    elif(action_no == 5):
        p, s = r1_inspect(env, state)
    elif(action_no == 6):
        p, s = r2_ccw(env, state)
    elif(action_no == 7):
        p, s = r2_cw(env,state)
    elif(action_no == 8):
        p, s = r2_inspect(env, state)

    return p, s


def r_model(env, s, action, sprime):

    reward = 0

    # rewards for blocked actions
    # the reward function doesn't strictly rely on the action taken,
    # but this is an easy way to see if it was a 'blocked' transition
    if env.blocked_model(env, s)[action]:
        reward += -0.1
        return reward

    # rel_action = action % env.num_actions  # 0=counter-clockwise, 1=clockwise, 2=engage

    # rewards for discovering goals - this only happens after a robot has moved
    for i in range(env.num_goals):
        # check if any goals have become checked
        if(s[f"goal{i} checked"] != sprime[f"goal{i} checked"]):
            return 100

    # rewards for completing goals
    for i in range(env.num_goals):
        if(s[f"goal{i} instantiated"] == 1 and sprime[f"goal{i} instantiated"] == 0):
            print("bingo")
            return 1000  # - (s["elapsed"] * env.num_robots) / 100 * 500

    return 0


def b_model(env, state):

    blocked_actions = np.zeros(env.action_space.n)

    for i in range(env.num_robots):
        moving_robot_loc = state[f"robot{i} location"]
        if (state[f"robot{i} clock"]):
            blocked_actions[i * env.num_actions:(i * env.num_actions) + env.num_actions] = 1
        else:
            for j in range(env.num_robots):
                other_robot_loc = state[f"robot{j} location"]

                if (i == j):
                    continue  # don't check robots against themselves

                if(moving_robot_loc == other_robot_loc):
                    raise ValueError(f"Two robots occupy the same location (r{i} & r{j} @ {moving_robot_loc}).")

            blocked_actions[(i * env.num_actions)] = get_counter_cw_blocked(env, state, i)
            blocked_actions[(i * env.num_actions) + 1] = get_cw_blocked(env, state, i)

            block_inspection = 1
            for k in range(env.num_goals):
                if (state[f"goal{k} location"] == moving_robot_loc and state[f"goal{k} instantiated"] == 1):
                    block_inspection = 0  # unblock this engage action
                    break
            blocked_actions[(i * env.num_actions) + 2] = block_inspection

    return blocked_actions

#%%


def get_counter_cw_blocked(env, state, robot_no):

    moving_robot_loc = state[f"robot{robot_no} location"]

    for j in range(env.num_robots):
        other_robot_loc = state[f"robot{j} location"]
        if(robot_no == j):  # don't need to check robots against themselves
            continue
        if (moving_robot_loc == other_robot_loc):
            raise ValueError(f"Two robots occupy the same location (r{robot_no} & r{j} @ {moving_robot_loc}).")
        if(other_robot_loc == (moving_robot_loc + 1) % env.size):
            return True
    return False


def get_cw_blocked(env, state, robot_no):

    moving_robot_loc = state[f"robot{robot_no} location"]

    for j in range(env.num_robots):
        other_robot_loc = state[f"robot{j} location"]
        if(robot_no == j):  # don't need to check robots against themselves
            continue
        if (moving_robot_loc == other_robot_loc):
            raise ValueError(f"Two robots occupy the same location (r{robot_no} & r{j} @ {moving_robot_loc}).")
        if(other_robot_loc == (env.size - 1 if moving_robot_loc - 1 < 0 else moving_robot_loc - 1)):
            return True

    return False


#%%
if (__name__ == "__main__"):
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
