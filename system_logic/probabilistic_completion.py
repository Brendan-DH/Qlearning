#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:40:11 2024

@author: brendandevlin-hill


System version: probabilistic completion
    - when a robot attempts to complete a task, it has a chance of failing or
    succeeding


This file defines the transition model of the environment; the logic herein
defines how the environment evolves under a state/action map

The logic is used both by the gymnasium environments (tokamakenv10) and
the mdp translation functionaltiy.


"""

import numpy as np

#%%

"""
This block defines abstractaction functions that can be used to define the
physical actions of the system
"""


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
robot_no = int(torch.floor_divide(action_no, env.num_actions).item())    current_location = new_state[f"robot{robot_no} location"]
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

    p_array = [1]
    s_array = [new_state]

    return p_array, s_array


def template_inspect(env, state, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

robot_no = int(torch.floor_divide(action_no, env.num_actions).item())
    new_state = state.copy()

    for i in range(env.num_goals):
        if (state[f"goal{i} location"] == state[f"robot{robot_no} location"]):

            s_array = []
            p_array = []

            prob1 = state[f"goal{i} probability"]
            state1 = new_state.copy()
            state1[f"goal{i} active"] = 0
            p_array.append(prob1)
            state1 = clock_effect(env, state1, robot_no)  # at this point, goals have been inspected
            s_array.append(state1)

            # this goal does not exist
            prob2 = round(1 - state[f"goal{i} probability"], 5)  # this rounding may cause problems!!!!!
            state2 = new_state.copy()
            state2[f"goal{i} active"] = 1
            p_array.append(prob2)
            state2 = clock_effect(env, state2, robot_no)  # at this point, goals have been inspected
            s_array.append(state2)

    state1 = clock_effect(env, state1, robot_no)  # at this point, goals have been inspected

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
    # print(f"clock: {robot_no}")

    for i in range(env.num_robots):
        if new_state[f"robot{i} clock"] == 0:
            return new_state.copy()  # if any clocks are not ticked, return

    # else if all clocks are ticked:
    for i in range(env.num_robots):
        new_state[f"robot{i} clock"] = 0  # set all clocks to 0
        # new_state["elapsed ticks"] += 1

    return new_state.copy()


#%%


"""
This block defines the transition model using the above abstract templates
"""


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
    robot_no = int(torch.floor_divide(action_no, env.num_actions).item())        new_state = clock_effect(env, state, robot_no)
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

    # print(str(s[0].values()))

    return p, s

#%%


"""
This block defines the rewards model of the system
"""


def r_model(env, s, action, sprime):

    reward = 0

    # rewards for blocked actions
    # the reward function doesn't strictly rely on the action taken,
    # but this is an easy way to see if it was a 'blocked' transition
    if env.blocked_model(env, s)[action]:
        reward += -0.1
        return reward

    rel_action = action % env.num_actions  # 0=counter-clockwise, 1=clockwise, 2=engage
    if(rel_action == 2):
        reward += 100

    # # rewards for completing goals
    for i in range(env.num_goals):
        # check if any goals have become checked
        if(s[f"goal{i} active"] != sprime[f"goal{i} active"]):
            reward += 1000

    return reward

#%%


"""
This block defines which actions are blocked in each state
Essentially, this is an auxiliary part of the transition model
"""


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

            block_inspection = True
            for k in range(env.num_goals):
                if (state[f"goal{k} location"] == moving_robot_loc and state[f"goal{k} active"] == 1):
                    block_inspection = False  # unblock this engage action
                    break
            blocked_actions[(i * env.num_actions) + 2] = block_inspection

    return blocked_actions


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
