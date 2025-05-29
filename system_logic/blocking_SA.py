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
import math
import sys
import numpy as np
import torch

global_device = "cpu"
# %%

"""
This block defines abstractaction functions that can be used to define the
physical actions of the system
"""


def template_move(env, state_dict, robot_no, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    state_dict = state_dict.copy()
    current_location = state_dict[f"robot{robot_no} location"]

    # deterministic part of the result:
    if (action_no == 0):
        # counter-clockwise
        if (current_location < env.unwrapped.size - 1):
            state_dict[f"robot{robot_no} location"] = current_location + 1
        elif (current_location == env.unwrapped.size - 1):  # cycle round
            state_dict[f"robot{robot_no} location"] = 0
    elif (action_no == 1):
        # clockwise
        if (current_location > 0):
            state_dict[f"robot{robot_no} location"] = current_location - 1
        elif (current_location == 0):  # cycle round
            state_dict[f"robot{robot_no} location"] = env.unwrapped.size - 1
    else:
        raise ValueError("Error: invalid action number for movement effect function!")

    p_array, s_array = discovery_effect(env, state_dict)  # get resultant states and probabilities

    return p_array, s_array


def template_complete(env, state_dict, robot_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    state_dict = state_dict.copy()

    num_robots_present = np.sum([state_dict[f"robot{i} location"] == state_dict[f"robot{robot_no} location"] for i in range(env.unwrapped.num_robots)])

    for i in range(env.unwrapped.num_goals):
        # if (state_dict[f"goal{i} location"] == state_dict[f"robot{robot_no} location"]):
        if (state_dict[f"goal{i} location"] == state_dict[f"robot{robot_no} location"]):
            # goal is successfully completed
            prob1 = state_dict[f"goal{i} completion probability"]
            state1 = state_dict.copy()
            state1[f"goal{i} active"] = 0

            # failure to complete goal
            prob2 = 1 - prob1
            state2 = state_dict.copy()
            state2[f"goal{i} active"] = 1

            if (num_robots_present == 2):
                prob1 = prob1 ** (0.33)
                prob2 = 1 - prob1

            return ([prob1, prob2], [state1, state2])

    return [], []


def template_wait(env, state_dict, action_no):
    """
    Allows a robot to wait a tick without doing anything.
    """

    new_state = state_dict.copy()

    return ([1], [new_state])


def discovery_effect(env, state_dict):
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
    state_dict = state_dict.copy()
    robot_no = state_dict["clock"]

    # return ([1], [state_dict])

    # deal with the discovery of goals in this location:
    # if there is a goal here, get the index (i.e. which goal it is)
    goal_index = -1
    for i in range(env.unwrapped.num_goals):
        gloc = state_dict[f"goal{i} location"]
        rloc = state_dict[f"robot{robot_no} location"]  # made a change here were an extra loop over num_robots appeared to do nothing
        if (gloc == rloc):  # this requires a CPU transfer so is expensive
            goal_index = i
            break

    if (goal_index == -1 or state_dict[f"goal{goal_index} checked"] == 1 or state_dict[f"goal{goal_index} active"] == 1):  # no goals here; return the original state dict
        p_array = [1]
        s_array = [state_dict]
        return p_array, s_array  # is this correct?

    # if a goal needs to be revealed:
    else:
        # goal becomes 'checked'
        state_dict[f"goal{goal_index} checked"] = 1

        # a goal was discovered
        prob1 = state_dict[f"goal{goal_index} discovery probability"]
        state1 = state_dict.copy()
        state1[f"goal{goal_index} active"] = 1

        # no goal was found here
        prob2 = 1 - prob1
        state2 = state_dict.copy()
        state2[f"goal{goal_index} active"] = 0

        return ([prob1, prob2], [state1, state2])


# %%

def t_model(env, state_dict, robot_no, action_no):
    """
    Get complete PDF of possible resultant states
    """

    # based on the action_no chosen, apply the correct effect function
    # yes, this is very messy and restricts the functionality to the same action_no space
    # (i.e. number of robots). could possibly be made dynamic later.

    if (b_model(env, state_dict, state_dict["clock"])[action_no] == 1):
        return [1], [state_dict]

    state_dict = state_dict.copy()

    # use the appropriate function to get the probability and state array for each possible action type:
    if (action_no == 0 or action_no == 1):
        p, s = template_move(env, state_dict, robot_no, action_no)
    elif (action_no == 2):
        p, s = template_complete(env, state_dict, robot_no)
    elif (action_no == 3):
        p, s = template_wait(env, state_dict, robot_no)

    for state in s:
        state["clock"] = (state["clock"] + 1) % env.unwrapped.num_robots

    return p, s


def initial_state_logic(env, state_dict):
    state = state_dict.copy()

    for i in range(env.unwrapped.num_robots):
        rloc = state_dict[f"robot{i} location"]  # made a change here were an extra loop over num_robots appeared to do nothing
        for j in range(env.unwrapped.num_goals):
            gloc = state_dict[f"goal{j} location"]
            if rloc == gloc:
                gprob = state_dict[f"goal{j} discovery probability"]
                roll = np.random.random()
                if (roll < gprob):
                    state[f"goal{j} active"] = 1
                state[f"goal{j} checked"] = 1

    return state


# %%

"""
This block defines the rewards model of the system
"""


def r_model(env, old_state_dict, action_no, next_state_dict):
    # rewards for blocked actions
    # this is necessary to stop the total rewards shooting up when blocked actions are taken
    # mostly a diagnostic thing... I think
    
    robot_no = old_state_dict["clock"]

    if (b_model(env, old_state_dict, robot_no)[action_no] == 1):
        return 0

    reward = 0

    if (action_no == 0 or action_no == 1):
        # penalty for being in another robot's way
        moving_robot_loc = next_state_dict[f"robot{robot_no} location"]
        for i in range(env.unwrapped.num_robots):
            other_robot_loc = next_state_dict[f"robot{i} location"]
            if moving_robot_loc - other_robot_loc == 1 or \
                    moving_robot_loc - other_robot_loc == -1 or \
                    (moving_robot_loc == 0 and other_robot_loc == env.unwrapped.size - 1) or \
                    (moving_robot_loc == env.unwrapped.size - 1 and other_robot_loc == 0):
                reward -= 0.5
                break
            else:
                reward += 0.05
        
        
                
        # reward for checking a goal by moving onto its position
        for i in range(env.unwrapped.num_goals):
            if (old_state_dict[f"goal{i} checked"] != next_state_dict[f"goal{i} checked"]):
                reward += 0.2

    # rewards for attempting goals - more for harder goals
    if (action_no == 2):
        for i in range(env.unwrapped.num_goals):
            prob = old_state_dict[f"goal{i} completion probability"]
            reward += 0.2  + 0.4*(1 - 1 / (1 + np.exp(-7 * (prob - 0.5))))

    # reward for staying still if there are few goals remaining
    if (action_no == 3):
        num_goals_active = np.sum([next_state_dict[f"goal{i} active"] for i in range(env.unwrapped.num_goals)])
        num_goals_unchecked = np.sum([0 if next_state_dict[f"goal{i} checked"] else 1 for i in range(env.unwrapped.num_goals)])
        if (num_goals_active < robot_no and num_goals_unchecked < robot_no):
            return 0.05
            
    # reward for having moved into a terminal state
    if (state_is_final(env, next_state_dict)):
        reward += 1

    return reward

def logit(x):
    """
    Logit function for scaling probabilities
    """
    return math.log(x / (1 - x))

# %%

def pseudoreward_function(env, state_dict):
    # defining a pseudoreward function that roughly describes the proximity to the `completed' state
    pr = env.unwrapped.size * env.unwrapped.num_robots * 2  # initialising to a high value
    for i in range(env.unwrapped.num_robots):
        rob_position = state_dict[f"robot{i} location"]
        goal_min_mod_dist = env.unwrapped.size + 1  # store the mod distance to closest goal
        # rob_min_mod_dist = env.unwrapped.size + 1  # store the mod distance to closest robot

        # for j in range(env.unwrapped.num_robots):
        #     if i == j:
        #         continue
        #     other_robot_pos = state_dict[f"robot{j} location"]
        #     naive_dist = abs(rob_position - other_robot_pos)  # non-mod distance
        #     rob_mod_dist = min(naive_dist, env.unwrapped.size - naive_dist)  # to account for cyclical space
        #     rob_min_mod_dist = min(rob_min_mod_dist, rob_mod_dist)  # update the smaller of the two

        # pr += 0.2 * rob_min_mod_dist  # give a small bonus for being farther away from nearest robot

        for j in range(env.unwrapped.num_goals):
            goal_active = state_dict[f"goal{j} active"]
            goal_checked = state_dict[f"goal{j} checked"]
            if (goal_active == 0 and goal_checked == 1):
                pr += env.unwrapped.size + 2  # bonus for completing a goal; ensures PR always increases when goals completed
            else:
                goal_position = state_dict[f"goal{j} location"]
                naive_dist = abs(rob_position - goal_position)  # non-mod distance
                goal_mod_dist = min(naive_dist, env.unwrapped.size - naive_dist)  # to account for cyclical space
                goal_min_mod_dist = min(goal_mod_dist, goal_min_mod_dist)  # update the smaller of the two

        pr -= goal_min_mod_dist  # subtract the distance 'penalty' from total possible reward

    return pr * 0.1


"""
This block defines which actions are blocked in each state
Essentially, this is an auxiliary part of the transition model
"""


def b_model(env, state_dict, robot_no):
    blocked_actions = np.zeros(env.action_space.n)
    active_robot_loc = state_dict[f"robot{robot_no} location"]

    blocked_actions[0] = get_counter_cw_blocked(env, state_dict, robot_no)
    blocked_actions[1] = get_cw_blocked(env, state_dict, robot_no)

    block_task_completion = True
    other_robot_adjacent = False

    for m in range(env.unwrapped.num_robots):
        if robot_no == m:
            continue
        elif abs(min(active_robot_loc - state_dict[f"robot{m} location"], env.unwrapped.size - (active_robot_loc - state_dict[f"robot{m} location"]))) == 1:
            other_robot_adjacent = True
            break

    if not other_robot_adjacent:
        for k in range(env.unwrapped.num_goals):
            if (state_dict[f"goal{k} location"] == active_robot_loc and state_dict[f"goal{k} active"] == 1):
                block_task_completion = False  # unblock this engage action
                break

    blocked_actions[2] = block_task_completion

    # keeping this as a tensor as it makes some masking easier
    return torch.tensor(blocked_actions, dtype=torch.bool, device=global_device, requires_grad=False)


def get_counter_cw_blocked(env, state_dict, robot_no):
    moving_robot_loc = state_dict[f"robot{robot_no} location"]
    n_robots_on_new_space = 0

    for j in range(env.unwrapped.num_robots):
        other_robot_loc = state_dict[f"robot{j} location"]
        if (robot_no == j):  # don't need to check robots against themselves
            continue
        if (other_robot_loc == (moving_robot_loc + 1) % env.unwrapped.size):
            n_robots_on_new_space += 1
            
    return n_robots_on_new_space >= 1


def get_cw_blocked(env, state_dict, robot_no):
    moving_robot_loc = state_dict[f"robot{robot_no} location"]
    n_robots_on_new_space = 0

    for j in range(env.unwrapped.num_robots):
        other_robot_loc = state_dict[f"robot{j} location"]
        if (robot_no == j):  # don't need to check robots against themselves
            continue
        if (other_robot_loc == (env.unwrapped.size - 1 if moving_robot_loc - 1 < 0 else moving_robot_loc - 1)):
            n_robots_on_new_space += 1

    return n_robots_on_new_space >= 1


def state_is_final(env, state_dict):
    for i in range(env.unwrapped.num_goals):
        # iterate over goals in state
        # if (state_dict[f"goal{i} checked"] == 0 or (state_dict[f"goal{i} checked"] == 1 and state_dict[f"goal{i} active"] == 1)):
        #     return False
        if (state_dict[f"goal{i} checked"] == 0 or (
                state_dict[f"goal{i} checked"] == 1 and state_dict[f"goal{i} active"] == 1)):
            return False

    return True
