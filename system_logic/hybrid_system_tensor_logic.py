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

import numpy as np
import torch

global_device = "cpu"
print("Transition logic running on: ", global_device)

# %%

"""
This block defines abstractaction functions that can be used to define the
physical actions of the system
"""


def template_move(env, state_tensor, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    # the effect function for moving robot 1 ccw
    new_state = state_tensor.detach().clone()
    robot_no = int(torch.floor_divide(action_no, env.num_actions).item())
    current_location = state_tensor[robot_no * 2]
    rel_action = action_no % env.num_actions

    # deterministic part of the result:
    if (rel_action == 0):
        # counter-clockwise
        if (current_location < env.size - 1):
            # print(f"{new_state[robot_no * 2]}->{current_location + 1}")
            new_state[robot_no * 2] = current_location + 1
        elif (current_location == env.size - 1):  # cycle round
            # print(f"{new_state[robot_no * 2]}->{0}")
            new_state[robot_no * 2] = 0
    elif (rel_action == 1):
        # clockwise
        if (current_location > 0):
            # print(f"{new_state[robot_no * 2]}->{current_location - 1}")
            new_state[robot_no * 2] = current_location - 1
        elif (current_location == 0):  # cycle round
            # print(f"{new_state[robot_no * 2]}->{env.size - 1}")
            new_state[robot_no * 2] = env.size - 1
    else:
        raise ValueError("Error: invalid action number for movement effect function!")

    clock_state = clock_effect(env, new_state, robot_no)  # advance clocks
    p_tensor, s_tensor = discovery_effect(env, clock_state, robot_no)  # get resultant states and probabilities

    return p_tensor, s_tensor


def template_complete(env, state_tensor, action_no):
    """

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:
        ef_dict - dictionary describing the resultant state

    """

    robot_no = int(torch.floor_divide(action_no, env.num_actions).item())

    for i in range(env.num_goals):
        # if (state_tensor[f"goal{i} location"] == state_tensor[f"robot{robot_no} location"]):
        if (state_tensor[(env.num_robots * 2) + (i * 5)] == state_tensor[robot_no * 2]):
            p_tensor = torch.empty(2, device=global_device)
            s_tensor = torch.empty(2, device=global_device)

            # goal is successfully completed
            prob1 = state_tensor[(env.num_robots * 2) + (i * 5) + 4]
            state1 = state_tensor.detach().clone()
            state1[(env.num_robots * 2) + (i * 5) + 1] = 0
            state1 = clock_effect(env, state1, robot_no)  # at this point, goals have been inspected

            # failure to complete goal
            prob2 = round(1 - state_tensor[(env.num_robots * 2) + (i * 5) + 4].item(), 5)  # this rounding may cause problems!!!!!
            state2 = state_tensor.detach().clone()
            state2[(env.num_robots * 2) + (i * 5) + 1] = 1
            state2 = clock_effect(env, state2, robot_no)  # at this point, goals have been inspected

            return ([prob1, prob2], torch.stack([state1, state2]))

    return [], torch.empty(0)


def template_wait(env, state_tensor, action_no):
    """
    Allows a robot to wait a tick.
    """

    robot_no = int(torch.floor_divide(action_no, env.num_actions).item())
    new_state = clock_effect(env, state_tensor, robot_no)

    return (torch.tensor([1], device=global_device, dtype=torch.float32),
            new_state.to(device=global_device, dtype=torch.float32).unsqueeze(0))


def clock_effect(env, state_tensor, robot_no):
    """

    Deals with the clock variables after an action has taken place

    Inputs:
        env - gymnasium environment
        state - dictionary describing the current state
        action_no - number of the action to be executed

    Outputs:

        new_state - the state dict after the clocks have been advanced

    """

    new_state = state_tensor.detach().clone()
    new_state[(robot_no * 2) + 1] = 1
    # print(f"clock: {robot_no}")

    for i in range(env.num_robots):
        if new_state[(i * 2) + 1] == 0:
            return new_state  # if any clocks are not ticked, return

    # else if all clocks are ticked:
    for i in range(env.num_robots):
        new_state[(i * 2) + 1] = 0  # set all clocks to 0
        # new_state["elapsed ticks"] += 1

    return new_state


def discovery_effect(env, state_tensor, robot_no):
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
    new_state = state_tensor.detach().clone()

    # deal with the discovery of goals in this location:
    # if there is a goal here, get the index (i.e. which goal it is)
    goal_index = -1
    for i in range(env.num_goals):
        gloc = new_state[(env.num_robots * 2) + (i * 5)].item()
        rloc = new_state[(robot_no * 2)].item()  # made a change here were an extra loop over num_robots appeared to do nothing
        # print("locations", gloc, rloc)
        if (gloc == rloc):  # this requires a CPU transfer so is expensive
            goal_index = i
#             print(f"goal index: {i}")
            break

    if (goal_index == -1):  # no goals here; return the original state dict
        p_tensor = torch.tensor([1], device=global_device, dtype=torch.float32)
        s_tensor = new_state.to(device=global_device, dtype=torch.float32).unsqueeze(0)
#         print("no goal")
        return p_tensor, s_tensor  # is this correct?

    # if there is a goal here, has it already been checked?
    if (state_tensor[(env.num_robots * 2) + (goal_index * 5) + 2] == 1):
        p_tensor = torch.tensor([1], device=global_device, dtype=torch.float32)
        s_tensor = new_state.to(device=global_device, dtype=torch.float32).unsqueeze(0)
#         print("goal, already checked")
        return p_tensor, s_tensor

    # if a goal needs to be revealed:
    else:

#         print("revealing a goal...")

        # goal becomes 'checked'
        new_state[(env.num_robots * 2) + (goal_index * 5) + 2] = 1

        # a goal was discovered
        prob1 = state_tensor[(env.num_robots * 2) + (goal_index * 5) + 3]
        state1 = new_state.detach().clone()
        state1[(env.num_robots * 2) + (goal_index * 5) + 1] = 1

        # no goal was found here
        prob2 = round(1 - state_tensor[(env.num_robots * 2) + (goal_index * 5) + 3].item(), 5)  # this rounding may cause problems!!!!!
        state2 = new_state.detach().clone()
        state2[(env.num_robots * 2) + (goal_index * 5) + 1] = 0

        return ([prob1, prob2], torch.stack([state1, state2]))


# %%

def t_model(env, state_tensor, action_no):
    """
    Get complete PDF of possible resultant states
    """

    # based on the action_no chosen, apply the correct effect function
    # yes, this is very messy and restricts the functionality to the same action_no space
    # (i.e. number of robots). could possibly be made dynamic later.

    robot_no = int(torch.floor_divide(action_no, env.num_actions).item())

    if (env.blocked_model(env, state_tensor)[action_no] == 1):
        new_state = clock_effect(env, state_tensor, robot_no)
        p = torch.tensor([1], device=global_device, dtype=torch.float32)
        s = torch.tensor([new_state], device=global_device, dtype=torch.float32)
        return p, s

    rel_action = action_no % env.num_actions  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait

    # use the appropriate function to get the probability and state array for each possible action type:
    if (rel_action == 0 or rel_action == 1):
        p, s = template_move(env, state_tensor, action_no)
    elif (rel_action == 2):
        p, s = template_complete(env, state_tensor, action_no)
    elif (rel_action == 3):
        p, s = template_wait(env, state_tensor, action_no)
    return p, s


# %%


"""
This block defines the rewards model of the system
"""


def r_model(env, state_tensor, action, next_state_tensor):
    reward = 0

    # rewards for blocked actions
    # this is necessary to stop the total rewards shooting up when blocked actions are taken
    # mostly a diagnostic thing... I think
    if (env.blocked_model(env, state_tensor)[action] == 1):
        # print("action blocked")  # pretty sure this should be firing more often -- dunno
        return 0

    rel_action = action % env.num_actions  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait
    # print(f"action: {rel_action} ( 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait) ")
    # print(f"robot location {state_tensor[math.floor(action/env.num_robots) * 2]}")

    # reward for checking a goal by moving onto its position
    if (rel_action < 2):
        for i in range(env.num_goals):
            # check if goal went from unchecked to checked
            if (state_tensor[(env.num_robots * 2) + (i * 5) + 2] != next_state_tensor[(env.num_robots * 2) + (i * 5) + 2]):
                reward += 100

    # rewards for completing goals
    for i in range(env.num_goals):
        # check if any goals have been accomplished
        if (state_tensor[(env.num_robots * 2) + (i * 5) + 1] == 1 and next_state_tensor[(env.num_robots * 2) + (i * 5) + 1] == 0):
            reward += 1000

    return reward


# %%


"""
This block defines which actions are blocked in each state
Essentially, this is an auxiliary part of the transition model
"""


def b_model(env, state_tensor):
    blocked_actions = np.zeros(env.action_space.n)

    for i in range(env.num_robots):
        # print(state_tensor)
        active_robot_loc = state_tensor[i * 2]
        if (state_tensor[(i * 2) + 1]):  # clock
            blocked_actions[i * env.num_actions: (i * env.num_actions) + env.num_actions] = 1  # block these actions
        else:
            for j in range(env.num_robots):
                other_robot_loc = state_tensor[j * 2]

                if (i == j):
                    continue  # don't check robots against themselves

                if (active_robot_loc == other_robot_loc):
                    raise ValueError(f"Two robots occupy the same location (r{i} & r{j} @ {active_robot_loc}).")

            blocked_actions[(i * env.num_actions)] = get_counter_cw_blocked(env, state_tensor, i)
            blocked_actions[(i * env.num_actions) + 1] = get_cw_blocked(env, state_tensor, i)

            block_task_completion = True
            for k in range(env.num_goals):
                # if (state_tensor[f"goal{k} location"] == active_robot_loc and state_tensor[f"goal{k} active"] == 1):
                if (state_tensor[(env.num_robots * 2) + (k * 5)] == active_robot_loc and state_tensor[(env.num_robots * 2) + (k * 5) + 1] == 1):
                    block_task_completion = False  # unblock this engage action
                    break
            blocked_actions[(i * env.num_actions) + 2] = block_task_completion

    return torch.tensor(blocked_actions, dtype=torch.bool, device=global_device)


def get_counter_cw_blocked(env, state_tensor, robot_no):
    moving_robot_loc = state_tensor[robot_no * 2]

    for j in range(env.num_robots):
        other_robot_loc = state_tensor[j * 2]
        if (robot_no == j):  # don't need to check robots against themselves
            continue
        if (moving_robot_loc == other_robot_loc):
            raise ValueError(f"Two robots occupy the same location (r{robot_no} & r{j} @ {moving_robot_loc}).")
        if (other_robot_loc == (moving_robot_loc + 1) % env.size):
            return True
    return False


def get_cw_blocked(env, state_tensor, robot_no):
    moving_robot_loc = state_tensor[robot_no * 2]

    for j in range(env.num_robots):
        other_robot_loc = state_tensor[j * 2]
        if (robot_no == j):  # don't need to check robots against themselves
            continue
        if (moving_robot_loc == other_robot_loc):
            raise ValueError(f"Two robots occupy the same location (r{robot_no} & r{j} @ {moving_robot_loc}).")
        if (other_robot_loc == (env.size - 1 if moving_robot_loc - 1 < 0 else moving_robot_loc - 1)):
            return True

    return False


def state_is_final(env, state_tensor):
    for i in range(env.num_goals):
        # iterate over goals in state
        # if (state_tensor[f"goal{i} checked"] == 0 or (state_tensor[f"goal{i} checked"] == 1 and state_tensor[f"goal{i} active"] == 1)):
        #     return False
        if (state_tensor[(env.num_robots * 2) + (i * 5) + 2] == 0 or (state_tensor[(env.num_robots * 2) + (i * 5) + 2] == 1 and state_tensor[(env.num_robots * 2) + (i * 5) + 1] == 1)):
            return False

    return True
