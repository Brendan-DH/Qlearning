#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:15:27 2024

@author: brendandevlin-hill
"""


import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import DQN
import os
import numpy as np
import mdp_translation as mdpt
import collections
# from abc import ABC, abstractmethod
env_to_use = "Tokamak-v9"


def propagate_ensemble(env, num_particles, num_samples, initial_state, time_horizon):

    particle_rewards = np.zeros((num_particles))
    particle_states = []

    # initialise each particle into initial state
    for i in range(num_particles):
        particle_states.append(initial_state.copy())

    # create a random sample of actions for each time step
    actions = [i for i in range(env.action_space.n)]
    blocked = set(env.blocked_model(env, initial_state))
    actions = set(actions)
    allowed = list(actions - blocked)
    evaluations = np.zeros((env.action_space.n))
    print("Allowed actions:", allowed)

    # choose an action
    for start_action in allowed:
        print(f"Testing: {start_action}")
        # propagate up to time horizon
        for t in range(time_horizon):
            particle_rewards = np.zeros((num_particles))
            # loop over particles each time step
            for p in range(num_particles):
                if (t > 0):
                    # blocked_actions
                    blocked = set(env.blocked_model(env, particle_states[p]))
                    actions = set(actions)
                    allowed = actions - blocked
                    action_no = np.random.choice(list(allowed), 1)[0]
                else:
                    action_no = start_action

                p_array, s_array = env.transition_model(env, particle_states[p], action_no)
                #implement r_array/reward

                # roll dice to detemine resultant state from possibilities
                roll = np.random.random()
                thres = 0
                new_state_index = -1
                for i in range(len(p_array)):
                    thres += p_array[i]
                    # print("probs:", t, p_array)
                    if (roll < thres):
                        new_state_index = i
                        break
                if(new_state_index < 0):
                    raise ValueError("Something has gone wrong with choosing the state")

                new_state = s_array[new_state_index].copy()
                # terminated = True
                # for i in range(env.num_goals):
                #     if (new_state[f"goal{i} checked"] == 0 or new_state[f"goal{i} instantiated"] == 1):
                #         terminated = False
                #         break

                # # if a terminating state is found, the time horizon must be shortened to meet it
                # # this will ensure that during this propogation, the
                # if(terminated):
                #     time_to_termination = min(time_to_termination, t)

                reward = env.reward_model(env, particle_states[p], action_no, new_state)

                # print(reward)
                particle_rewards[p] += max(0, reward * np.exp(-t)) / num_particles
                particle_states[p] = new_state.copy()

            evaluations[start_action] += np.sum(particle_rewards)

        evaluations[start_action] = evaluations[start_action]

    return evaluations


starting_parameters = DQN.system_parameters(
    size=12,
    robot_status=[1,1,1],
    robot_locations=[1,6,9],
    goal_locations=[11,3,5],  # 8, 1, 0],
    goal_probabilities=[0.7,0.7,0.7],  # 0.7, 0.7, 0.7],
    goal_instantiations=[0,0,0,0,0,0],
    goal_resolutions=[0,0,0,0,0,0],
    goal_checked=[0,0,0,0,0,0,0],
    elapsed_ticks=0,
)

env = gym.make(env_to_use,
               system_parameters=starting_parameters,
               transition_model=mdpt.t_model,
               reward_model=mdpt.r_model,
               blocked_model=mdpt.b_model,
               training=True,
               render_mode="human")

state, info = env.reset()
plan = []
states = []
states.append(state)
for i in range(100):
    print(f"Time: {i}")
    e = propagate_ensemble(env, 1000, 9, state, 5)
    action_no = np.argmax(e)
    print(f"Chosen: {action_no}")
    plan.append(action_no)
    state, reward, terminated, truncated, info = env.step(action_no)
    states.append(state)
    if(terminated):
        break


print(plan)
print(states)
