#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:48:07 2024

@author: brendandevlin-hill
"""

import gymnasium as gym
import DQN

device = "cpu"

env = gym.make("MazeEnv",
               size=100,
               goal=[99,99],
               walls=[])

state, info = env.reset()

def decay_function(ep, e_max, e_min, num_eps):
    return DQN.linear_epsilon_decay(episode=ep, epsilon_max=e_max, epsilon_min=e_min, num_episodes=num_eps, max_epsilon_time=50, min_epsilon_time=50)

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
target_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

trained_dqn, dur, re, eps = DQN.train_model(env,
                                            policy_net,
                                            target_net,
                                            epsilon_decay_function=decay_function,
                                            alpha=1e-2,
                                            gamma=0.75,
                                            reset_options={"type": "statetree"},
                                            num_episodes=400,
                                            usePseudorewards=True,
                                            plot_frequency=10,
                                            tree_prune_frequency=50,
                                            state_tree_capacity=100,
                                            batch_size=256)

s, a, st = DQN.evaluate_model(trained_dqn, 1000, env)

print(s)
