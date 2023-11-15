#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:12:23 2023

@author: brendandevlin-hill
"""

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import DQN

num_robots = 3
size = 12
goal_locations = [11,5,2]
goal_probabilities = [0.1, 0.9, 0.7]

env_to_use = "Tokamak-v5"

env = gym.make(env_to_use,
               num_robots=num_robots,
               size=size,
               goal_locations=goal_locations,
               goal_probabilities = goal_probabilities,
               render_mode = None )
reset_options = {"robot_locations" : [1,3,5]}
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#%%
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
# policy_net.load_state_dict(torch.load(os.getcwd() + "/outputs/policy_weights_647487388"))
target_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

    
#%%
trained_dqn, dur, re, eps = DQN.train_model(env,
                                            policy_net,
                                            target_net,
                                            reset_options,
                                            # tau=0.005,
                                            num_episodes=1000,
                                            batch_size=128)    
filename = f"policy_weights_{int(np.random.rand()*1e9)}"
print(f"Saving as {filename}")

#%%
torch.save(trained_dqn.state_dict(), f"./outputs/{filename}")
_ = DQN.evaluate_model(dqn = policy_net,
                       num_episodes = 10,
                       template_env = env,
                       reset_options = reset_options,
                       env_name = env_to_use,
                       render = True)


#%%

# policy_net = DQN.DeepQNetwork(n_observations, n_actions).to(device)
# policy_net.load_state_dict(torch.load(os.getcwd() + f"/outputs/{filename}"))
# _ = DQN.evaluate_model(policy_net, 10000, env, reset_options, False)





