#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:31:21 2023

@author: brendandevlin-hill
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import Qlearning as ql
import numpy as np

from IPython import display

import time

torch.set_grad_enabled(True)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Returns the row of the qtable for this state
    def forward(self, x):
        print("forward x.shape:", x, x.shape)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
            
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    
    """
        Replay memory allows us to store a fragment of the agent's experiences.
        This can then be sampled from later for learning.
        The advantage of this approach is that it breaks the correlation between samples that
        occurs if one simply samples each timestep in a linear fashion.
        
        Each 'experience' a Transition object containing a state, action, resultant state, and reward
        
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
    
def PlotDurations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())

#%%

## instantiate the env environment
env = gym.make('SimpleGrid-4x4-v0', 
               obstacle_map = [
                    "0000",
                    "0000",
                    "0000",
                    "0000",
                ],
                render_mode = "rgb_array") # appears to have an obstacle map by default
env.metadata["render_fps"] = 100

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) 

n_training_episodes = int(1e6)
max_epsilon = 0.9 # exploration rate
decay_rate = 1e-9 # decay of exploration rate
min_epsilon = 0.05 
learning_rate = 0.5 # learning rate of the optimiser
gamma = 0.7 # discount factor
batch_size = 128 # num experiences in the replay buffer
tau = 0.5

device = "cpu"

observation, info = env.reset(options={'start_loc':0, 'goal_loc':15})
num_actions = env.action_space.n

## initialise the DQNs
policy_DQN = DQN(1, num_actions).to(device) # this will return our policy
target_DQN = DQN(1, num_actions).to(device) # to train the policy DQN
target_DQN.load_state_dict(policy_DQN.state_dict())

loss_fn = nn.HuberLoss()
optimiser = optim.AdamW(policy_DQN.parameters(), lr=learning_rate, amsgrad=True)
memory = ReplayMemory(capacity = 10000) # initialise the replay memory, which will store experiences of the DQN(s)

#%%

def OptimiseModel():
    if (len(memory) < batch_size):
        return
    # print("############# transitions/batch")
    transitions = memory.sample(5) # take experiences from memory
    # creates Transition obj wherein entries are tuples of tensors corresponding to entries of 'transitions array':
    print("################# Transitions")
    print(transitions)
    print(*zip(transitions))
    batch = Transition(*zip(*transitions))
    print(batch)

    # create a tensor with 'true' for non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                            device = "cpu",
                                            dtype = torch.bool)
    # all the next states (that exist)
    # torch.cat concatenates the tensors together
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
 
    state_batch = torch.cat(batch.state)
    print(state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    ## Computation of Q(s,a)
    # select the q_values associated with performing actions in states
    # the .gather iterates down the states and selects the qvalue corresponding
    # to the appropriate action in action_batch
    state_action_values = policy_DQN(state_batch).gather(1, action_batch)
    
    ## Computation of max(Q(s',a')), i.e. V(s')
    # calculate V(s') with the target DQN and select max q
    # if the state is final, V(s') will be zero thanks to the state mask
    next_state_values = torch.zeros(batch_size, device = "cpu")
    with torch.no_grad():
        next_state_values[non_final_mask] = target_DQN(non_final_next_states).max(1)[0]
    # compute expected q values:
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = nn.SmoothL1Loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_DQN.parameters(), 100)
    optimiser.step()
    
#%%
done_counter_period = 0 # counts how many times the task is completed per period
period_times = [] # stores time per run, is emptied at end of episode
period_rewards = [] # stores rewards per run, is reset after 'period_length' episodes
period_length = 100 # length of the period. For diagnostics only.
max_steps = 10 # maximum steps before a run terminates
verbose = False
episode_durations = []

# training loop
for episode in range(n_training_episodes):

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)    
    episode_reward = 0
    step = 0
    done = False

    # Reset the environment
    state, info = env.reset(options={'start_loc':0, 'goal_loc':15})
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    terminated = False
    
    # action codes:
    # 0: decrease x by 1 (left)
    # 1: increase x by 1 (right)
    # 2: decrease y by 1 (down)
    # 3: increase y by 1 (up)
    
    # repeat
    for t in count():
        
        ## during the step, we want to construct a single 'experience' (Transtition)
        
        # qvalues = policy_DQN.forward(x = state) # get the qvalues based on the the current state
        action = ql.TensorEpsilonPolicyGreedy(state, policy_DQN, env, epsilon) # find what action to take
        observation, reward, terminated, _, info = env.step(action.item()) # move the system and collect information (.item() extracts entry from single-item tensors)
        reward = torch.tensor([reward], device = "cpu")
        done = terminated or t > max_steps
        episode_reward += reward # for diagnostics
        
        if(terminated):
            next_state = None
            done_counter_period += 1
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward) # store the transition
        state = next_state 
        
        OptimiseModel() # perform one step of optimisation

        # perform a soft update of the target DQN
        target_state_dict = target_DQN.state_dict()
        policy_state_dict = policy_DQN.state_dict()
        
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * tau + target_state_dict[key] * (1-tau)
            target_DQN.load_state_dict(target_state_dict)
            
        if done:
            episode_durations.append(t+1)
            PlotDurations(episode_durations)
            break
        


    period_times.append(step)
    period_rewards.append(episode_reward)

    if (episode%period_length == 0):
        period_done_percent = (done_counter_period*100/period_length)
        if(episode > 0): # goes a bit weird if episode 0 is printed
            print("\nEpisode {: >5d}/{:>5d} | epsilon: {:0<7.5f} | Av. steps: {: >4.2f} | Min steps: {: >4d} | Av. reward: {: >4.6f} | Completed: {: >4.2f}%".format(episode,
                                                                                                                                        n_training_episodes,
                                                                                                                                        epsilon,
                                                                                                                                        np.mean(period_times),
                                                                                                                                        np.min(period_times),
                                                                                                                                        np.mean(period_rewards),
                                                                                                          period_done_percent), end="")
        done_counter_period = 0 # counts how many times the task is completed per period
        period_times = [] # stores time per run, is emptied at end of episode
        period_rewards = [] # stores rewards per run, is reset after 'period_length' episodes
        period_length = 100 # length of the period. For diagnostics only.

print("\nTraining finished.")


        
    