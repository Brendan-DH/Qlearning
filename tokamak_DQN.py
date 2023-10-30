    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:12:23 2023

@author: brendandevlin-hill
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("Tokamak-v2", num_robots=3, size=24, num_goals=3, goal_locations=[1,5,9])  
# env = gym.make("CartPole-v1")
env.reset(options={"robot_locations" : [1,2,3]})
torch.set_grad_enabled(True)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print("forward x.shape", x, x.shape)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# gamma is the discount factor as mentioned in the previous section
# epsilon_max is the starting value of epsilon
# epsilon_min is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
gamma = 0.6 # a lower gamma will prioritise immediate rewards, naturally favouring shorter paths
epsilon_max = 0.9
epsilon_min = 0.05
explore_time = 0 # number of steps for which epsilon is held constant before starting to decay
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state, epsilon):
    global steps_done
    sample = random.random()
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
epsilons = []
rewards = []

fig = plt.figure()

def plot_durations(show_result=False):
    global fig
    plt.close(fig)
    
    fig, host = plt.subplots(figsize=(10,10), layout="constrained")
    host.grid(False)
    
    epsilon_ax = host.twinx()
    reward_ax = host.twinx()
    
    host.set_ylabel('Duration')
    host.set_yticks(np.linspace(0, int(max(episode_durations)), int(max(episode_durations)/20)+1))
    epsilon_ax.set_ylabel("Epsilon")
    epsilon_ax.set_ylim(0,1)
    epsilon_ax.set_yticks(np.linspace(0,1,21))
    
    reward_ax.set_ylabel("Reward")
    host.set_xlabel('Episode')
    
    epsilon_ax.spines['right'].set_position(('outward', 60))

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
        
    color1, color2, color3 = plt.cm.viridis([0, .5, .9])
    
    reward_plot = reward_ax.plot(np.array(rewards), color = "mediumseagreen", alpha=0.5, label = "rewards");
    duration_plot = host.plot(np.array(episode_durations), color = "royalblue", label = "durations");
    epsilon_plot = epsilon_ax.plot(np.array(epsilons), color = "orange", label = "epsilon");


    

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        av_plot = host.plot(means.numpy(), color="indianred", label="average");
        host.axhline(means.numpy()[-1], color = "indianred", alpha = 0.5, ls = "--")
        host.text(0, means.numpy()[-1], "avg: "+ str(means.numpy()[-1]))
        handles=duration_plot+epsilon_plot+reward_plot+av_plot
    else:
        handles=duration_plot+epsilon_plot+reward_plot
        host.axhline(episode_durations[-1], color = "grey", ls = "--")
        host.text(0, episode_durations[-1], episode_durations[-1])


    host.legend(handles=handles, loc='best')
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(128)

    # conglomerate the transitions into one object wherein each entry state, action,
    # etc is a tensor containing all of the corresponding entries of the original transitions array
    batch = Transition(*zip(*transitions))
    
    # boolean mask of which states are final
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # the qvalues of actions in this state. the .gather gets the qvalue corresponding to the
    # indices in 'action_batch'
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # q values of action in the next state
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # stops the gradients from becoming too large
    optimizer.step()
    
    
def evaluate_model():
        
    state, info = env.reset(options={"robot_locations" : [1,2,3]})
    state = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)
    
    states = [state]
    actions = []
    
    for t in count():
        
        action = select_action(state, 0)
        observation, reward, terminated, truncated, info = env.step(action.item())
        state = torch.tensor(list(observation.values()), dtype=torch.float32, device=device).unsqueeze(0)
        
        print(observation, action)
        states.append(observation)
        actions.append(action)
        
        done = terminated or truncated
        
        if(done):
            print("Completed in: " +  str(t+1))
            break
    
    return states, actions
        

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1000

epsilon_decay_rate =  np.log(100 * (epsilon_max-epsilon_min)) / (num_episodes-explore_time) # ensures epsilon ~= epsilon_min at end


for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset(options={"robot_locations" : [1,2,3]})
    av_dist = info["av_dist"] # average distance of robots from tasks, used for pseudorewards
    state = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)
    
    epsilon = epsilon_max if i_episode < explore_time else epsilon_min + (epsilon_max - epsilon_min) * \
        math.exp(-1. * (i_episode-explore_time) * epsilon_decay_rate)
    epsilons.append(epsilon)
    
    ep_reward = 0
    
    for t in count():
        action = select_action(state, epsilon)
        observation, reward, terminated, truncated, info = env.step(action.item())
        
        # calculate pseudoreward
        old_av_dist = av_dist
        av_dist = info["av_dist"]
        pseudoreward = gamma * 1/av_dist - 1/old_av_dist  
        
        reward = torch.tensor([reward+pseudoreward], device=device)
        ep_reward += reward.item()
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(list(observation.values()), dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(state, action, next_state, reward) # Store the transition in memory
        state = next_state
        optimize_model()

        # Soft update of the target DQN
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            rewards.append(ep_reward)
            if(t%10==0):
                plot_durations();
            break

print('Complete')
plot_durations(show_result=True);
plt.ioff()
plt.show()

#%%

states, actions = evaluate_model()




