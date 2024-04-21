#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:43:45 2023

@author: brendandevlin-hill
"""


import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import sys

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

system_parameters = namedtuple("system_parameters",
                               ("size",
                                "robot_status",
                                "robot_locations",
                                "goal_locations",
                                "goal_activations",
                                "goal_checked",
                                "goal_completion_probabilities",
                                "goal_discovery_probabilities",
                                "elapsed_ticks"
                                ))


# these are needed to get around some restrictions on how tuples work:
def update_system_parameters(t, key, new_value):
    new_tuple = t._asdict()
    new_tuple[key] = new_value
    return system_parameters(**new_tuple)


def copy_system_parameters(t):
    return system_parameters(**t._asdict())


class hashdict(dict):

    def __hash__(self):
        return hash(frozenset(self))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.warning = False
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        if(len(self) == self.capacity and self.warning is False):
            print("REPLAY AT CAPACITY: " + str(len(self)))
            self.warning = True

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, nodes_per_layer=128):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, nodes_per_layer)
        self.layer2 = nn.Linear(nodes_per_layer, nodes_per_layer)
        self.layer3 = nn.Linear(nodes_per_layer, nodes_per_layer)
        self.layer4 = nn.Linear(nodes_per_layer, nodes_per_layer)
        self.layer5 = nn.Linear(nodes_per_layer, nodes_per_layer)
        self.layer6 = nn.Linear(nodes_per_layer, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print("forward x.shape", x, x.shape)
        try:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            x = F.relu(self.layer5(x))
            return self.layer6(x)
        except RuntimeError:
            print(x)


def select_action(dqn, env, state, epsilon, forbidden_actions=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roll = random.random()
    if roll > epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # if(epsilon == 0):
            # print(dqn(state), dqn(state).max(1)[1].view(1, 1))
            return dqn(state).max(1)[1].view(1, 1)
    else:
        sample = env.action_space.sample()
        print(sample)
        while forbidden_actions[sample] == 1:
            sample = env.action_space.sample()
        return torch.tensor([[sample]], device=device, dtype=torch.long)


def plot_status(episode_durations, rewards, epsilons):

    fig = plt.figure()

    fig, mid_ax = plt.subplots(figsize=(10,10), layout="constrained")
    mid_ax.grid(False)

    bot_ax = mid_ax.twinx()
    upper_ax = mid_ax.twinx()

    mid_ax.set_ylabel('Duration (steps)')
    # mid_ax.set_yticks(np.linspace(0,
    #                               np.floor(max(episode_durations) / 5) * int(max(episode_durations)) + 6,
    #                               5))
    bot_ax.set_ylabel("Epsilon")
    bot_ax.set_ylim(0,1)
    bot_ax.set_yticks(np.linspace(0,1,21))

    upper_ax.set_ylabel("Reward")
    mid_ax.set_xlabel('Episode')

    bot_ax.spines['right'].set_position(('outward', 60))

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    rewards_t = torch.tensor(rewards, dtype=torch.float)

    color1, color2, color3 = plt.cm.viridis([0, .5, .9])

    epsilon_plot = bot_ax.plot(epsilons, color="orange", label="epsilon", zorder=0)
    duration_plot = mid_ax.plot(episode_durations, color="royalblue", alpha=0.2, label="durations", zorder=5)
    reward_plot = upper_ax.plot(rewards, color="mediumseagreen", alpha=0.5, label="rewards",zorder=10)

    bot_ax.set_zorder(0)
    mid_ax.set_zorder(5)
    mid_ax.set_facecolor("none")
    upper_ax.set_zorder(10)

    # Take 100-episode averages and plot them too
    if len(durations_t) >= 100:

        duration_means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        duration_means = torch.cat((torch.zeros(99), duration_means))
        duration_av_plot = mid_ax.plot(duration_means.numpy(), color="indianred", label="average dur. ", lw=3, zorder=20)
        mid_ax.axhline(duration_means.numpy()[-1], color="indianred", alpha=1, ls="--", zorder=40)
        mid_ax.text(0, duration_means.numpy()[-1], "avg dur.: {:.2f}".format(duration_means.numpy()[-1]), zorder=60)

        reward_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        reward_means = torch.cat((torch.zeros(99), reward_means))
        reward_av_plot = upper_ax.plot(reward_means.numpy(), color="green", label="average r.", lw=3, zorder=20)
        upper_ax.axhline(reward_means.numpy()[-1], color="green", alpha=1, ls="--", zorder=40)
        upper_ax.text(0, reward_means.numpy()[-1], "avg r.: {:.2f}".format(reward_means.numpy()[-1]), zorder=60)

        handles = duration_plot + epsilon_plot + reward_plot + duration_av_plot + reward_av_plot

    else:
        handles = duration_plot + epsilon_plot + reward_plot
        mid_ax.axhline(episode_durations[-1], color="grey", ls="--")
        mid_ax.text(0, episode_durations[-1], episode_durations[-1])

    mid_ax.legend(handles=handles, loc='best').set_zorder(100)
    # plt.pause(0.001)  # pause a bit so that plots are updated

    return fig


def exponential_epsilon_decay(episode, epsilon_max, epsilon_min, max_epsilon_time, min_epsilon_time, num_episodes, decay_rate=None):

    if(not decay_rate):
        decay_rate = np.log(100 * (epsilon_max - epsilon_min)) / (num_episodes - (max_epsilon_time + min_epsilon_time))  # ensures epsilon ~= epsilon_min at end

    if(episode < max_epsilon_time):
        # print("max ep")
        epsilon = epsilon_max
    elif(episode > num_episodes - min_epsilon_time):
        epsilon = epsilon_min
    else:
        # print("decaying ep")
        decay_term = math.exp(-1. * (episode - max_epsilon_time) * decay_rate)
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * decay_term

    return epsilon


def linear_epsilon_decay(episode, epsilon_max, epsilon_min, max_epsilon_time, min_epsilon_time, num_episodes):

    gradient = (epsilon_min - epsilon_max) / (num_episodes - max_epsilon_time - min_epsilon_time)

    if(episode < max_epsilon_time):
        epsilon = epsilon_max
    elif(episode > num_episodes - min_epsilon_time):
        epsilon = epsilon_min
    else:
        epsilon = epsilon_max + (gradient * (episode - max_epsilon_time))

    return epsilon


def train_model(
        env,                            # gymnasium environment
        policy_net,                     # policy network to be trained
        target_net,                     # target network to be soft updated
        reset_options=None,             # options passed when resetting env
        num_episodes=1000,              # number of episodes for training
        gamma=0.6,                      # discount factor
        epsilon_max=0.95,               # max exploration rate
        epsilon_min=0.05,               # min exploration rate
        epsilon_decay_function=None,    # will be exponential if not set.
        alpha=1e-3,                     # learning rate for policy DeepQNetwork
        tau=0.005,                      # soft update rate for ap DeepQNetwork
        usePseudorewards=True,          # whether to calculate and use pseudorewards
        max_steps=None,                 # max steps per episode
        batch_size=128,                 # batch size of the replay memory
        buffer_size=10000,              # total size of replay memory buffer
        state_tree_capacity=200,        # capacity of the state tree
        tree_prune_frequency=10,        # number of episodes between pruning the state tree
        plot_frequency=10,              # number of episodes between status plots (0=disabled)
        checkpoint_frequency=0,          # number of episodes between saving weights (0=disabled)
        # save_plot_data=False            # whether to save the plotting data to a file
):
    """
    For training a DQN on a gymnasium environment.

    Inputs:
        env,                            # gymnasium environment
        policy_net,                     # policy network to be trained
        target_net,                     # target network to be soft updated
        reset_options=None,             # optioreset_optionsns passed when resetting env
        num_episodes=1000,              # number of episodes for training
        gamma=0.6,                      # discount factor
        epsilon_max=0.95,               # max exploration rate
        epsilon_min=0.05,               # min exploration rate
        epsilon_decay_function=None,    # will be exponential if not set. takes (epsiode, max_epsilon, min_epsilon) as arguments
        alpha=1e-3,                     # learning rate for policy DeepQNetwork
        tau=0.005,                      # soft update rate for target DeepQNetwork
        usePseudorewards=True,          # whether to calculate and use pseudorewards
        max_steps=None,                 # max steps per episode
        batch_size=128,                 # batch size of the replay memory
        buffer_size=10000               # total size of replay memory buffer
        state_tree_capacity = 200,      # capacity of the state tree
        tree_prune_frequency=10,        # number of episodes between pruning the state tree
        plot_frequency=10,              # number of episodes between status plots (0=disabled)
        checkpoint_frequency=0          # number of episodes between saving weights (0=disabled)

    Outputs:
        (None)
    """

    # store values for plotting
    epsilons = np.empty(num_episodes)
    episode_durations = np.empty(num_episodes)
    rewards = np.empty(num_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_string = str(env.state).replace(',', ',\n\t\t\t')

    print(f"""
            Commensing training.
            Device: {device}
            Environment: {env.unwrapped.spec.id}
            ----

            Environmental parameters:
            {state_string}
            {reset_options}

            ----

            Training hyperparameters:
            num_episodes = {num_episodes}
            gamma = {gamma}
            epsilon_max = {epsilon_max}
            epsilon_min = {epsilon_min}
            epsilon_decay_function = {"default" if not epsilon_decay_function else "custom"}
            alpha = {alpha}
            tau = {tau}
            state_tree_capacity = {state_tree_capacity}
            tree_prune_frequency = {tree_prune_frequency}
            max_steps = {"as per env" if not max_steps else max_steps}
            batch_size = {batch_size}
            buffer_size = {buffer_size}

            ----

            Diagnostic values:
            plot_frequency = {plot_frequency}
            checkpoint_frequency = {checkpoint_frequency}
          """)

    # Initialisation of NN apparatus
    optimiser = optim.AdamW(policy_net.parameters(), lr=alpha, amsgrad=True)
    memory = ReplayMemory(buffer_size)
    torch.set_grad_enabled(True)
    plotting_on = plot_frequency < num_episodes and plot_frequency != 0
    checkpoints_on = checkpoint_frequency < num_episodes and checkpoint_frequency != 0
    if(checkpoints_on):
        file = open(os.getcwd() + "/outputs/checkpoints/diagnostics", "w")
        file.write("# no data yet...")

    state, info = env.reset()  # reset to init
    # state tree reset optimisation

    # Initialise some hyperparameters
    # if no decay function is supplied, set it to a default exponential decay
    if epsilon_decay_function is None:
        decay_rate = np.log(100 * (epsilon_max - epsilon_min)) / (num_episodes)  # ensures epsilon ~= epsilon_min at end

        def epsilon_decay_function(ep, e_max, e_min, num_eps):
            return exponential_epsilon_decay(episode=ep,
                                             epsilon_max=e_max,
                                             epsilon_min=e_min,
                                             max_epsilon_time=0,
                                             min_epsilon_time=0,
                                             decay_rate=decay_rate,
                                             num_episodes=num_eps)
    if not max_steps:
        max_steps = np.inf  # remember: ultimately defined by the gym environment

    # Loop over training epsiodes
    start_time = time.time()
    for i_episode in range(num_episodes):

        if((i_episode % int(num_episodes / 10)) == 0):
            print(f"{i_episode}/{num_episodes} complete...")

        # calculate the new epsilon
        epsilon = epsilon_decay_function(i_episode, epsilon_max, epsilon_min, num_episodes)
        if(plotting_on or checkpoints_on):
            epsilons[i_episode] = epsilon

        state, info = env.reset()

        # Initialise the first state
        if(usePseudorewards):
            phi_sprime = info["pseudoreward"]  # phi_sprime is the pseudoreward of the new state
        stateT = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)
        ep_reward = 0

        # Navigate the environment
        for t in count():

            # calculate action utilities and choose action
            action_utilities = policy_net.forward(stateT)[0]
            # get blocked actions
            # print(action_utilities)
            blocked = env.blocked_model(env, state)
            masked_utilities = [action_utilities[i] if not blocked[i] else -1000 for i in range(len(action_utilities))]
            # print(masked_utilities, type(masked_utilities))
            action_utilities = torch.tensor([masked_utilities], dtype=torch.float32, device=device)
            if(np.random.random() < epsilon):
                sample = env.action_space.sample()
                while blocked[sample] == 1:
                    # print("blocked")
                    sample = env.action_space.sample()
                action = torch.tensor([[sample]], device=device, dtype=torch.long)
            else:
                # print(action_utilities)
                action = action_utilities.max(1)[1].view(1, 1)
                # print(action)

            # apply action to environment
            state, reward, terminated, truncated, info = env.step(action)

            # calculate pseudoreward
            if(usePseudorewards):
                phi = phi_sprime
                phi_sprime = info["pseudoreward"]
                pseudoreward = (gamma * phi_sprime - phi)
            else:
                pseudoreward = 0

            # calculate reward
            reward = torch.tensor([reward + pseudoreward], device=device)
            ep_reward += reward.item()

            # work out if the run is over
            done = terminated or truncated or (t > max_steps)
            if terminated:
                next_stateT = None
            else:
                next_stateT = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

            # move transition to the replay memory
            memory.push(stateT, action, next_stateT, reward)
            stateT = next_stateT

            # run optimiser
            optimise_model(policy_net, target_net, memory, optimiser, gamma, batch_size)

            # Soft-update the target net
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)

            # if done, process data and make plots
            if done:
                # print("done")
                # print([env.state[f"goal{i} active"] for i in range(12)])
                if(plotting_on or checkpoints_on):
                    episode_durations[i_episode] = info["elapsed steps"]
                    rewards[i_episode] = ep_reward
                if (plotting_on and i_episode % plot_frequency == 0 and i_episode > 0):
                    f = plot_status(episode_durations[:i_episode], rewards[:i_episode], epsilons[:i_episode])
                    plt.show()
                    plt.close(f)
                if (checkpoints_on and i_episode % checkpoint_frequency == 0 and i_episode > 0):
                    # write durations, rewards and epsilons to file
                    np.savetxt(os.getcwd() + "/outputs/checkpoints/diagnostics", np.vstack((episode_durations,rewards,epsilons)).transpose())
                    torch.save(policy_net.state_dict(), os.getcwd() + f"/outputs/checkpoints/policy_weights_epoch{i_episode}")
                break

    print(f"Training complete in {int(time.time()-start_time)} seconds.")
    return policy_net, episode_durations, rewards, epsilons


def optimise_model(policy_dqn, target_dqn, replay_memory, optimiser, gamma, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(replay_memory) < batch_size:
        return
    transitions = replay_memory.sample(batch_size)  # why 128? should this be batch_size?

    # conglomerate the transitions into one object wherein each entry state, action,
    # etc is a tensor containing all of the corresponding entries of the original transitions array
    batch = Transition(*zip(*transitions))

    # boolean mask of which states are final (i.e. termination occurs in this state)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

    # collection of non-final s
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # the qvalues of actions in this state. the .gather gets the qvalue corresponding to the
    # indices in 'action_batch'
    try:
        state_action_values = policy_dqn(state_batch).gather(1, action_batch)
    except AttributeError:
        print("caught")

    # q values of action in the next state
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimise the model
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), 100)  # stops the gradients from becoming too large
    optimiser.step()


def plot_state_tree(state_tree, env, reset=False):
    plt.figure(figsize=(10,10))
    plt.scatter([state["x"] for state in state_tree], [state["y"] for state in state_tree])
    plt.plot([env.goal[0]], [env.goal[1]], marker="x", color="orange", markersize=10)
    if(reset):
        plt.plot([env.state["x"]], [env.state["y"]], marker="$O$", color="magenta", markersize=20)
    else:
        plt.plot([env.state["x"]], [env.state["y"]], marker="x", color="red", markersize=10)

    plt.show()


def evaluate_model(dqn,
                   num_episodes,
                   env,
                   max_steps,
                   reset_options=None,
                   render=False):

    print("Evaluating...")

    if ("win" in sys.platform and render):
        print("Cannot render on windows...")
        render = False

    env.set_rendering(render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation running on {device}.")

    # ticks = []
    # goal_resolutions = []
    steps = []
    deadlock_counter = 0
    deadlock_traces = deque([], maxlen=10)  # store last 10 deadlock traces

    for i in range(num_episodes):
        if(reset_options):
            state, info = env.reset(options=reset_options.copy())
        else:
            state, info = env.reset()

        states = [state]
        actions = []
        stateT = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            # action_utilities = dqn.forward(stateT)
            # action = action_utilities.max(1)[1].view(1, 1)

            action_utilities = dqn.forward(stateT)[0]
            # get blocked actions
            blocked = env.blocked_model(env, state)
            masked_utilities = [action_utilities[i] if not blocked[i] else -1000 for i in range(len(action_utilities))]
            action_utilities = torch.tensor([masked_utilities], dtype=torch.float32, device=device)
            action = action_utilities.max(1)[1].view(1, 1)
            # print(action_utilities)

            # apply action to environment
            state, reward, terminated, truncated, info = env.step(action.item())

            states.append(state)
            actions.append(action)

            stateT = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

            done = terminated

            if (done or truncated or t > max_steps):
                # ticks.append(info["elapsed ticks"])
                # goal_resolutions.append(np.sum(info["goal_resolutions"]))
                if (int(num_episodes / 10) > 0 and i % int(num_episodes / 10) == 0):
                    print(f"{i}/{num_episodes} episodes complete")
                break

        if(not done):
            deadlock_traces.append(states)
            deadlock_counter += 1

        steps.append(t)

    # ticks = np.array(ticks)
    # plt.figure(figsize=(10,10))
    # ticks_start = 0
    # # process 'ticks' into sub-arrays based on the unique entries in goal_resolutions
    # unique_res = np.unique(goal_resolutions)
    # for unique in unique_res:
    #     unique_ticks = ticks[goal_resolutions == unique]  # groups episodes with this unique number of tasks
    #     # plot the ticks. assign a range on x for each group based on the size of the group and where the last group ended.
    #     plt.plot(np.array(range(len(unique_ticks))) + ticks_start,
    #              unique_ticks,
    #              ls="",
    #              marker="o",
    #              label="{} goals - avg {:.2f}".format(int(unique), np.mean(unique_ticks)))
    #     ticks_start = len(unique_ticks) + ticks_start + num_episodes / 20

    # plt.legend()
    # plt.hlines(np.mean(ticks), 0, len(ticks) + len(unique_res) * num_episodes / 20, ls="--", color="grey")
    # plt.text(0,np.mean(ticks), f"avg: {np.mean(ticks)}")
    # plt.xticks([])
    # plt.ylabel("Duration / ticks")
    # plt.xlabel("Episode, sorted by number goals encountered")
    # plt.title("Evaluation durations")
    # plt.show()

    print("Evaluation complete.")
    print(f"{'CONVERGENCE SUCCESSFUL' if deadlock_counter==0 else 'FAILURE'} - Failed to complete {deadlock_counter} times")
    print(f"Percentage converged: {100 - (deadlock_counter*100/num_episodes)}")

    return states, actions, steps, deadlock_traces  # states, actions, ticks, steps


def verify_model(policy_net, env):
    """
    Inputs:
        policy_net - the NN encoding the policy to be verified
        env  - the environment in which the policy operates
    """
