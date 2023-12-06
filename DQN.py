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
                                "breakage_probability",
                                "goal_locations",
                                "goal_probabilities",
                                "goal_instantiations",
                                "goal_resolutions",
                                "goal_checked",
                                "port_locations",
                                "elapsed"
                                ))


# these are needed to get around some restrictions on how tuples work:
def update_system_parameters(t, key, new_value):
    new_tuple = t._asdict()
    new_tuple[key] = new_value
    return system_parameters(**new_tuple)


def copy_system_parameters(t):
    return system_parameters(**t._asdict())


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


class DeepQNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print("forward x.shape", x, x.shape)
        try:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)
        except RuntimeError:
            print(x)


def select_action(dqn, env, state, epsilon, forbidden_actions=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # if(epsilon == 0):
            # print(dqn(state), dqn(state).max(1)[1].view(1, 1))
            return dqn(state).max(1)[1].view(1, 1)
    else:
        sample = env.action_space.sample()
        return torch.tensor([[sample]], device=device, dtype=torch.long)


def plot_status(episode_durations, rewards, epsilons):

    fig = plt.figure()

    fig, mid_ax = plt.subplots(figsize=(10,10), layout="constrained")
    mid_ax.grid(False)

    bot_ax = mid_ax.twinx()
    upper_ax = mid_ax.twinx()

    mid_ax.set_ylabel('Duration (ticks)')
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

    epsilon_plot = bot_ax.plot(np.array(epsilons), color="orange", label="epsilon", zorder=0)
    duration_plot = mid_ax.plot(np.array(episode_durations), color="royalblue", label="durations", zorder=5)
    reward_plot = upper_ax.plot(np.array(rewards), color="mediumseagreen", alpha=0.5, label="rewards",zorder=10)

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


def train_model(
        env,                      # gymnasium environment
        policy_net,               # policy network to be trained
        target_net,               # target network to be soft updated
        reset_options=None,       # options passed when resetting env
        num_episodes=1000,        # number of episodes for training
        gamma=0.6,                # discount factor
        epsilon_max=0.95,         # max exploration rate
        epsilon_min=0.05,         # min exploration rate
        epsilon_decay=None,       # decay rate, will be set automatically if None
        max_epsilon_time=0,       # time at maximum epsilon
        min_epsilon_time=100,     # time at minimum epsilon
        alpha=1e-3,               # learning rate for policy DeepQNetwork
        tau=0.005,                # soft update rate for target DeepQNetwork
        usePseudorewards=True,    # whether to calculate and use pseudorewards
        max_steps=None,           # max steps per episode
        batch_size=128,           # batch size of the replay memory
        plot_frequency=10,        # number of episodes between status plots (0=disabled)
        checkpoint_frequency=0    # number of episodes between saving weights (0=disabled)
):

    # store values for plotting
    epsilons = []
    episode_durations = []
    rewards = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"""
            Commensing training.
            Device: {device}
            Environment: {env.unwrapped.spec.id}
            ----

            Environmental parameters:
            {env.get_parameters()}
            {reset_options}

            ----

            Training hyperparameters:
            num_episodes = {num_episodes}
            gamma = {gamma}
            epsilon_max = {epsilon_max}
            epsilon_min = {epsilon_min}
            epsilon_decay = {"default" if not epsilon_decay else epsilon_decay}
            max_epsilon_time = {max_epsilon_time}
            alpha = {alpha}
            tau = {tau}
            max_steps = {"as per env" if not max_steps else max_steps}
            batch_size = {batch_size}

            ----

            Diagnostic values:
            plot_frequency = {plot_frequency}
            checkpoint_frequency = {checkpoint_frequency}
          """)

    optimiser = optim.AdamW(policy_net.parameters(), lr=alpha, amsgrad=True)
    memory = ReplayMemory(10000)
    torch.set_grad_enabled(True)

    if not epsilon_decay:
        epsilon_decay = np.log(100 * (epsilon_max - epsilon_min)) / (num_episodes - min_epsilon_time - max_epsilon_time)  # ensures epsilon ~= epsilon_min at end

    if not max_steps:
        max_steps = np.inf

    start_time = time.time()
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        if reset_options:
            state, info = env.reset(options=reset_options.copy())
        else:
            state, info = env.reset()
        av_dist = info["av_dist"]  # average distance of robots from tasks, used for pseudorewards
        # print("state", state, list(state.values()))

        state = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

        if(i_episode < max_epsilon_time):
            # print("max ep")
            epsilon = epsilon_max
        elif(i_episode > num_episodes - min_epsilon_time):
            # print("min ep")
            decay_term = math.exp(-1. * (num_episodes - min_epsilon_time) * epsilon_decay)
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * decay_term + 0.025 * np.sin((i_episode - max_epsilon_time) * 2 * np.pi / 20)
        else:
            # print("decaying ep")
            decay_term = math.exp(-1. * (i_episode - max_epsilon_time) * epsilon_decay)
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * decay_term

        epsilons.append(epsilon)

        ep_reward = 0

        for t in count():
            action = select_action(policy_net, env, state, epsilon)
            observation, reward, terminated, truncated, info = env.step(action.item())

            # calculate pseudoreward
            old_av_dist = av_dist  # for phi(s)
            av_dist = info["av_dist"]  # for phi(s')
            elapsed_ticks = observation["elapsed_ticks"]
            if(usePseudorewards):
                pseudoreward = (gamma * 1 / (av_dist + 1) - 1 / (old_av_dist + 1))
            else:
                pseudoreward = 0

            reward = torch.tensor([reward + pseudoreward], device=device)
            ep_reward += reward.item()

            done = terminated or truncated or (t > max_steps)
            # print(terminated, truncated, t, done)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(list(observation.values()), dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)  # Store the transition in memory
            state = next_state
            optimise_model(policy_net, target_net, memory, optimiser, gamma, 128)

            # Soft update of the target DeepQNetwork
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(elapsed_ticks)
                rewards.append(ep_reward)
                if (plot_frequency != 0 and i_episode % plot_frequency == 0 and i_episode > 0):
                    f = plot_status(episode_durations, rewards, epsilons)
                    plt.show()
                    plt.close(f)
                if (checkpoint_frequency != 0 and i_episode % checkpoint_frequency == 0 and i_episode > 0):
                    torch.save(policy_net.state_dict(), os.getcwd() + f"./outputs/policy_weights_{int(np.random.rand()*1e9)}")
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


def evaluate_model(dqn, num_episodes, system_parameters, reset_options=None, env_name="SEAMSOperationalTokamak-v1", render=False):

    print("Evaluating...")

    if ("win" in sys.platform and render):
        print("Cannot render on windows...")
        render = False

    env = gym.make(env_name,
                   system_parameters=system_parameters,
                   training=True,
                   render_mode="human" if render else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation running on {device}.")

    times = []
    goal_resolutions = []
    ts = []

    for i in range(num_episodes):
        state, info = env.reset(options=reset_options.copy())

        states = [state]
        actions = []
        state = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            action = select_action(dqn, env, state, 0)
            observation, reward, terminated, truncated, info = env.step(action.item())
            state = torch.tensor(list(observation.values()), dtype=torch.float32, device=device).unsqueeze(0)

            states.append(observation)
            actions.append(action)

            done = terminated

            if (done or truncated):
                times.append(observation["elapsed_ticks"])
                goal_resolutions.append(np.sum(info["goal_resolutions"]))
                if (int(num_episodes / 10) > 0 and i % int(num_episodes / 10) == 0):
                    print(f"{i}/{num_episodes} episodes complete")
                break

        if(not done):
            print("deadlock", observation, action)
        ts.append(t)

    times = np.array(times)
    plt.figure(figsize=(10,10))
    times_start = 0
    # process 'times' into sub-arrays based on the unique entries in goal_resolutions
    unique_res = np.unique(goal_resolutions)
    for unique in unique_res:
        unique_times = times[goal_resolutions == unique]  # groups episodes with this unique number of tasks
        # plot the times. assign a range on x for each group based on the size of the group and where the last group ended.
        plt.plot(np.array(range(len(unique_times))) + times_start,
                 unique_times,
                 ls="",
                 marker="o",
                 label="{} goals - avg {:.2f}".format(int(unique), np.mean(unique_times)))
        times_start = len(unique_times) + times_start + num_episodes / 20

    plt.legend()
    plt.hlines(np.mean(times), 0, len(times) + len(unique_res) * num_episodes / 20, ls="--", color="grey")
    plt.text(0,np.mean(times), f"avg: {np.mean(times)}")
    plt.xticks([])
    plt.ylabel("Duration / ticks")
    plt.xlabel("Episode, sorted by number goals encountered")
    plt.title("Evaluation durations")
    plt.show()

    print("Evaluation complete.")

    return states, actions, times, ts


def evaluate_ensemble(dqn, num_episodes, template_env, reset_options, env_name="Tokamak-v6", render=False):

    print("Evaluating...")

    if ("win" in sys.platform and render):
        print("Cannot render on windows...")
        render = False

    env = gym.make(env_name,
                   size=template_env.parameters["size"],
                   num_robots=template_env.parameters["num_robots"],
                   goal_locations=template_env.parameters["goal_locations"],
                   goal_probabilities=template_env.parameters["goal_probabilities"],
                   render_mode="human" if render else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation running on {device}.")

    times = []
    goal_resolutions = []

    state, info = env.reset(options=reset_options.copy())
    previous_status = info["status"]

    for i in range(num_episodes):
        state, info = env.reset(options=reset_options.copy())

        states = [state]
        actions = []
        state = torch.tensor(list(state.values()), dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            action = select_action(dqn, env, state, 0)
            observation, reward, terminated, truncated, info = env.step(action.item())

            # check if any robots have broken
            current_status = info["status"]  # check status here
            if current_status != previous_status:  # this will probably cause an error. check how to properly compare arrays

                # if (0 in current_status):  # this indicates that a robot is broken
                # get all parameters that will be needed to pass forwards
                parameters = env.get_parameters()
                num_active_robots = len(parameters["active_robots"])
                num_broken_robots = len(parameters["broken_robots"])

                # this approach will probably make robots lose their numbering...
                active_robot_locations = parameters["robot_locations"][parameters["active_robots"]]
                broken_robot_locations = parameters["robot_locations"][parameters["broken_robots"]]

                if (current_status == "recover"):  # recovery regime

                    # load the appropriate environment and DQN to continue the simulation
                    dqn.load_state_dict(torch.load(
                        os.getcwd() + "/trained_weights/" + f"{num_active_robots}-{num_broken_robots}recovery")
                    )

                    env = gym.make("SEAMSRecoveryTokamakEnv1",
                                   # normal intantiation parameters:
                                   size=parameters["size"],
                                   num_active_robots=num_active_robots,
                                   num_broken_robots=num_broken_robots,
                                   goal_locations=parameters["goal_locations"],
                                   goal_probabilities=parameters["goal_probabilities"],
                                   # extra parameters to set the initial state of the system:
                                   active_robot_locations=active_robot_locations,
                                   broken_robot_locations=broken_robot_locations,
                                   goal_instantiations=parameters["goal_instantiations"],
                                   goal_checked=parameters["goal_checked"],
                                   elapsed=parameters["elapsed"],
                                   robot_status=current_status
                                   )

                elif (current_status == "operate"):  # normal operation

                    # load the appropriate environment and DQN to continue the simulation
                    dqn.load_state_dict(torch.load(
                        os.getcwd() + "/trained_weights/" + f"{num_active_robots}operate")
                    )

                    env = gym.make("SEAMSOperationalTokamakEnv1",
                                   # normal intantiation parameters:
                                   size=parameters["size"],
                                   num_robots=parameters["num_robots"],
                                   goal_locations=parameters["goal_locations"],
                                   goal_probabilities=parameters["goal_probabilities"],
                                   # extra parameters to set the initial state of the system:
                                   robot_locations=parameters["robot_locations"],
                                   goal_instantiations=parameters["goal_instantiations"],
                                   goal_checked=parameters["goal_checked"],
                                   elapsed=parameters["elapsed"],
                                   robot_status=current_status
                                   )

            state = torch.tensor(list(observation.values()), dtype=torch.float32, device=device).unsqueeze(0)

            states.append(observation)
            actions.append(action)

            done = terminated or truncated

            if (done):
                times.append(info["elapsed_ticks"])
                goal_resolutions.append(np.sum(info["goal_resolutions"]))
                if (i % int(num_episodes / 10) == 0):
                    print(f"{i}/{num_episodes} episodes complete")
                break

    times = np.array(times)
    plt.figure(figsize=(10,10))
    times_start = 0
    # process 'times' into sub-arrays based on the unique entries in goal_resolutions
    unique_res = np.unique(goal_resolutions)
    for unique in unique_res:
        unique_times = times[goal_resolutions == unique]  # groups episodes with this unique number of tasks
        # plot the times. assign a range on x for each group based on the size of the group and where the last group ended.
        plt.plot(np.array(range(len(unique_times))) + times_start,
                 unique_times,
                 ls="",
                 marker="o",
                 label="{} goals - avg {:.2f}".format(int(unique), np.mean(unique_times)))
        times_start = len(unique_times) + times_start + num_episodes / 20

    plt.legend()
    plt.hlines(np.mean(times), 0, len(times) + len(unique_res) * num_episodes / 20, ls="--", color="grey")
    plt.text(0,np.mean(times), f"avg: {np.mean(times)}")
    plt.xticks([])
    plt.ylabel("Duration / ticks")
    plt.xlabel("Episode, sorted by number goals encountered")
    plt.title("Evaluation durations")
    plt.show()

    print("Evaluation complete.")

    return states, actions
