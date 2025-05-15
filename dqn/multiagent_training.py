import gc
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import json

import torch
import torch.optim as optim
import time
import os
from collections import deque

from dqn.plotting import plot_status
from dqn.dqn_collections import FiniteDict
from dqn.memory import PriorityMemory
from dqn.decay_functions import exponential_epsilon_decay
from dqn.optimisation import optimise_model_with_importance_sampling
import warnings
import math


def train_model(
        env,  # gymnasium environment
        policy_net,  # policy network to be trained
        target_net,  # target network to be soft updated
        reset_options=None,  # options passed when resetting env
        num_episodes=1000,  # number of episodes for training
        gamma=0.6,  # discount factor
        epsilon_max=0.95,  # max exploration rate
        epsilon_min=0.05,  # min exploration rate
        epsilon_decay_function=None,  # will be exponential if not set.
        alpha=1e-3,  # learning rate for policy DeepQNetwork
        tau=0.005,  # soft update rate for ap DeepQNetwork
        use_pseudorewards=True,  # whether to calculate and use pseudorewards
        max_steps=None,  # max steps per episode
        batch_size=128,  # batch size of the replay memory
        buffer_size=10000,  # total size of replay memory buffer
        plot_frequency=10,  # number of episodes between status plots (0=disabled)
        checkpoint_frequency=0,  # number of episodes between saving weights (0=disabled)
        memory_sort_frequency=100,  # number of episodes between sorting the replay memory
        priority_coefficient=0.5,  # alpha in the sampling probability equation, higher prioritises importance more
        weighting_coefficient=0.7,  # beta in the transition weighting equation, higher ameliorates sampling bias more
        reward_sharing_coefficient=0.1,  # determines how much reward each robot gets from teammates' actions
        run_id=None
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
        use_pseudorewards=True,          # whether to calculate and use pseudorewards
        max_steps=None,                 # max steps per episode
        batch_size=128,                 # batch size of the replay memory
        buffer_size=10000               # total size of replay memory buffer
        state_tree_capacity = 200,      # capacity of the state tree
        tree_prune_frequency=10,        # number of episodes between pruning the state tree
        plot_frequency=10,              # number of episodes between status plots (0=disabled)
        checkpoint_frequency=0,         # number of episodes between saving weights (0=disabled)
        memory_sort_frequency=100,      # number of episodes between sorting the replay memory
        priority_coefficient=0.5,       # alpha in the sampling probability equation, higher prioritises importance more
        weighting_coefficient=0.7       # beta in the transition weighting equation, higher ameliorates sampling bias more

    Outputs:
        (None)
    """

    # store values for plotting
    print("reward_sharing_coefficient", reward_sharing_coefficient)

    epsilons = np.empty(num_episodes)
    episode_durations = np.empty(num_episodes)
    losses = np.empty(num_episodes)
    rewards = np.empty(num_episodes)
    with warnings.catch_warnings(action="ignore"):
        optimiser_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.to(torch.device("cpu"))
    target_net.to(torch.device("cpu"))

    # with open(os.getcwd() + f"/outputs/env_desc_{run_id}.txt", "w") as file:
    #     json.dump(env.unwrapped.state, file)

    print(f"""
        Commensing training.
        Optimisation Device: {optimiser_device}
        Environment: {env.unwrapped.spec.id}
        Run Id: {run_id}
        ----
        """)

    # Initialisation of NN apparatus
    optimiser = optim.AdamW(policy_net.parameters(), lr=alpha, amsgrad=True)

    memory = PriorityMemory(buffer_size)
    torch.set_grad_enabled(True)
    plotting_on = plot_frequency < num_episodes and plot_frequency != 0
    checkpoints_on = checkpoint_frequency < num_episodes and checkpoint_frequency != 0
    if (checkpoints_on):
        file = open(os.getcwd() + "/outputs/diagnostics", "w")
        file.write("# no data yet...")

    _ = env.reset()  # reset to init

    gamma_tensor = torch.tensor(gamma, device=optimiser_device)

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

    # Loop over training episodes
    start_time = time.time()
    obs_visits = FiniteDict(max_size=10000)

    for i_episode in range(num_episodes):
        optimisation_time = 0

        # sort out memory
        gc.collect()
        torch.cuda.empty_cache()

        base_epsilon = epsilon_decay_function(i_episode, epsilon_max, epsilon_min, num_episodes)
        epsilon = base_epsilon
        if ((i_episode % plot_frequency) == 0):
            print(f"{i_episode}/{num_episodes} complete, epsilon = {base_epsilon}")
            if (torch.cuda.is_available()): print(f"CUDA memory summary:\n{torch.cuda.memory_summary(device='cuda')}")

        if (i_episode % int(memory_sort_frequency) == 0):
            memory.sort(batch_size, priority_coefficient)

        # calculate the new epsilon
        if (plotting_on or checkpoints_on):
            epsilons[i_episode] = base_epsilon

        obs_state, info = env.reset()
        obs_state["epsilon"] = epsilon
        obs_state["episode"] = i_episode

        # Initialise the first state
        if (use_pseudorewards):
            phi_sprime = env.unwrapped.pseudoreward_function(env, env.unwrapped.state_tensor)  # phi_sprime is the pseudoreward of the new state
        ep_reward = 0

        # Navigate the environment
        recent_state_capacity = 5
        recent_states = deque([], maxlen=recent_state_capacity)
        # recent_states.appendleft(str(obs_state.values()))
        rel_actions = ["move cc", "move_cw", "engage", "wait"]  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait

        latest_observations = np.empty(env.unwrapped.num_robots, dtype=dict)
        latest_actions = np.zeros(env.unwrapped.num_robots, dtype=int)
        latest_rewards = np.zeros(env.unwrapped.num_robots, dtype=float)

        latest_env_states = np.empty(env.unwrapped.num_robots, dtype=dict)

        nlc = "\n"

        for t in count():

            robot_no = env.unwrapped.clock

            obs_state = env.unwrapped.get_obs()
            obs_state["episode"] = i_episode
            obs_state["epsilon"] = epsilon

            # to resolve this robot's PREVIOUS action, we see how the system has now changed:
            if (t > env.unwrapped.num_robots):  # does this make sense?
                prev_trans_s = latest_observations[robot_no]
                prev_trans_a = latest_actions[robot_no]
                prev_trans_sprime = obs_state
                prev_trans_r = (1-reward_sharing_coefficient) * latest_rewards[robot_no] + reward_sharing_coefficient * (np.sum(latest_rewards) - latest_rewards[robot_no])
                memory.push(torch.tensor(list(prev_trans_s.values()), dtype=torch.float, device="cpu", requires_grad=False),
                            prev_trans_a,
                            torch.tensor(list(prev_trans_sprime.values()), dtype=torch.float, device="cpu", requires_grad=False),
                            prev_trans_r)

                robot_locations = np.empty(env.unwrapped.num_robots)
                robot_locations[robot_no] = obs_state["my location"]
                robot_locations[(robot_no + 1) % env.unwrapped.num_robots] = obs_state["teammate1 location"]
                robot_locations[(robot_no + 2) % env.unwrapped.num_robots] = obs_state["teammate2 location"]

                prev_env_state = latest_env_states[robot_no]

                # env_state_diffs = {k: (f"{prev_env_state[k]} --> {new_env_state[k]}") for k in set(prev_env_state) | set(new_env_state) if prev_env_state[k] != new_env_state[k]}
                # obs_diffs = {k: (f"{prev_trans_s[k]} --> {prev_trans_sprime[k]}") for k in set(prev_trans_s) | set(prev_trans_sprime) if prev_trans_s[k] != prev_trans_sprime[k]}
                #
                # print(f"""
                # #####
                # Rewards ({i_episode}/{epsilon})
                # ----
                # robot:{robot_no}
                # obs diffs: {str(obs_diffs).replace(",", nlc)}
                # env state diffs: {str(env_state_diffs).replace(",", nlc)}
                # action: {prev_trans_a}
                # reward: {latest_rewards[robot_no]} + {0.2 * (np.sum(latest_rewards) - latest_rewards[robot_no])} = {prev_trans_r}
                # ----
                # All robot rewards: {latest_rewards}
                # All robot locations: {robot_locations}
                #
                #
                # """)

            obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)
            action_utilities = policy_net.forward(obs_tensor.unsqueeze(0))[0]  # why is this indexed?

            blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_dict, env.unwrapped.clock)
            if (torch.all(blocked)):
                print("WARNING: all actions were blocked. Continuing to next episode.")
                print(f"Offending state: {obs_state}")
                break

            if (np.random.random() < epsilon):
                sample = env.action_space.sample()
                while blocked[sample] == 1:
                    sample = env.action_space.sample()
                action = sample
            else:
                action_utilities = torch.where(blocked, -1000, action_utilities)  # can mask these strongly as they aren't used for backprop
                action = torch.argmax(action_utilities).item()

            # action = torch.argmax(action_utilities).item()

            # apply action to environment
            old_env_state = env.unwrapped.state_dict.copy()
            new_obs_state, reward, terminated, truncated, info = env.step(action)
            new_env_state = env.unwrapped.state_dict.copy()
            new_obs_state["epsilon"] = epsilon
            new_obs_state["episode"] = i_episode

            # now this needs to be the resultant state for the NEXT robot in the order.
            # the NEXT robot also needs to get its rewards for its previous decision, and a weighted-down reward from the intermediate robot
            latest_observations[robot_no] = obs_state  # i.e. the thing this robot based its action on
            latest_rewards[robot_no] = reward
            latest_actions[robot_no] = action

            latest_env_states[robot_no] = old_env_state

            if (epsilon < 0.01):
                print(f"Action: {rel_actions[action % env.unwrapped.num_actions]} on robot {math.floor(action / env.unwrapped.num_actions)}\nReward: {reward}\nStep: {t}")

            # calculate pseudoreward
            if (use_pseudorewards):
                phi = phi_sprime
                phi_sprime = env.unwrapped.pseudoreward_function(env, env.unwrapped.state_tensor)
                pseudoreward = (gamma * phi_sprime - phi)
            else:
                pseudoreward = 0

            # calculate reward
            reward = reward + pseudoreward  # torch.tensor([reward + pseudoreward], device=device, dtype=torch.float32)
            ep_reward += reward

            # work out if the run is over
            done = terminated or truncated or (t > max_steps)
            if terminated:
                for robot_index in range(env.unwrapped.num_robots):
                    prev_trans_s = latest_observations[robot_index]
                    prev_trans_a = latest_actions[robot_index]
                    prev_trans_r = (1 - reward_sharing_coefficient) * latest_rewards[robot_index] + reward_sharing_coefficient * (np.sum(latest_rewards) - latest_rewards[robot_index])
                    memory.push(
                        torch.tensor(list(prev_trans_s.values()), dtype=torch.float, device="cpu", requires_grad=False),
                        prev_trans_a,
                        None,  # set to none for masking purposes in optimiser.
                        prev_trans_r  # reward is still collected for getting here.
                    )
                    # env_state_diffs = {k: (f"{prev_env_state[k]} --> {new_env_state[k]}") for k in set(prev_env_state) | set(new_env_state) if prev_env_state[k] != new_env_state[k]}
                    # obs_diffs = {k: (f"{prev_trans_s[k]} --> {prev_trans_sprime[k]}") for k in set(prev_trans_s) | set(prev_trans_sprime) if prev_trans_s[k] != prev_trans_sprime[k]}

                    # print(f"""
                    # #####
                    # Rewards ({i_episode}/{epsilon})
                    # ----
                    # robot:{robot_index}
                    # origin state: {prev_trans_s}
                    # resultant state: None (terminated)
                    # action: {prev_trans_a}
                    # reward: {latest_rewards[robot_index]} + {reward_sharing_coefficient * (np.sum(latest_rewards) - latest_rewards[robot_index])} = {prev_trans_r}
                    # ----
                    # All robot rewards: {latest_rewards}
                    # All robot locations: {robot_locations}
                    #
                    #
                    # """)
                # WORK OUT IF THIS IS ACTUALLY OPERATING CORRECTLY
                # does it make sense for all robots to have their final transitions resolved?

            # run optimiser
            timer_start = time.time()
            loss = optimise_model_with_importance_sampling(policy_net,
                                                           target_net,
                                                           memory,
                                                           optimiser,
                                                           optimiser_device,
                                                           gamma_tensor,
                                                           batch_size,
                                                           priority_coefficient,
                                                           weighting_coefficient)
            optimisation_time += time.time() - timer_start

            # Soft-update the target net -- doing this in-place for better efficiency
            with torch.no_grad():
                for target_param_tensor, policy_param_tensor in zip(target_net.parameters(), policy_net.parameters()):
                    target_param_tensor.mul_(1 - tau).add_(policy_param_tensor, alpha=tau)  # in-place update for better efficiency

            # if done, process data and make plots
            if done:
                if (plotting_on or checkpoints_on):
                    losses[i_episode] = loss if loss is not None else 0
                    episode_durations[i_episode] = info["elapsed steps"]
                    rewards[i_episode] = ep_reward
                if (plotting_on and i_episode % plot_frequency == 0 and i_episode > 0):
                    f = plot_status(episode_durations[:i_episode], rewards[:i_episode], epsilons[:i_episode], losses[:i_episode])
                    file_dir = os.getcwd() + f"/outputs/plots/plt_epoch{i_episode}_{run_id}.png"
                    print(f"Saving plot {i_episode} at {file_dir}")
                    f.savefig(file_dir)
                    plt.close(f)
                if (checkpoints_on and i_episode % checkpoint_frequency == 0 and i_episode > 0):
                    # write durations, rewards and epsilons to file
                    np.savetxt(os.getcwd() + "/outputs/diagnostics",
                               np.vstack((episode_durations, rewards, epsilons)).transpose())
                    torch.save(policy_net.state_dict(), os.getcwd() + f"/outputs/checkpoints/policy_weights_epoch{i_episode}")
                break
        if i_episode > memory_sort_frequency: print(f"Total time for optimisation this episode: {optimisation_time * 1000:.3f}ms")

    print(f"Training complete in {int(time.time() - start_time)} seconds.")
    return policy_net, episode_durations, rewards, epsilons
