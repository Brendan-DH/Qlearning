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
from dqn.replay_memory import ReplayMemory
from dqn.priority_memory import PriorityMemory
from dqn.fingerprint_priority_memory import FingerprintPriorityMemory
from dqn.decay_functions import exponential_epsilon_decay
from dqn.optimisation import optimise
from dqn.evaluation import evaluate_model_by_trial_MA
import warnings
import sys


def train_model(
    env,  # gymnasium environment
    policy_net,  # policy network to be trained
    target_net,  # target network to be soft updated
    policy_net_gpu=None,
    num_episodes=1000,  # number of episodes for training
    optimisation_frequency=10,  # how often to optimise the model (in steps)
    gamma=0.6,  # discount factor
    epsilon_max=0.95,  # max exploration rate
    epsilon_min=0.05,  # min exploration rate
    fingerprint_window=100,
    fingerprint_mode="episode",
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
    sample_states = None,
    run_id=None,
):
    """
    For training a DQN on a gymnasium environment.

    Inputs:
        env,                            # gymnasium environment
        policy_net,                     # policy network to be trained
        target_net,                     # target network to be soft updated
        reset_options=None,             # optioreset_optionsns passed when resetting env
        num_episodes=1000,              # number of episodes for training
        optimisation_frequency=10,  # how often to optimise the model (in steps)
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
        cuda_enabled = False
        optimiser_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {optimiser_device} with type {optimiser_device.type}")
        if optimiser_device.type == "cuda":
            print(f"Using GPU {torch.cuda.get_device_name(optimiser_device)} for training.")
            if policy_net_gpu is not None:
                cuda_enabled = True
                policy_net_gpu.to(optimiser_device)
                print(f"Sent GPU policy network to {optimiser_device}.")
            else:
                print("WARNING: No GPU policy network provided, using CPU policy network for training.")

    if cuda_enabled:
        target_net.to(optimiser_device)
        print(f"Target network moved to {optimiser_device}.")

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
    if not cuda_enabled:
        print("Creating optimiser for CPU network.")
        optimiser = optim.AdamW(policy_net.parameters(), lr=alpha, amsgrad=True)
    else:
        print("Creating optimiser for GPU network.")
        optimiser = optim.AdamW(policy_net_gpu.parameters(), lr=alpha, amsgrad=True)

    # memory = ReplayMemory(buffer_size)
    memory = FingerprintPriorityMemory(buffer_size, fingerprint_window)
    torch.set_grad_enabled(True)
    plotting_on = plot_frequency < num_episodes and plot_frequency != 0
    checkpoints_on = checkpoint_frequency < num_episodes and checkpoint_frequency != 0
    if checkpoints_on:
        file = open(os.getcwd() + "/outputs/diagnostics", "w")
        file.write("# no data yet...")

    _ = env.reset()
        
    gamma_tensor = torch.tensor(gamma, device=optimiser_device)

    # Initialise some hyperparameters
    # if no decay function is supplied, set it to a default exponential decay
    if epsilon_decay_function is None:
        decay_rate = np.log(100 * (epsilon_max - epsilon_min)) / (num_episodes)  # ensures epsilon ~= epsilon_min at end

        def epsilon_decay_function(ep, e_max, e_min, num_eps):
            return exponential_epsilon_decay(
                episode=ep,
                epsilon_max=e_max,
                epsilon_min=e_min,
                max_epsilon_time=0,
                min_epsilon_time=0,
                decay_rate=decay_rate,
                num_episodes=num_eps,
            )

    if not max_steps:
        max_steps = np.inf  # remember: ultimately defined by the gym environment

    # Loop over training episodes
    start_time = time.time()
    total_optimisation_time = 0
    period_start_time = time.time()  # start the timer for this episode
    period_optimisation_time = 0  # time spent optimising this episode
    optimisation_counter = 0 # the total number of optimisations, used for the fingerprinting

    print(f"Using fingerprint mode '{fingerprint_mode}'.")

    for i_episode in range(num_episodes):
        # sort out memory
        gc.collect()
        torch.cuda.empty_cache()

        epsilon = epsilon_decay_function(i_episode, epsilon_max, epsilon_min, num_episodes)
        if (i_episode % plot_frequency) == 0:
            print(f"{i_episode}/{num_episodes} complete, epsilon = {epsilon}")

        # calculate the new epsilon
        if plotting_on or checkpoints_on:
            epsilons[i_episode] = epsilon

        obs_state, info = env.reset()
        match fingerprint_mode:
            case "optimisation_counter":
                fingerprint = optimisation_counter
            case "epsilon":
                fingerprint = epsilon
            case "episode":
                fingerprint = i_episode
            case _:
                print("No fingerprint mode chosen. Exiting.")
                sys.exit(1)
                
            
        obs_state["fingerprint"] = fingerprint

        # Initialise the first state
        if use_pseudorewards:
            phi_sprime = env.unwrapped.pseudoreward_function(env, env.unwrapped.state_tensor)  # phi_sprime is the pseudoreward of the new state
        ep_reward = 0
        ep_loss = 0

        # Navigate the environment
        # recent_states.appendleft(str(obs_state.values()))
        rel_actions = [
            "move cc",
            "move_cw",
            "engage",
            "wait",
        ]  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait

        latest_observations = np.empty(env.unwrapped.num_robots, dtype=dict)
        latest_actions = np.empty(env.unwrapped.num_robots, dtype=int)
        latest_rewards = np.zeros(env.unwrapped.num_robots, dtype=float)
        latest_blocked_actions = torch.zeros((env.unwrapped.num_robots, env.action_space.n), dtype=bool)

        latest_env_states = np.empty(env.unwrapped.num_robots, dtype=dict)


        for t in count():
            robot_no = env.unwrapped.state_dict["clock"]

            obs_state = env.unwrapped.get_obs()
            
            match fingerprint_mode:
                case "optimisation_counter":
                    fingerprint = optimisation_counter
                case "epsilon":
                    fingerprint = epsilon
                case "episode":
                    fingerprint = i_episode
                case _:
                    print("No fingerprint mode chosen. Exiting.")
                    sys.exit(1)
                    
            
            obs_state["fingerprint"] = fingerprint

            # to resolve this robot's PREVIOUS action, we see how the system has now changed:
            if t > env.unwrapped.num_robots:  
                prev_trans_s = latest_observations[robot_no].copy()
                fingerprint = prev_trans_s["fingerprint"]
                prev_trans_a = latest_actions[robot_no]
                prev_trans_sprime = obs_state.copy()
                prev_trans_sprime["fingerprint"] = fingerprint # fingerprint stays constant for stability
                prev_trans_r = (1 - reward_sharing_coefficient) * latest_rewards[robot_no] + reward_sharing_coefficient * (np.sum(latest_rewards) - latest_rewards[robot_no])
                prev_blocked_actions = latest_blocked_actions[robot_no].clone()

                if "priority" in memory.memory_type:
                    memory.push(
                        torch.tensor(
                            list(prev_trans_s.values()),
                            dtype=torch.float,
                            device="cpu",
                            requires_grad=False,
                        ),
                        prev_trans_a,
                        torch.tensor(
                            list(prev_trans_sprime.values()),
                            dtype=torch.float,
                            device="cpu",
                            requires_grad=False,
                        ),
                        prev_trans_r,
                        prev_blocked_actions,
                        fingerprint,
                    )
                else:
                    memory.push(
                        torch.tensor(
                            list(prev_trans_s.values()),
                            dtype=torch.float,
                            device="cpu",
                            requires_grad=False,
                        ),
                        prev_trans_a,
                        torch.tensor(
                            list(prev_trans_sprime.values()),
                            dtype=torch.float,
                            device="cpu",
                            requires_grad=False,
                        ),
                        prev_trans_r,
                        prev_blocked_actions,
                    )

            obs_tensor = torch.tensor(
                list(obs_state.values()),
                dtype=torch.float,
                device="cpu",
                requires_grad=False,
            )
            action_utilities = policy_net.forward(obs_tensor.unsqueeze(0))[0]  # why is this indexed?
            blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_dict, robot_no)

            if torch.all(blocked):
                print("WARNING: all actions were blocked. Continuing to next episode.")
                print(f"Offending state: {obs_state}")
                break

            if np.random.random() < epsilon:
                sample = env.action_space.sample()
                while blocked[sample] == 1:
                    sample = env.action_space.sample()
                action = sample
            else:
                action_utilities[blocked == 1] = -np.inf
                action = torch.argmax(action_utilities).item()
                # print(action_utilities, action)

            # action = torch.argmax(action_utilities).item()

            # apply action to environment
            old_env_state = env.unwrapped.state_dict.copy()
            new_obs_state, reward, terminated, truncated, info = env.step(action)
            
            match fingerprint_mode:
                case "optimisation_counter":
                    fingerprint = optimisation_counter
                case "epsilon":
                    fingerprint = epsilon
                case "episode":
                    fingerprint = i_episode
                case _:
                    print("No fingerprint mode chosen. Exiting.")
                    sys.exit(1)
                    
            new_obs_state["fingerprint"] = fingerprint 

            # create a hanging transition
            latest_observations[robot_no] = obs_state.copy()  # i.e. the thing this robot based its action on
            latest_rewards[robot_no] = reward
            latest_actions[robot_no] = action
            latest_blocked_actions[robot_no] = blocked.clone()

            latest_env_states[robot_no] = old_env_state

            # calculate pseudoreward
            if use_pseudorewards:
                phi = phi_sprime
                phi_sprime = env.unwrapped.pseudoreward_function(env, env.unwrapped.state_tensor)
                pseudoreward = gamma * phi_sprime - phi
            else:
                pseudoreward = 0

            # calculate reward
            reward = reward + pseudoreward  # torch.tensor([reward + pseudoreward], device=device, dtype=torch.float32)
            ep_reward += reward

            # work out if the run is over
            done = terminated or truncated or (t > max_steps)
            if terminated:
                # only adding for the last robot. the others shouldn't interpret their personal states as terminal.
                prev_trans_s = latest_observations[robot_no].copy()
                prev_trans_a = latest_actions[robot_no]
                prev_trans_r = (1 - reward_sharing_coefficient) * latest_rewards[robot_no] + reward_sharing_coefficient * (np.sum(latest_rewards) - latest_rewards[robot_no])
                prev_blocked_actions = latest_blocked_actions[robot_no].clone()
                fingerprint = prev_trans_s["fingerprint"]

                if prev_blocked_actions[prev_trans_a] == 1:
                    print(f"WARNING: action {rel_actions} was blocked for robot {robot_no}!!!!!!!!!!!!!!!!!!!")
                    print(
                        f"Robot {robot_no}",
                        f"action {prev_trans_a}",
                        f"locations: {prev_trans_s['my location']}",
                        f"{prev_trans_s['teammate1 location']}",
                        f"{prev_trans_s['teammate2 location']}",
                        f"new location: {prev_trans_sprime['my location']}",
                        f"prev_blocked_actions: {prev_blocked_actions}",
                    )

                if "priority" in memory.memory_type:
                    memory.push(
                        torch.tensor(
                            list(prev_trans_s.values()),
                            dtype=torch.float,
                            device="cpu",
                            requires_grad=False,
                        ),
                        prev_trans_a,
                        None,  # set to none for masking purposes in optimiser.
                        prev_trans_r,  # reward is still collected for getting here.
                        prev_blocked_actions,
                        fingerprint,
                    )
                else:
                    memory.push(
                        torch.tensor(
                            list(prev_trans_s.values()),
                            dtype=torch.float,
                            device="cpu",
                            requires_grad=False,
                        ),
                        prev_trans_a,
                        None,  # set to none for masking purposes in optimiser.
                        prev_trans_r,  # reward is still collected for getting here.
                        prev_blocked_actions,
                    )

            if (t % int(memory_sort_frequency) == 0) and "priority" in memory.memory_type:
                memory.sort(batch_size, priority_coefficient, i_episode)

            # run optimiser
            if t % optimisation_frequency == 0:
                optimisation_counter += 1
                op_start_time = time.time()
                loss = optimise(
                    policy_net if not cuda_enabled else policy_net_gpu,
                    target_net,
                    memory,
                    optimiser,
                    optimiser_device,
                    gamma_tensor,
                    batch_size,
                    priority_coefficient,
                    weighting_coefficient,
                )
                ep_loss += loss if loss is not None else 0
                period_optimisation_time += time.time() - op_start_time
                # print("time to optimise: ", time.time() - op_start_time)
                # print("period_optimisation_time", period_optimisation_time)

                # Soft-update the target net -- doing this in-place for better efficiency
                if cuda_enabled:
                    # Update the target network (on GPU) from the GPU policy net
                    with torch.no_grad():
                        for target_param_tensor, policy_param_tensor in zip(target_net.parameters(), policy_net_gpu.parameters()):
                            target_param_tensor.mul_(1 - tau).add_(policy_param_tensor, alpha=tau)  # in-place update for better efficiency

                    # Also update the CPU policy net from the GPU policy net
                    policy_net.load_state_dict(policy_net_gpu.state_dict())

                else:
                    # Update the target net (on cpu) from the CPU policy net
                    with torch.no_grad():
                        for target_param_tensor, policy_param_tensor in zip(target_net.parameters(), policy_net.parameters()):
                            target_param_tensor.mul_(1 - tau).add_(policy_param_tensor, alpha=tau)  # in-place update for better efficiency
            # if done, process data and make plots
            if done:
                if plotting_on or checkpoints_on:
                    losses[i_episode] = ep_loss
                    episode_durations[i_episode] = info["elapsed steps"]
                    rewards[i_episode] = ep_reward
                if plotting_on and i_episode % plot_frequency == 0 and i_episode > 0:
                    period_time = time.time() - period_start_time
                    print(f"Optimisation took {period_optimisation_time:.2f}s out of {period_time:.2f}s since last plot ({(period_optimisation_time / period_time) * 100:.2f}%).")
                    period_start_time = time.time()
                    total_optimisation_time += period_optimisation_time
                    period_optimisation_time = 0
                    f = plot_status(
                        episode_durations[:i_episode],
                        rewards[:i_episode],
                        epsilons[:i_episode],
                        losses[:i_episode],
                    )
                    file_dir = os.getcwd() + f"/outputs/plots/plt_{run_id}.png"
                    print(f"Saving plot {i_episode} at {file_dir}")
                    f.savefig(file_dir)
                    plt.close(f)
                if checkpoints_on and i_episode % checkpoint_frequency == 0 and i_episode > 0:
                    # write durations, rewards and epsilons to file
                    np.savetxt(
                        os.getcwd() + "/outputs/diagnostics",
                        np.vstack((episode_durations, rewards, epsilons)).transpose(),
                    )
                    torch.save(
                        policy_net.state_dict() if not cuda_enabled else policy_net_gpu.state_dict(),
                        os.getcwd() + f"/outputs/checkpoints/policy_weights_epoch{i_episode}",
                    )
                break
        # if i_episode > memory_sort_frequency: print(f"Total time for optimisation this episode: {optimisation_time * 1000:.3f}ms")
          

    training_time = time.time() - start_time
    print(f"Training complete in {int(training_time)} seconds, of which {int(total_optimisation_time)}s ({(total_optimisation_time / training_time) * 100:.2f}%) was spent on optimisation.")
    
    if cuda_enabled:
        policy_net.load_state_dict(policy_net_gpu.state_dict())
        
    print("TOTAL OPTIMISATION STEPS: ", optimisation_counter)
    
    return policy_net, episode_durations, rewards, epsilons
