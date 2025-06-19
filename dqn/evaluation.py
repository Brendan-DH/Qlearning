from itertools import count
import numpy as np
import math
import os

import torch
import sys
from collections import deque, OrderedDict
from dqn.dqn import DeepQNetwork
from queue import Queue
import time


def evaluate_model_by_trial(dqn, num_episodes, env, max_steps, reset_options=None, render=False):
    print("Evaluating...")

    if "win" in sys.platform and render:
        print("Cannot render on windows...")
        render = False

    env.unwrapped.set_rendering(render)
    device = torch.device("cpu")
    print(f"Evaluation running on {device}.")

    steps = np.empty(num_episodes)
    deadlock_counter = 0
    deadlock_traces = deque([], maxlen=10)  # store last 10 deadlock traces

    for i in range(num_episodes):
        obs_state, info = env.reset()

        states = [env.unwrapped.state_dict]
        actions = []
        obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)
        rel_actions = ["move cc", "move_cw", "engage", "wait"]  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait

        for t in count():
            # calculate action utilities and choose action
            action_utilities = dqn.forward(obs_tensor.unsqueeze(0))[0]  # why is this indexed?
            # block:
            blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_dict, env.unwrapped.state_dict["clock"])
            action_utilities[blocked == 1] = -np.inf
            print(env.unwrapped.state_dict["clock"], ": ", action_utilities)
            action = torch.argmax(action_utilities).item()

            # apply action to environment
            new_obs_state, reward, terminated, truncated, info = env.step(action)

            states.append(env.unwrapped.state_dict)

            actions.append(action)

            obs_state = new_obs_state
            obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

            done = terminated

            if render:
                env.render_frame(states[-1], True)

            if done or truncated or t > max_steps:
                if int(num_episodes / 10) > 0 and i % int(num_episodes / 10) == 0:
                    print(f"{i}/{num_episodes} episodes complete")
                break

        if not done:
            deadlock_traces.append(states)
            deadlock_counter += 1

        steps[i] = t

    print("Evaluation complete.")
    print(f"{'CONVERGENCE SUCCESSFUL' if deadlock_counter == 0 else 'FAILURE'} - Failed to complete {deadlock_counter} times")
    print(f"Percentage converged: {100 - (deadlock_counter * 100 / num_episodes)}")

    return states, actions, steps, deadlock_traces  # states, actions, ticks, steps


def evaluate_model_by_trial_MA(dqn, num_episodes, env, max_steps, render=False, render_deadlocks=False):
    
    print("Evaluating...")

    if "win" in sys.platform and render:
        print("Cannot render on windows...")
        render = False

    env.unwrapped.set_rendering(render)
    device = torch.device("cpu")
    print(f"Evaluation running on {device}.")

    steps = np.empty(num_episodes)
    deadlock_counter = 0
    broken_deadlock_counter = 0
    deadlock_traces = deque([], maxlen=100)  # store last 1000 deadlock traces

    canonical_fingerprint = 0  # epsilon for multiagent evaluation

    for i in range(num_episodes):
        obs_state, info = env.reset()
        obs_state["fingerprint"] = canonical_fingerprint

        states = [env.unwrapped.state_dict]
        actions = []
        obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

        deadlock_breaker = [False] * env.unwrapped.num_robots  # deadlock breaker for each robot
        last_origin_observations = np.empty(3, dtype=dict)  # last observation for each robot

        for t in count():
            robot_no = env.unwrapped.state_dict["clock"]

            if deadlock_breaker[robot_no]:
                broken_deadlock_counter += 1
                action = 3  # wait if no change in observation
                deadlock_breaker = [False] * env.unwrapped.num_robots

            else:
                # calculate action utilities and choose action
                action_utilities = dqn.forward(obs_tensor.unsqueeze(0))[0]
                blocked = env.unwrapped.blocked_model(env, env.unwrapped.state_dict, env.unwrapped.state_dict["clock"])
                action_utilities = torch.where(blocked, -np.inf, action_utilities)
                if render:
                    print(robot_no, action_utilities)
                action = torch.argmax(action_utilities).item()

            # apply action to environment
            new_obs_state, reward, terminated, truncated, info = env.step(action)
            new_obs_state["fingerprint"] = canonical_fingerprint
            # new_obs_state["episode"] = canonical_episode

            states.append(env.unwrapped.state_dict)
            if t > env.unwrapped.num_robots and states[-1] in states[-(2 * env.unwrapped.num_robots) - 1 : -1] and action != 2 and action != 3:
                # deadlock_breaker[robot_no] = True
                pass

            actions.append(action)

            last_origin_observations[robot_no] = obs_state.copy()  # store the last observation for this robot
            obs_state = new_obs_state.copy()
            obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

            done = terminated

            if done or truncated or t > max_steps:
                if int(num_episodes / 10) > 0 and i % int(num_episodes / 10) == 0:
                    print(f"{i}/{num_episodes} episodes complete")
                break

        if not done:
            deadlock_traces.append(states)
            deadlock_counter += 1

        steps[i] = t

    print("Evaluation complete.")
    print(f"{'CONVERGENCE SUCCESSFUL' if deadlock_counter == 0 else 'FAILURE'} - Failed to complete {deadlock_counter} times")
    print(f"Percentage converged: {100 - (deadlock_counter * 100 / num_episodes)}")
    print(f"Deadlock breaker triggered {broken_deadlock_counter} times.")

    if render_deadlocks and len(deadlock_traces) > 0:
        env.unwrapped.set_rendering(True)  # stop rendering after evaluation
        for trace in deadlock_traces:
            print("Deadlock trace:")
            for r, state in enumerate(trace):
                obs = env.unwrapped.state_dict_to_observable(state, state["clock"])
                obs["epsilon"] = canonical_fingerprint
                print(dqn.forward(torch.tensor(list(obs.values()), dtype=torch.float, device="cpu", requires_grad=False).unsqueeze(0)))
                env.unwrapped.render_frame(state, False)

    return states, actions, steps, deadlock_traces  # states, actions, ticks, steps


def generate_dtmc_file(weights_file, env, system_logic, output_name="dtmc", order = "LIFO", run_id=""):
    # load the DQN

    n_actions = env.action_space.n
    init_obs, info = env.reset()  # ask the DQN what action should be taken here
    init_obs["epsilon"] = 0  # set epsilon to 0 for exploration

    exploration_state_queue = deque()
    exploration_observation_queue = deque()

    exploration_state_queue.appendleft(env.unwrapped.state_dict)  # state dicts for full state description
    # print("States in exploration_state_queue:")
    # queue_states = list(exploration_state_queue.queue)
    # for idx, state in enumerate(queue_states):
    #     print(f"{idx}: {state}")
    exploration_observation_queue.appendleft(init_obs)  # observations for forward passes

    if not weights_file:
        print("No weights file specified, exiting.")
        sys.exit(1)

    try:
        loaded_weights = torch.load(weights_file)
        nodes_per_layer = len(loaded_weights["hidden_layers.0.weight"])
        num_hidden_layers = int((len(loaded_weights.keys()) - 4) / 2)  # -4 accounts for input and output weights and biases
        print(f"Loading policy from '{weights_file}'")
    except FileNotFoundError:
        print(f"Weights file {weights_file} not found, exiting.")
        sys.exit(1)

    n_observations = len(init_obs)
    policy_net = DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer)
    policy_net.load_state_dict(loaded_weights)

    new_id = 0  # an unencountered state will get this id, after which it will be incremented
    states_id_dict = {str(env.unwrapped.state_dict.values()): 0}  # dictionary of state dicts to id
    keySet = set(states_id_dict.keys())
    labels_set = {"0 init\n"}  # set of state labels ([id] [label] )

    new_id += 1
    transitions_array = []
    rewards_array = []
    clock = 0
    start_time = time.time()
    print("Beginning DTMC construction.")
    while not len(exploration_state_queue) == 0:
        # print("EXPLORATION STEP")
        display_string = f"{len((exploration_state_queue))} -- Total states encountered: {new_id + 1}"

        if (int(time.time() - start_time) % 100) == 0:
            print(f"\r[{int(time.time() - start_time)}s] States in exploration queue: {' ' * (50 - len(display_string))}{display_string}", end="")

        if order.lower() == "lifo":
            state_dict = exploration_state_queue.popleft()
            obs_state = exploration_observation_queue.popleft()
        elif order.lower() == "fifo":
            state_dict = exploration_state_queue.pop()
            obs_state = exploration_observation_queue.pop()
        else:
            print(f"Error: unknown order '{order}' for exploration queue, exiting.")
            sys.exit(1)
        obs_state["epsilon"] = 0  # set epsilon to 0 for exploration

        obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

        action_utilities = policy_net.forward(obs_tensor.unsqueeze(0))[0]
        blocked = env.unwrapped.blocked_model(env, state_dict, state_dict["clock"])
        action_utilities = torch.where(blocked, -np.inf, action_utilities)
        action = torch.argmax(action_utilities).item()

        # state_dict = env.unwrapped.state_vector_to_dict(state_vector)
        robot_no = state_dict["clock"]
        result = system_logic.t_model(env, state_dict, robot_no, action)  # get the result of the action from the transition model

        # label end states
        all_done = system_logic.state_is_final(env, state_dict)
        if all_done:
            labels_set.add(f"{states_id_dict[str(state_dict.values())]} done\n")  # label end states
            transitions_array.append(f"{states_id_dict[str(state_dict.values())]} {states_id_dict[str(state_dict.values())]} 1")  # end states loop to themselves (formality):
            continue  # continue as we don't care about other transitions from end states

        for i in range(len(result[0])):  # iterate over result states:
            prob = result[0][i]
            result_state_dict = result[1][i]
            # print(result_state_dict["clock"])

            # print(prob, result_state_dict["robot0 location"])

            if str(result_state_dict.values()) not in keySet:  # register newly discovered states
                states_id_dict[str(result_state_dict.values())] = new_id
                exploration_state_queue.append(result_state_dict)
                exploration_observation_queue.append(env.unwrapped.state_dict_to_observable(result_state_dict, result_state_dict["clock"]))
                new_id += 1
                keySet.add(str(result_state_dict.values()))

            if result_state_dict["clock"] == int(env.unwrapped.num_robots - 1):  # assign awards to clock ticks
                rewards_array.append(f"{states_id_dict[str(state_dict.values())]} {states_id_dict[str(result_state_dict.values())]} 1")

            # print("prob", prob, type(prob))
            transitions_array.append(f"{states_id_dict[str(state_dict.values())]} {states_id_dict[str(result_state_dict.values())]} {prob}")  # write the transitions into the file/array

        clock = (clock + 1) % env.unwrapped.num_robots  # increment clock for the next state

    print(f"\nWriting file to {os.getcwd()}/outputs/storm_files/{output_name}.tra, {output_name}.lab, {output_name}.transrew")

    f = open(os.getcwd() + f"/outputs/storm_files/{output_name}.tra", "w")  # create DTMC file .tra
    f.write("dtmc\n")
    for i in range(len(transitions_array)):
        f.write(transitions_array[i] + "\n")
    f.close()

    f = open(os.getcwd() + f"/outputs/storm_files/{output_name}.lab", "w")  # create labels file .lab
    f.write("""
    #DECLARATION
    init done
    #END
    """)
    labels_list = list(labels_set)
    labels_list.sort(key=lambda x: int(x.split()[0]))  # label file must list states in numerical order
    for i in range(len(labels_list)):
        f.write(labels_list[i])
    f.close()

    f = open(os.getcwd() + f"/outputs/storm_files/{output_name}.transrew", "w")  # rewards file .transrew
    for i in range(len(rewards_array)):
        f.write(rewards_array[i] + "\n")
    f.close()

    print(f"Saved policy DTMC as {output_name}.tra.")

    # check DTMC for invalid states
    p_problem_states, unacknowledged_states = check_dtmc(os.getcwd() + f"/outputs/storm_files/{output_name}.tra")

    if len(p_problem_states) == 0:
        print("Success: all probabilities sum to 1")
    else:
        print("Error! Some outgoing probabilities do not sum to 1\nstate | total p")
        for i in range(len(p_problem_states)):
            print(f"{p_problem_states[i][0]} | {p_problem_states[i][1]}")

    if len(unacknowledged_states) == 0:
        print("Success: all states included in transition structure")
    else:
        print("Error! Some encountered states have no outgoing transitions!\nStates:")
        for i in range(len(unacknowledged_states)):
            print(unacknowledged_states[i])


def generate_mdp_file(weights_file, env, system_logic, output_name="mdp"):
    # load the DQN

    n_actions = env.action_space.n
    obs_state, info = env.reset()
    n_observations = len(obs_state)

    init_obs, info = env.reset()  # ask the DQN what action should be taken here

    exploration_queue = Queue()
    exploration_observation_queue = Queue()

    state_vector = env.unwrapped.state_dict.values()

    exploration_queue.put(state_vector)  # tensors for full state description
    exploration_observation_queue.put(init_obs)  # observations for forward passes

    if not weights_file:
        print("No weights file specified, exiting.")
        sys.exit(1)

    try:
        loaded_weights = torch.load(weights_file)
        nodes_per_layer = len(loaded_weights["hidden_layers.0.weight"])
        num_hidden_layers = int((len(loaded_weights.keys()) - 4) / 2)  # -4 accounts for input and output weights and biases
        print(f"Loading policy from '{weights_file}'")
    except FileNotFoundError:
        print(f"Weights file {weights_file} not found, exiting.")
        sys.exit(1)

    policy_net = DeepQNetwork(n_observations, n_actions, num_hidden_layers, nodes_per_layer)
    policy_net.load_state_dict(loaded_weights)

    new_id = 0  # an unencountered state will get this id, after which it will be incremented
    states_id_dict = {str(env.unwrapped.state_dict.values()): 0}  # dictionary of state dicts to id
    labels_set = {"0 init\n"}  # set of state labels ([id] [label] )

    new_id += 1
    transitions_array = []
    rewards_array = []
    max_utility = 0
    extra_states_counter = 0

    while not exploration_queue.empty():
        print(f"\rStates in exploration queue: {' ' * (10 - len(str(exploration_queue.qsize())))}{exploration_queue.qsize()} (Total #decisions: {extra_states_counter})", end="")

        state_tensor = exploration_queue.get()
        obs_state = exploration_observation_queue.get()

        obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

        action_utilities = policy_net.forward(obs_tensor.unsqueeze(0))[0]
        blocked = env.unwrapped.blocked_model(env, state_tensor)
        action_utilities = torch.where(blocked, -np.inf, action_utilities)
        # action = torch.argmax(action_utilities).item()

        utilities, actions = action_utilities.topk(2)
        action1, action2 = actions[0].item(), actions[1].item()
        results = [system_logic.t_model(env, state_tensor, action1)]
        # max_utility = max(max_utility, utilities[0])
        if utilities[0] > max_utility:
            max_utility = utilities[0]
        if utilities[1] >= 0.97 * utilities[0] and utilities[1] > 0.75 * max_utility:
            extra_states_counter += 1
            results.append(system_logic.t_model(env, state_tensor, action2))

        # label end states
        all_done = system_logic.state_is_final(env, state_tensor)
        if all_done:
            labels_set.add(f"{states_id_dict[str(state_tensor)]} done\n")  # label end states
            transitions_array.append(f"{states_id_dict[str(state_tensor)]} 0 {states_id_dict[str(state_tensor)]} 1")  # end states loop to themselves (formality):
            continue  # continue as we don't care about other transitions from end states

        for a, result in enumerate(results):
            for i in range(len(result[0])):  # iterate over result states:
                prob = float(result[0][i].item()) if torch.is_tensor(result[0][i]) else result[0][i]
                result_state_tensor = result[1][i]
                result_state_dict = env.unwrapped.state_tensor_to_observable(result_state_tensor)

                if str(result_state_tensor) not in list(states_id_dict.keys()):  # register newly discovered states
                    states_id_dict[str(result_state_tensor)] = new_id
                    exploration_queue.put(result_state_tensor)
                    exploration_observation_queue.put(result_state_dict)
                    new_id += 1

                if np.sum([result_state_dict[f"robot{i} clock"] for i in range(env.unwrapped.num_robots)]) == 0:  # assign awards to clock ticks
                    if f"{states_id_dict[str(state_tensor)]} {states_id_dict[str(result_state_tensor)]} 1" not in rewards_array:
                        rewards_array.append(f"{states_id_dict[str(state_tensor)]} {a} {states_id_dict[str(result_state_tensor)]} 1")

                # print("prob", prob, type(prob))
                transitions_array.append(f"{states_id_dict[str(state_tensor)]} {a} {states_id_dict[str(result_state_tensor)]} {prob}")  # write the transitions into the file/array

    print(f"\nWriting file to {os.getcwd()}/outputs/storm_files/{output_name}.tra, {output_name}.lab, {output_name}.transrew")

    f = open(os.getcwd() + f"/outputs/storm_files/{output_name}.tra", "w")  # create DTMC file .tra
    f.write("mdp\n")
    for i in range(len(transitions_array)):
        f.write(transitions_array[i] + "\n")
    f.close()

    f = open(os.getcwd() + f"/outputs/storm_files/{output_name}.lab", "w")  # create labels file .lab
    f.write("""
    #DECLARATION
    init done
    #END
    """)
    labels_list = list(labels_set)
    labels_list.sort(key=lambda x: int(x.split()[0]))  # label file must list states in numerical order
    for i in range(len(labels_list)):
        f.write(labels_list[i])
    f.close()

    f = open(os.getcwd() + f"/outputs/storm_files/{output_name}.transrew", "w")  # rewards file .transrew
    for i in range(len(rewards_array)):
        f.write(rewards_array[i] + "\n")
    f.close()

    print(f"Saved policy MDP as {output_name}.tra.")

    # check DTMC for invalid states
    # p_problem_states, unacknowledged_states = check_dtmc(os.getcwd() + f"/outputs/storm_files/{output_name}.tra")
    #
    # if (len(p_problem_states) == 0):
    #     print("Success: all probabilities sum to 1")
    # else:
    #     print("Error! Some outgoing probabilities do not sum to 1\nstate | total p")
    #     for i in range(len(p_problem_states)):
    #         print(f"{p_problem_states[i][0]} | {p_problem_states[i][1]}")
    #
    # if (len(unacknowledged_states) == 0):
    #     print("Success: all states included in transition structure")
    # else:
    #     print("Error! Some encountered states have no outgoing transitions!\nStates:")
    #     for i in range(len(unacknowledged_states)):
    #         print(unacknowledged_states[i])
    #


def check_dtmc(filepath, verbose=False):
    print(f"Checking {filepath} for valid DTMC structure...")

    success = True
    p_outs = {}
    origin_states = set([])  # states mentioned in structure
    accessible_states = set([])  # states mentioned in structure

    accessible_states.add("0")  # add initial state
    no = 1
    print("Reading file...") if verbose else None
    with open(filepath) as f:
        header = f.readline()
        assert header.strip() == "dtmc"
        for line in f:
            no += 1
            s, s_prime, p = line.strip().split(" ")
            origin_states.add(s)
            accessible_states.add(s_prime)
            if s in p_outs.keys():
                p_outs[s] += float(p)
            else:
                p_outs[s] = float(p)

    out_states = list(p_outs.keys())
    out_probabilities = list(p_outs.values())
    p_problem_states = []
    for i in range(len(out_states)):
        if (i % 100 == 0) and verbose:
            print(f"\rChecking state transitions sum to 1: {i + 1}/{len(out_states)}", end="")
        if round(out_probabilities[i], 3) != 1.0:  # rounding is needed for numerical issues
            if verbose:
                print(f"Error! s={out_states[i]} -> total p={out_probabilities[i]}")
            p_problem_states.append([out_states[i], out_probabilities[i]])

    if len(p_problem_states) == 0:
        print("\nAll state transitions sum to 1.") if verbose else None
    else:
        success = False

    print("Checking for inaccesible states...") if verbose else None
    inaccessible_states = set.difference(origin_states, accessible_states)  # states that are not mentioned in the structure
    if len(inaccessible_states) > 0:
        print(f"Error! {len(inaccessible_states)} inaccessible states found: {inaccessible_states}") if verbose else None
        success = False
    else:
        print("All states are accessible.") if verbose else None

    # for i, s in enumerate(accessible_states):
    #     if (i % 100 == 0) and verbose:
    #         print(f"\rChecking states are accessible: {i+1}/{len(accessible_states)}", end="")
    #     if s not in out_states:
    #         if (verbose):
    #             print(f"Error! s={s} has no outgoing transitions!")
    #         unacknowledged_states.append(s)

    print(f"Check complete: {'SUCCESS' if success else 'FAILURE'}") if verbose else None
    # return p_problem_states, unacknowledged_states
    return p_problem_states, []


def get_terminal_trace(transition_file, reward_file, terminal_state_id):
    """
    Reads a file of format "id1 id2 prob" and returns the trace (sequence of state ids)
    leading to the given terminal_state_id, if possible.
    """

    transitions = {}
    rewards = {}
    with open(transition_file, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            s, s_prime, p = line.strip().split()
            if s_prime not in transitions:
                transitions[s_prime] = []
            transitions[s_prime].append(s)  # store predecessors for each state
    with open(reward_file, "r") as f:
        # header = f.readline()  # skip header
        for line in f:
            s, s_prime, r = line.strip().split()
            if s_prime not in rewards.keys():
                rewards[s_prime] = int(r)
    # Backtrack from terminal_state_id to 0
    trace = [str(terminal_state_id)]
    # print(rewards.keys())
    reward_trace = [rewards[str(terminal_state_id)]] if str(terminal_state_id) in rewards else [0]
    current = str(terminal_state_id)
    while current != "0":
        if current not in transitions or len(transitions[current]) == 0:
            break
        # Take the first predecessor (could be multiple, but just pick one)
        prev = transitions[current][0]
        trace.append(prev)
        if current in rewards.keys():
            reward_trace.append(rewards[current])
        else:
            reward_trace.append(0)
        current = prev
    trace.reverse()
    reward_trace.reverse()
    return trace, reward_trace
