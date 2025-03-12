from itertools import count
import numpy as np
import math
import os

import torch
import sys
from collections import deque, OrderedDict
from dqn import DeepQNetwork
from queue import Queue


def evaluate_model_by_trial(dqn,
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
    device = torch.device("cpu")
    print(f"Evaluation running on {device}.")

    steps = np.empty(num_episodes)
    deadlock_counter = 0
    deadlock_traces = deque([], maxlen=10)  # store last 10 deadlock traces

    for i in range(num_episodes):
        if (reset_options):
            obs_state, info = env.reset(options=reset_options.copy())
        else:
            obs_state, info = env.reset()

        states = [env.interpret_state_tensor(env.obs_state)]
        actions = []
        obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)
        rel_actions = ["move cc", "move_cw", "engage", "wait"]  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait

        for t in count():

            # calculate action utilities and choose action
            action_utilities = dqn.forward(obs_tensor.unsqueeze(0))[0]  # why is this indexed?
            blocked = env.blocked_model(env, env.obs_state)
            print("blocked", blocked)
            print("blocked actions:",
                  [f"{math.floor(i / env.num_actions)}-{rel_actions[i % env.num_actions]} --- BLOCKED" if b else f"{math.floor(i / env.num_actions)}-{rel_actions[i % env.num_actions]}" for i, b in
                   enumerate(blocked)])
            action_utilities = torch.where(blocked, -1000, action_utilities)
            action = torch.argmax(action_utilities).item()

            # apply action to environment
            new_obs_state, reward, terminated, truncated, info = env.step(action)

            states.append(env.interpret_state_tensor(env.obs_state))
            actions.append(action)
            robot_no = math.floor(action / env.num_actions)
            rel_action = rel_actions[action % env.num_actions]
            print(f"{robot_no}-{rel_action}")

            obs_state = new_obs_state
            obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

            done = terminated

            if (render):
                env.render_frame(states[-1], True)

            if (done or truncated or t > max_steps):
                # if (not torch.cuda.is_available()) : ticks.append(info["elapsed ticks"])
                # if (not torch.cuda.is_available()) : goal_resolutions.append(np.sum(info["goal_resolutions"]))
                if (int(num_episodes / 10) > 0 and i % int(num_episodes / 10) == 0):
                    print(f"{i}/{num_episodes} episodes complete")
                break

        if (not done):
            deadlock_traces.append(states)
            deadlock_counter += 1

        steps[i] = t

    print("Evaluation complete.")
    print(
        f"{'CONVERGENCE SUCCESSFUL' if deadlock_counter == 0 else 'FAILURE'} - Failed to complete {deadlock_counter} times")
    print(f"Percentage converged: {100 - (deadlock_counter * 100 / num_episodes)}")

    return states, actions, steps, deadlock_traces  # states, actions, ticks, steps


def generate_dtmc_file(weights_file, env, system_logic, output_name="dtmc"):
    # load the DQN

    n_actions = env.action_space.n
    state_tensor, info = env.reset()
    initial_state_tensor = state_tensor.detach().clone()
    initial_state_dict = env.interpret_state_tensor(initial_state_tensor)
    n_observations = len(state_tensor)

    if (not weights_file):
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
    states_id_dict = {str(initial_state_dict): 0}  # dictionary of state dicts to id
    labels_set = {"0 init\n"}  # set of state labels ([id] [label] )
    exploration_queue = Queue()

    new_id += 1
    transitions_array = []
    rewards_array = []

    init_state, info = env.reset()  # ask the DQN what action should be taken here
    exploration_queue.put(init_state)

    while (not exploration_queue.empty()):

        print(f"\rStates in exploration queue: {' ' * (10 - len(str(exploration_queue.qsize())))}{exploration_queue.qsize()}", end="")

        state_tensor = exploration_queue.get().detach().clone()  # what is the differnce between state and stateT
        state_dict = env.interpret_state_tensor(state_tensor)

        action_utilities = policy_net.forward(state_tensor.unsqueeze(0))[0]  # why is this indexed?
        blocked = env.blocked_model(env, state_tensor)
        action_utilities = torch.where(blocked, -1000, action_utilities)
        action = torch.argmax(action_utilities).item()

        result = system_logic.t_model(env, state_tensor, action)  # get the result of the action from the transition model

        # label end states
        all_done = system_logic.state_is_final(env, state_tensor)
        if (all_done):
            labels_set.add(f"{states_id_dict[str(state_dict)]} done\n")  # label end states
            transitions_array.append(f"{states_id_dict[str(state_dict)]} {states_id_dict[str(state_dict)]} 1")  # end states loop to themselves (formality):
            continue  # continue as we don't care about other transitions from end states

        for i in range(len(result[0])):  # iterate over result states:

            prob = result[0][i]
            result_state_tensor = result[1][i]
            result_state_dict = env.interpret_state_tensor(result_state_tensor)

            if (str(result_state_dict) not in list(states_id_dict.keys())):  # register newly discovered states
                states_id_dict[str(result_state_dict)] = new_id
                exploration_queue.put(result_state_tensor)
                new_id += 1

            if (np.sum([result_state_dict[f"robot{i} clock"] for i in range(env.num_robots)]) == 0):  # assign awards to clock ticks
                rewards_array.append(f"{states_id_dict[str(state_dict)]} {states_id_dict[str(result_state_dict)]} 1")

            transitions_array.append(f"{states_id_dict[str(state_dict)]} {states_id_dict[str(result_state_dict)]} {prob}")  # write the transitions into the file/array

    print(f"Writing file to {os.getcwd()}/{output_name}.tra, {output_name}.lab, {output_name}.transrew")

    f = open(os.getcwd() + f"/outputs/{output_name}.tra", "w")  # create DTMC file .tra
    f.write("dtmc\n")
    for i in range(len(transitions_array)):
        f.write(transitions_array[i] + "\n")
    f.close()

    f = open(os.getcwd() + f"/outputs/{output_name}.lab", "w")  # create labels file .lab
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

    f = open(os.getcwd() + f"/outputs/{output_name}.transrew", "w")  # rewards file .transrew
    for i in range(len(rewards_array)):
        f.write(rewards_array[i] + "\n")
    f.close()

    print(f"Saved policy DTMC as {output_name}.")

    # check DTMC for invalid states
    p_problem_states, unacknowledged_states = check_dtmc(os.getcwd() + f"/outputs/{output_name}.tra")

    if (len(p_problem_states) == 0):
        print("Success: all probabilities sum to 1")
    else:
        print("Error! Some outgoing probabilities do not sum to 1\nstate | total p")
        for i in range(len(p_problem_states)):
            print(f"{p_problem_states[i][0]} | {p_problem_states[i][1]}")

    if (len(unacknowledged_states) == 0):
        print("Success: all states included in transition structure")
    else:
        print("Error! Some encountered states have no outgoing transitions!\nStates:")
        for i in range(len(unacknowledged_states)):
            print(unacknowledged_states[i])


def check_dtmc(filepath, verbose=False):
    p_outs = {}
    accessible_states = set([])

    with open(filepath) as f:
        header = f.readline()
        assert header.strip() == "dtmc"
        for line in f:
            s,s_prime,p = line.strip().split(" ")
            accessible_states.add(s)
            accessible_states.add(s_prime)
            if(str(s) in p_outs):
                p_outs[str(s)] += float(p)
            else:
                p_outs[str(s)] = float(p)

    out_states = list(p_outs.keys())
    out_probabilities = list(p_outs.values())
    p_problem_states = []
    for i in range(len(out_states)):
        if(out_probabilities[i] != 1.0):
            if(verbose):
                print(f"Error! s={out_states[i]} -> total p={out_probabilities[i]}")
            p_problem_states.append([out_states[i], out_probabilities[i]])

    unacknowledged_states = []
    for s in accessible_states:
        if s not in out_states:
            if(verbose):
                print(f"Error! s={s} has no outgoing transitions!")
            unacknowledged_states.append(s)

    return p_problem_states, unacknowledged_states
