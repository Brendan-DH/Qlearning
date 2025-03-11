import random
import numpy as np
import torch
import torch.nn as nn

from dqn.dqn_collections import DeltaTransition


def optimise_model_with_importance_sampling(policy_dqn,
                                            target_dqn,
                                            replay_memory,
                                            optimiser,
                                            optimiser_device,
                                            gamma_tensor,
                                            batch_size,
                                            priority_coefficient,
                                            weighting_coefficient):
    # optimiser_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(replay_memory.memory) < batch_size or len(replay_memory.bounds) == 0:
        # print(f"memory not yet ready {len(replay_memory.memory)}/{batch_size} | {len(replay_memory.bounds)}")
        return

    # print("Attempting to optimise...")

    # policy_dqn = nn.parallel.DistributedDataParallel(policy_dqn).to(optimiser_device)
    # target_dqn = nn.parallel.DistributedDataParallel(target_dqn).to(optimiser_device)

    policy_dqn = policy_dqn.to(optimiser_device)
    target_dqn = target_dqn.to(optimiser_device)

    # get the batch of transitions. sample one transition from each of k linear segments
    lower = 0
    transitions = np.empty(batch_size, dtype=DeltaTransition)
    transition_indices = torch.tensor(np.empty(batch_size, dtype=int), device=torch.device("cpu"), requires_grad=False)
    weights = torch.empty(batch_size, dtype=torch.float, device=torch.device("cpu"), requires_grad=False)

    # loop over the k linear segments
    for i in range(batch_size):
        upper = replay_memory.bounds[i]
        tr_index = random.randint(lower, upper - 1)  # get a random index that falls in the segment
        transition_indices[i] = tr_index  # must be stored to update the tr delta later
        transitions[i] = replay_memory.memory[tr_index]

        tr_priority = 1 / (tr_index + 1)
        p_tr = (tr_priority ** priority_coefficient) * replay_memory.prob_divisor
        weight = ((batch_size * p_tr) ** -weighting_coefficient) * (1 / replay_memory.max_priority)
        weights[i] = weight

        lower = upper

    # print(transitions)
    batch = DeltaTransition(*zip(*transitions))
    weights = weights.to(optimiser_device)

    # boolean mask of which states are NOT final (final = termination occurs in this state)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=torch.device(optimiser_device), dtype=torch.bool)

    # collection of non-final states
    # with torch.no_grad():
    #     non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0).to(optimiser_device)  # tensor
    #     state_batch = torch.cat([state.unsqueeze(0) for state in batch.state], dim=0).to(optimiser_device)  # tensor

    # with torch.no_grad():
    non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0).to(optimiser_device)  # tensor
    state_batch = torch.cat([state.unsqueeze(0) for state in batch.state], dim=0).to(optimiser_device)  # tensor

    action_batch = torch.tensor(batch.action, requires_grad=False).to(optimiser_device)  # tensor
    reward_batch = torch.tensor(batch.reward, requires_grad=False).to(optimiser_device)  # tensor

    # the qvalues of actions in this state as according to the policy network
    try:
        q_values = policy_dqn(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1))
    except AttributeError:
        raise Exception("Something went wrong with gathering the state/action q-values")

    # q values of action in the next state
    next_state_values = torch.zeros(batch_size, device=optimiser_device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * gamma_tensor) + reward_batch
    # print("expected_state_action_values.device", expected_state_action_values.device)

    # Compute loss. Times by weight of transition
    criterion = nn.SmoothL1Loss(reduction="none")  # (Huber loss)
    loss_vector = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) * weights
    loss = torch.mean(loss_vector)
    #     print("loss_vector.device, loss.device ", loss_vector.device, loss.device)

    # optimise the model
    # optimiser_device_check(optimiser)
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), 100)  # stops the gradients from becoming too large
    optimiser.step()

    # now update the priorities of the transitions that were used
    for i in range(len(transition_indices)):  # would it be more efficient to store just delta_i? might require fewer calculations
        index = transition_indices[i]
        # the new priority (delta) is the mean loss for this transition (how surprising it was)
        replay_memory.update_priorities(index, torch.mean(loss_vector[i]).item())

    target_dqn.to(torch.device("cpu"))
    policy_dqn.to(torch.device("cpu"))
