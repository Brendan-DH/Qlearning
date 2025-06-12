import random
import numpy as np
import torch
import torch.nn as nn
from numpy.ma.core import shape

from dqn.dqn_collections import DeltaTransition, Transition


def optimise(policy_dqn,
            target_dqn,
            replay_memory,
            optimiser,
            optimiser_device,
            gamma_tensor,
            batch_size,
            priority_coefficient,
            weighting_coefficient):
    
    if len(replay_memory.memory) < batch_size:
        return

    mem_size = len(replay_memory.memory)
    
    if(replay_memory.memory_type == "priority"):
        
        if len(replay_memory.bounds) == 0:
            return None
            
        
        # print(replay_memory.bounds)

        # get the batch of transitions. sample one transition from each of k linear segments
        lower = 0
        transitions = np.empty(batch_size, dtype=DeltaTransition)
        transition_indices = np.empty(batch_size, dtype=int)  # torch.tensor(np.empty(batch_size, dtype=int), device=torch.device("cpu"), requires_grad=False)
        weights = np.empty(batch_size, dtype=float)  # torch.empty(batch_size, dtype=torch.float, device=torch.device("cpu"), requires_grad=False)

        # loop over the k linear segments
        max_w = (mem_size * replay_memory.prob_divisor) ** -weighting_coefficient
        for i in range(batch_size):
            upper = replay_memory.bounds[i]
            tr_index = random.randint(lower, upper - 1)  # get a random index that falls in the segment
            transition_indices[i] = tr_index  # must be stored to update the tr delta later
            transitions[i] = replay_memory.memory[tr_index]
            
            tr_priority = 1 / (tr_index + 1)  # priority of the transition (rank-based sampling)
            p_tr = (tr_priority ** priority_coefficient) * replay_memory.prob_divisor  # the probability of this tr being picked
            weight = (mem_size * p_tr) ** -weighting_coefficient
            weights[i] = weight
            lower = upper

        batch = DeltaTransition(*zip(*transitions))

        # for calculating reward normalisation:

        r_tr = DeltaTransition(*zip(*replay_memory.sample(int(len(replay_memory) / 4)))).reward
        r_mean = np.mean(r_tr)
        r_std = np.std(r_tr)

        weights = weights * (1 / max_w)
        weights = torch.as_tensor(weights, dtype=torch.float, device=optimiser_device)
    
    elif(replay_memory.memory_type == "replay"):
        batch = Transition(*zip(*replay_memory.sample(batch_size)))
        weights = torch.ones(batch_size, dtype=torch.float, device=optimiser_device)  # no importance sampling weights
        r_tr = Transition(*zip(*replay_memory.sample(int(len(replay_memory) / 4)))).reward
        r_mean = np.mean(r_tr)
        r_std = np.std(r_tr)
        

    # boolean mask of which states are NOT final (final = termination occurs in this state)
    non_final_mask = torch.as_tensor(tuple(map(lambda ns: ns is not None, batch.next_state)), device=torch.device(optimiser_device), dtype=torch.bool)

    # collection of non-final states
    non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0).to(optimiser_device)  # tensor
    non_final_blocks = torch.stack([b for b, ns in zip(batch.blocked, batch.next_state) if ns is not None]).to(optimiser_device)
    state_batch = torch.cat([state.unsqueeze(0) for state in batch.state], dim=0).to(optimiser_device)  # tensor

    action_batch = torch.as_tensor(batch.action).to(optimiser_device)  # tensor
    blocked_batch = torch.stack(list(batch.blocked)).to(optimiser_device) 
    # print(blocked_batch.shape, blocked_batch, type(blocked_batch))
    
    reward_batch = torch.as_tensor((batch.reward - r_mean) / (r_std + 1e-6)).to(optimiser_device)  # tensor

    try:
        q_values = policy_dqn(state_batch)
        q_values = q_values.masked_fill(blocked_batch, -np.inf)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1))
    except (AttributeError, RuntimeError) as e:
        print("policy_dqn.device:", next(policy_dqn.parameters()).device)
        print("state_batch.device:", state_batch.device)
        print("optimiser_device:", optimiser_device)
        raise Exception("Something went wrong with gathering the state/action q-values")

    # q values of action in the next state
    next_state_values = torch.zeros(batch_size, device=optimiser_device)
    with torch.no_grad():
        next_q_values = target_dqn(non_final_next_states)
        next_q_values = next_q_values.masked_fill(non_final_blocks, -np.inf)
        # print(next_q_values)
        next_state_values[non_final_mask] = next_q_values.max(1)[0]

    expected_state_action_values = (next_state_values * gamma_tensor) + reward_batch
    
    # print(expected_state_action_values.shape, state_action_values.shape)

    # Compute loss. Times by weight of transition
    criterion = nn.SmoothL1Loss(reduction="none")  # (Huber loss)
    loss_vector = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) * weights.unsqueeze(1)
    loss = torch.mean(loss_vector).to(optimiser_device)

    # optimise the model
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), 1)  # stops the gradients from becoming too large
    optimiser.step()

    # now update the priorities of the transitions that were used
    if replay_memory.memory_type == "priority":
        for i in range(len(transition_indices)):  # would it be more efficient to store just delta_i? might require fewer calculations
            index = transition_indices[i]
            # the new priority (delta) is the mean loss for this transition (how surprising it was)
            replay_memory.update_priorities(index, torch.mean(loss_vector[i]).item())

    return loss.item()
