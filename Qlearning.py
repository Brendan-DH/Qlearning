#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:17:41 2023

@author: brendandevlin-hill
"""

import numpy as np
import random


def EpsilonGreedyPolicy(action_utilities, epsilon, forbidden):
    random_float = random.uniform(0, 1)
    if random_float > epsilon:
        # choose the action which has the max utility and which is
        # not disallowed by 'forbidden' - hope there is a cleaner way to accomplish
        allowed_actions = [utility if forbidden[index] == False else -
                            np.Inf for index, utility in enumerate(action_utilities)]
        action = np.argmax(allowed_actions)
        # action = np.argmax(action_utilities)
        return action
    else:
        action = random.randint(0, len(action_utilities)-1)
        while(forbidden[action]):  # check if forbidden
            action = random.randint(0, len(action_utilities)-1)
        return action


# def EpsilonGreedyPolicyNaive(action_utilities, epsilon):
#     random_float = random.uniform(0, 1)
#     if random_float > epsilon:
#         action = np.argmax(action_utilities)

#     else:
#         action = random.randint(0, len(action_utilities)-1)
#     return action


def GreedyPolicy(action_utilities):
    action = np.argmax(action_utilities)
    return action


def Train(qtable,
          system,
          initial_state_indices,
          transitions,
          n_training_episodes,
          max_steps,
          min_epsilon,
          max_epsilon,
          decay_rate,
          gamma,
          learning_rate):

    print("Training commensing...")
    neg_one_counter = 0
    done_counter_period = 0
    episode_times = []
    period_rewards = []


    for episode in range(n_training_episodes):

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        # print("-----------------------new ep------------------------------------")
        episode_reward = 0

        # Reset the environment
        state_index = random.choice(initial_state_indices)
        system.assume_state(system.system_states[state_index])
        
        state_trace = []

        # repeat
        for step in range(0, max_steps):

            state_trace.append(state_index)

            # select the new state_index
            action_utilities = qtable[state_index]
            action = EpsilonGreedyPolicy(action_utilities, epsilon, transitions[state_index] < 0)
            new_state_index = int(transitions[state_index, action])

            # Define the reward structure:
            # reward = (system.transition_rewards[state_index, action]) + \
            #     gamma * system.calculate_psuedorewards (system.system_states[new_state_index], action) - \
            #     system.calculate_psuedorewards(system.system_states[state_index], action)
            
            reward = (system.transition_rewards[state_index, action]) - 0.5
            episode_reward += reward

            # print(state_index, action, new_state_index)
            qtable[state_index, action] = qtable[state_index, action] + \
                learning_rate * (reward + gamma * np.max(qtable[new_state_index]) - qtable[state_index, action])
            
            # move the system on to the new state index
            state_index = new_state_index
            system.assume_state(system.system_states[state_index])   
            
            # If done, finish the episode
            if system.check_done():
                done_counter_period += 1
                # for index in state_trace:
                #     qtable[state_trace] += 10
                break


            if(state_index < 0 or state_index >= len(transitions)):
                neg_one_counter += 1
                print(f"Warning: at state {state_index}")
                print(action, transitions[state_index] < 0)
                print((transitions[state_index] < 0)[action])

        episode_times.append(step)
        period_rewards.append(episode_reward)

        if (episode % 5000 == 0 and episode > 0):
            period_done_percent = (done_counter_period*100/5000)
            print("\nEpisode {: >5d}/{:>5d} | epsilon: {:0<7.5f} | Av. steps: {: >4.2f} | Min steps: {: >4d} | Av. reward: {: >4.2f} | Completed: {: >4.2f}%".format(episode,
                                                                                                                                        n_training_episodes,
                                                                                                                                        epsilon,
                                                                                                                                        np.mean(episode_times),
                                                                                                                                        np.min(episode_times),
                                                                                                                                        np.mean(period_rewards),
                                                                                                                                        period_done_percent), end="")
            period_rewards = []
            episode_reward = 0                                                                                                          
            neg_one_counter = 0
            done_counter_period = 0
            episode_times = []

    print("\nTraining finished.")
    return qtable

def TrainGrid(qtable,
              system,
              n_training_episodes,
              max_steps,
              min_epsilon,
              max_epsilon,
              decay_rate,
              gamma,
              learning_rate):

    print("Training commensing...")
    neg_one_counter = 0
    done_counter_period = 0
    episode_times = []
    period_rewards = []


    for episode in range(n_training_episodes):

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        # print("-----------------------new ep------------------------------------")
        episode_reward = 0

        # Reset the environment
        state_index = system.reset()        

        # repeat
        for step in range(0, max_steps):
            
            # If done, finish the episode
            if system.check_done():
                done_counter_period += 1
                break

            # select the new state_index
            action_utilities = qtable[state_index]
            action = EpsilonGreedyPolicy(action_utilities, epsilon, system.system_transitions[state_index] < 0)
            
            new_state_index = int(system.system_transitions[state_index, action])

            #reward = (system.transition_rewards[state_index, action])*10 - 0.5
            reward = (system.transition_rewards[state_index, action])*10 + \
                    gamma * system.calculate_psuedorewards(system.system_states[new_state_index], action) - \
                    system.calculate_psuedorewards(system.system_states[state_index], action)
                    
            episode_reward += reward
     
            qtable[state_index, action] = qtable[state_index, action] + \
                learning_rate * (reward + gamma * np.max(qtable[new_state_index]) - qtable[state_index, action])
            
            # move the system on to the new state index
            state_index = new_state_index
            system.assume_state(system.system_states[state_index])   
        

            if(state_index < 0 or state_index >= len(system.system_transitions)):
                neg_one_counter += 1
                print(f"Warning: at state {state_index}")
                print(action, system.system_transitions[state_index] < 0)
                print((system.system_transitions[state_index] < 0), [action])

        episode_times.append(step)
        period_rewards.append(episode_reward)

        if (episode % 5000 == 0 and episode > 0):
            period_done_percent = (done_counter_period*100/5000)
            print("\nEpisode {: >5d}/{:>5d} | epsilon: {:0<7.5f} | Av. steps: {: >4.2f} | Min steps: {: >4d} | Av. reward: {: >4.2f} | Completed: {: >4.2f}%".format(episode,
                                                                                                                                        n_training_episodes,
                                                                                                                                        epsilon,
                                                                                                                                        np.mean(episode_times),
                                                                                                                                        np.min(episode_times),
                                                                                                                                        np.mean(period_rewards),
                                                                                                                                        period_done_percent), end="")
            period_rewards = []
            episode_reward = 0                                                                                                          
            neg_one_counter = 0
            done_counter_period = 0
            episode_times = []

    print("\nTraining finished.")
    return qtable


class GridSystem():
    
    def __init__(self, initial_states, goal_position, system_states, states_dict, system_transitions, transition_rewards, pseudorewards_function):
        self.system_states = system_states
        self.system_transitions = system_transitions
        self.states_dict = states_dict
        self.transition_rewards = transition_rewards
        self.initial_states = initial_states
        self.goal_position = goal_position
        self.pseudorewards_function = pseudorewards_function
        
        self.reset()
        
    def reset(self):
        self.robot_position = random.choice(self.initial_states)
        return self.states_dict[str(self.robot_position)]
    
    def calculate_psuedorewards(self, state, action):
        return self.pseudorewards_function(self, state, action)
    
    def assume_state(self, state):
        self.robot_position = state
        
    def check_done(self):
        if (self.robot_position[0] == self.goal_position[0] and self.robot_position[1] == self.goal_position[1]):
            return True
        else:
            return False
    
        
class PrimitiveSystem():

    def __init__(self, initial_state_index, system_states, system_transitions, transition_rewards, pseudorewards_function, num_robots=3, num_damaged=3):
        self.system_states = system_states
        self.system_transitions = system_transitions
        self.transition_rewards = transition_rewards
        self.pseudorewards_function = pseudorewards_function
        self.num_robots = num_robots
        self.num_damaged = num_damaged
        
        self.interpret_state(initial_state_index)

    def reset(self):
        self.damaged_segments = self.initial_state.copy() # must be a copy
        
    def interpret_state(self, state):
        robot_positions = state[:self.num_robots]
        robots_inspecting = state[self.num_robots:(self.num_robots*2)]
        damaged_segments = state[(self.num_robots*2):]
        print(damaged_segments)
        return robot_positions, robots_inspecting, damaged_segments
        
    def assume_state(self, state):
        self.robot_positions, self.robots_inspecting, self.damaged_segments = self.interpret_state(state)
        # print("going to ", self.robot_positions, self.robots_inspecting, self.damaged_segments )
        
    def calculate_psuedorewards(self, state, action):
        return self.pseudorewards_function(self, state, action)
        
    def check_done(self):
        # print(self.damaged_segments)
        # print([element == -1 for element in self.damaged_segments])
        return all(element == -1 for element in self.damaged_segments)


def Evaluate(qtable, system, initial_state_indices, transitions, n_eval_episodes, max_steps):

    print("Evaluation commensing...")

    episode_rewards = []
    episode_times = []
    done_counter = 0

    for episode in range(n_eval_episodes):

        state_index = random.choice(initial_state_indices)
        system.assume_state(system.system_states[state_index])
        total_rewards_ep = 0

        # Take the action (index) that have the maximum reward
        for step in range(max_steps):
            action = EpsilonGreedyPolicy(qtable[state_index], 0.05, transitions[state_index] < 0)
            new_state_index = int(transitions[state_index, action])
            reward = (system.transition_rewards[state_index, action])

            total_rewards_ep += reward

            if system.check_done():
                done_counter += 1
                break

            state_index = new_state_index

        episode_rewards.append(total_rewards_ep)
        episode_times.append(step)

        if (episode % 1000 == 0 and episode>0):
            print("\nEpisode {: >5d}/{:>5d} | Av. time: {: >4.2f} | Completed: {: >4.2f}%".format(episode,
                                                                                                  n_eval_episodes,
                                                                                                  np.mean(episode_times),
                                                                                                  done_counter*100/(episode+1)), end="")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_time = np.mean(episode_times)
    std_time = np.std(episode_times)
    fraction_done = done_counter/n_eval_episodes

    print(
        f"\nEvaluation complete. Average time: {mean_time} steps. Completed {done_counter*100/n_eval_episodes}% of missions.")

    return mean_reward, std_reward, mean_time, std_time, fraction_done

def EvaluateGrid(qtable, system, n_eval_episodes, max_steps, gamma):

    print("Evaluation commensing...")

    episode_rewards = []
    episode_times = []
    done_counter = 0

    for episode in range(n_eval_episodes):

        state_index = system.reset()
        state_trace = [state_index]
        total_rewards_ep = 0

        # Take the action (index) that have the maximum reward
        for step in range(max_steps):
            
            if system.check_done():
                done_counter += 1
                break
            
            action = GreedyPolicy(qtable[state_index])
        
        
            new_state_index = int(system.system_transitions[state_index, action])
            system.assume_state(system.system_states[new_state_index])   
            state_trace.append(new_state_index)
            
            reward = (system.transition_rewards[state_index, action])*10 + \
                    gamma * system.calculate_psuedorewards(system.system_states[new_state_index], action) - \
                    system.calculate_psuedorewards(system.system_states[state_index], action) - 0.5

            total_rewards_ep += reward


            state_index = new_state_index

        #print(state_trace)
        episode_rewards.append(total_rewards_ep)
        episode_times.append(step+1)

        if (episode % 1000 == 0 and episode>0):
            print("\nEpisode {: >5d}/{:>5d} | Av. time: {: >4.2f} | Completed: {: >4.2f}%".format(episode,
                                                                                                  n_eval_episodes,
                                                                                                  np.mean(episode_times),
                                                                                                  done_counter*100/(episode+1)), end="")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_time = np.mean(episode_times)
    std_time = np.std(episode_times)
    fraction_done = done_counter/n_eval_episodes

    print(
        f"\nEvaluation complete. Average time: {mean_time} steps. Completed {done_counter*100/n_eval_episodes}% of missions.")

    return mean_reward, std_reward, mean_time, std_time, fraction_done

def TrainDynamicQtable(system, reward_scheme, initial_state_indices, transitions, n_training_episodes, max_steps, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate):

    print("Training commensing...")

    done_counter = 0
    done_counter_period = 0
    period_times = []

    # construct initial qtable

    qtable = np.random.rand(len(initial_state_indices),
                            len(transitions[0])) * 10
    # this stores the state indices corresponding to each entry in the qtable
    q_lookup = initial_state_indices.copy()

    for episode in range(n_training_episodes):

        epsilon = min_epsilon + \
            (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

        # Reset the environment
        state_index = random.choice(initial_state_indices)
        system.reset()

        # repeat
        for step in range(max_steps):

            # select the new state_index
            # print(q_lookup, state_index)
            # print(np.argwhere(q_lookup == state_index))
            action_utilities = qtable[q_lookup.index(state_index)]
            action = EpsilonGreedyPolicy(
                action_utilities, epsilon, transitions[state_index] < 0)
            new_state_index = int(transitions[state_index, action])

            if(new_state_index not in q_lookup):
                # add a new entry to the qtable
                if(new_state_index == -1):
                    print(f"Warning: Adding state {new_state_index} to qtable")
                qtable = np.vstack(
                    (qtable, np.random.rand(len(transitions[0]))))
                q_lookup.append(new_state_index)

            # get reward
            reward = reward_scheme(system, state_index, step)

            # find where the state_index is in the qtable
            q_old_index = q_lookup.index(state_index)
            q_new_index = q_lookup.index(new_state_index)

            # print(q_old_index, q_new_index)

            qtable[q_old_index, action] = qtable[q_old_index, action] + learning_rate * \
                (reward + gamma *
                 np.max(qtable[q_new_index]) - qtable[q_old_index, action])

            # If done, finish the episode
            if system.check_done():
                done_counter += 1
                done_counter_period += 1
                break

            # Our state_index is the new state_index
            state_index = new_state_index

            period_times.append(step)

        if (episode % 1000 == 0):
            period_done_percent = (done_counter_period/1000)*100
            print("\nEpisode {: >5d}/{:>5d} | epsilon: {:0<7.5f} | Av. time: {: >4.2f} | Completed: {: >4.2f}%".format(episode,
                                                                                                                       n_training_episodes,
                                                                                                                       epsilon,
                                                                                                                       np.mean(
                                                                                                                           period_times),
                                                                                                                       period_done_percent), end="")
            done_counter_period = 0
            period_times = []

    print(
        f"\nTraining finished. Completed missions: {done_counter}/{n_training_episodes} ({done_counter * 100 /n_training_episodes}%)")
    return qtable