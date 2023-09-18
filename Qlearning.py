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

def GreedyPolicy(action_utilities):
    action = np.argmax(action_utilities)
    return action


def TrainGeneral(qtable,
                  system,
                  n_training_episodes,
                  max_steps,
                  min_epsilon,
                  max_epsilon,
                  decay_rate,
                  gamma,
                  learning_rate,
                  verbosity=1000):

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
            action = EpsilonGreedyPolicy(action_utilities, epsilon, system.transitions[state_index] < 0)
            
            new_state_index = int(system.transitions[state_index, action])
            
            pseudoreward = gamma * system.calculate_psuedorewards(new_state_index, action) - system.calculate_psuedorewards(state_index, action)    
            reward = (system.transition_rewards[state_index, action])*10 + pseudoreward
            
            episode_reward += reward
     
            qtable[state_index, action] = qtable[state_index, action] + \
                learning_rate * (reward + gamma * np.max(qtable[new_state_index]) - qtable[state_index, action])
            
            # move the system on to the new state index
            if(state_index==new_state_index): 
                print("warning: same state")
                
            state_index = new_state_index
            system.assume_state(state_index)   
        

            if(state_index < 0 or state_index >= len(system.transitions)):
                neg_one_counter += 1
                print(f"Warning: at state {state_index}")
                print(action, system.system_transitions[state_index] < 0)
                print((system.system_transitions[state_index] < 0), [action])
                
            # if(step==max_steps-1):
            #     print(f"Completed:{system.interpret_state(system.system_states[system.current_state_index])[-1]}, current index: {system.current_state_index}")
            
        episode_times.append(step)
        period_rewards.append(episode_reward)

        if (episode%verbosity == 0 and episode > 0):
            period_done_percent = (done_counter_period*100/verbosity)
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


class GeneralisedSystem():
    
    def __init__(self, system_states,
                 initial_indices,
                 absorbing_indices,
                 transitions,
                 transition_rewards,
                 pseudorewards_function=(lambda x: 0),
                 interpret_state_func=(lambda x: None)):
        
        self.system_states = system_states
        self.transitions = transitions
        self.transition_rewards = transition_rewards
        self.initial_indices = initial_indices
        self.absorbing_indices = absorbing_indices
        self.pseudorewards_function = pseudorewards_function
        self.interpret_state_func = interpret_state_func
        self.reset()
        
    def interpret_state(self, state):
        return self.interpret_state_func(self, state)
    
    def reset(self):
        choice = random.choice(self.initial_indices)
        self.assume_state(choice)
        return self.current_state_index
    
    def calculate_psuedorewards(self, state_index, action):
        return self.pseudorewards_function(self, self.system_states[state_index], action)
    
    def assume_state(self, state_index):
        self.current_state_index = state_index
        
    def check_done(self):
        if (self.current_state_index in self.absorbing_indices):
                return True
        return False
    

def EvaluateGeneral(qtable, system, n_eval_episodes, max_steps, gamma):

    print("Evaluation commensing...")

    episode_rewards = []
    episode_times = []
    done_counter = 0

    for episode in range(n_eval_episodes):

        state_index = system.reset()
        total_rewards_ep = 0

        # Take the action (index) that have the maximum reward
        for step in range(max_steps):
            
            if system.check_done():
                done_counter += 1
                break
            
            action = GreedyPolicy(qtable[state_index])
        
            new_state_index = int(system.transitions[state_index, action])
            system.assume_state(new_state_index)   
            
            pseudoreward = gamma * system.calculate_psuedorewards(new_state_index, action) - system.calculate_psuedorewards(state_index, action)    
            reward = (system.transition_rewards[state_index, action])*10 + pseudoreward

            total_rewards_ep += reward
            state_index = new_state_index

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

    print(f"\nEvaluation complete. Average time: {mean_time} steps. Completed {done_counter*100/n_eval_episodes}% of missions.")

    return mean_reward, std_reward, mean_time, std_time, fraction_done