import os
import re

import numpy as np
import pandas as pd
from dqn.dqn_collections import system_parameters
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'


def sigmoid(x, k=1, L=1):
    return 1 / (1 + np.exp(-k * x))


def generate_system_parameters(ihfs_dir=os.getcwd() + "/ihfs", robot_locations=None):
    parameter_dictionary = {}
    directory = os.listdir(ihfs_dir)
    if robot_locations is None:
        robot_locations = [0, 7, 13]

    all_pc = []
    all_pd = []

    for i in range(len(directory)):
        filename = directory[i]
        rects_name = re.findall(r"-(.*?)\.", filename)[0]
        print(f"Name: {rects_name}")
        ihfs = pd.read_pickle(ihfs_dir + "/" + filename)
        size = len(ihfs)
        print(f"Num segments: {size}")
        print(f"Max/min value segment: {np.argmax(ihfs['norm. hf'])}/{np.argmin(ihfs['norm. hf'])}")
        completion_prob =   list(-0.9 * ihfs['norm. hf'].values + 1)
        discovery_prob = list(sigmoid(ihfs['norm. hf'].values, 5) * 0.9)
        
        

        plt.figure()
        plt.plot(discovery_prob, label="Discovery probability")
        plt.plot(completion_prob, label="Completion probability")
        
        all_pc.append(completion_prob)
        all_pd.append(discovery_prob)
        
        plt.plot(ihfs["norm. hf"], label="Normalised heat")
        plt.xlabel("Segment")
        plt.xticks(range(0, size))
        plt.grid()
        plt.legend()
        plt.title(rects_name)
        plt.savefig(f"outputs/plots/{rects_name}_scenario.svg")
        plt.savefig(f"outputs/plots/{rects_name}_scenario.png")
        plt.close()

        sys_param = system_parameters(
            size=size,
            robot_locations=robot_locations,
            goal_locations=[i for i in range(size)],
            goal_completion_probabilities=completion_prob,
            goal_discovery_probabilities=discovery_prob,
            goal_activations=[0 for i in range(size)],
            goal_checked=[0 for i in range(size)],
            elapsed_ticks=0,
        )
        print(f"{rects_name}=", sys_param)
        parameter_dictionary[rects_name] = sys_param

        print("\n", end="")
        
    plt.figure(figsize=(5,6))
    for pc in all_pc:
        plt.plot(pc)
    plt.xlabel("Segment")
    plt.ylabel("$p_c$")
    plt.xticks(range(0,20))
    plt.title("$p_c$ versus Segment Number")
    plt.grid(alpha=0.5)
    plt.savefig("outputs/plots/all_pc.svg")
    
    plt.figure(figsize=(5,6))
    for d in all_pd:
        plt.plot(d)
    plt.xlabel("Segment")
    plt.ylabel("$p_d$")
    plt.xticks(range(0,20))
    plt.title("$p_d$ versus Segment Number")
    plt.grid(alpha=0.5)
    plt.savefig("outputs/plots/all_pd.svg")

    return parameter_dictionary


param_dict = generate_system_parameters()

print(param_dict)
