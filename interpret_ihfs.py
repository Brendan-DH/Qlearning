import os
import re

import numpy as np
import pandas as pd
from dqn.dqn_collections import system_parameters


def generate_system_parameters(ihfs_dir=os.getcwd() + "/ihfs", robot_locations=None):
    parameter_dictionary = {}
    directory = os.listdir(ihfs_dir)
    if robot_locations is None:
        robot_locations = [0, 7, 13]

    for i in range(len(directory)):
        filename = directory[i]
        rects_name = re.findall(r"-(.*?)\.", filename)[0]
        print(f"Name: {rects_name}")
        ihfs = pd.read_pickle(ihfs_dir + "/" + filename)
        size = len(ihfs)
        print(f"Num segments: {size}")
        print(f"Max/min value segment: {np.argmax(ihfs['norm. hf'])}/{np.argmin(ihfs['norm. hf'])}")
        completion_prob = list((1 - ihfs['norm. hf'] * 0.9).values)
        discovery_prob = list((ihfs['norm. hf'] * 0.80).values)

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
        print("system parameters:\n", sys_param)
        parameter_dictionary[rects_name] = sys_param

        print("\n", end="")

    return parameter_dictionary


generate_system_parameters()