#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:04:03 2024

A script to verify that a DTMC file is valid, i.e.:
- all outgoing probabilities sum to 1
- all states (initial and resultant) are accounted for in transitions

@author: brendandevlin-hill
"""

import os
import argparse
from dqn.evaluation import check_dtmc

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()
filepath = os.getcwd() + "/" + args.filepath

print(f"Checking {filepath}")

p_problem_states, unacknowledged_states = check_dtmc(filepath)

if (len(p_problem_states) == 0):
    print("\nSuccess: all probabilities sum to 1")
else:
    print("\nError! Some outgoing probabilities do not sum to 1\nstate | total p")
    for i in range(len(p_problem_states)):
        print(f"{p_problem_states[i][0]} | {p_problem_states[i][1]}")

if(len(unacknowledged_states) == 0):
    print("\nSuccess: all states included in transition structure")
else:
    print("\nError! Some encountered states have no outgoing transitions!\nStates:")
    for i in range(len(unacknowledged_states)):
        print(unacknowledged_states[i])
