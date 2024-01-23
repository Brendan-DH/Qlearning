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


def CheckDTMC(filepath, verbose=False):
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


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()
    filepath = os.getcwd() + "/" + args.filepath

    print(f"Checking {filepath}")

    p_problem_states, unacknowledged_states = CheckDTMC(filepath)

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
