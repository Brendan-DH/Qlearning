
# A DQN for Tokamak Maintenance

This repo contains work from my PhD thesis ``A Full-Plant, Physics-Aware Approach to the Planning and Control of Multi-Robot Nuclear Fusion Maintenance Systems''. Specifically, this work relates to a tokamak maintenance pipeline, in which tokamak plasma crashes are simulated, the data are used to inform the operation of an autonomous, multi-robot tokamak maintenance system using a neural network, and the resulting control policy is extracted into a Markov process such that it can be investigated using formal verification techniques.

This repo is intended for use on high-performance GPU infrastructure, as the learning is accelerated with CUDA. However, one can also execute the learning process on a CPU.

## Contents

This repo contains:

- A set of maintenance scenarios based on tokamak crashes, simulated by the JOREK-STARWALL simulation code [1,2].

- Gymnasium environments which define the structure of the tokamak maintenance problem.

- Modularised system logic which defines the logic of the robotic maintenance system and its environment.

- A Deep Q Network (DQN) trainer which learns the ideal control policy for the scenario.

- A DQN -> Markov process translator, which allows the user to verify properties of the learnt policy.

## Requirements:

PyTorch and Gymnasium are required for learning. PyGame is required if you wish to render the system. The Storm model checker [3,4] is required to verify the properties of the control policy.

## Setting up the tokamak gymnasium environment

The repo contains a script `intitialise` which performs most of the set up.  Calling this will:

- Load the correct python3.11 modules (on a module-compatible HPC)
- Create a python3.11 virtual environment,
- Install requirements via requirements.txt
- Register the relevant environment wtih gymnasium
- Generate the directory structure for the outputs

## Running

First, run `tokamak_trainer':

> python -m tokamak_trainer.py

This will generate a default_inputs.in file in the inputs directory. Edit this as you like, specifying the name of a scenario from scenarios.py. Then run tokamak_trainer again using your input file:

> python -m tokamak_trainer.py < inputs/INPUT.in

To evaluate the resulting policy, copy the saved weights from the output directory to the inputs directory and do:

> python -m tokamak_evaluator.py < inputs/INPUT.in

This will evaluate the policy by trial, construct a DTMC representing the policy, verify that the structure and format of the DTMC is valid, and then verify a few simple system properties using Storm.

References:

[1] M Hoelzl, GTA Huijsmans, SJP Pamela, M Becoulet, E Nardon, FJ Artola, B Nkonga, et al The JOREK non-linear extended MHD code and applications to large-scale instabilities and their control in magnetically confined fusion plasmas NF 61, 065001 (2021) 

[2] [www.jorek.eu](https://www.jorek.eu/)

[3] Christian Hensel, Sebastian Junges, Joost-Pieter Katoen, Tim Quatmann, and Matthias Volk, “The probabilistic model checker Storm,” Int. J. Softw. Tools Technol. Transf., no. 4, 2022.

[4] [www.stormchecker.org/](https://www.stormchecker.org/)

