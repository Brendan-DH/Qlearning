
# A DQN for Tokamak Maintenance

This repo contains work from my PhD thesis ``A Full-Plant, Physics-Aware Approach to the Planning and Control of Multi-Robot Nuclear Fusion Maintenance Systems''. Specifically, this work relates to a tokamak maintenance pipeline, in which tokamak plasma crashes are simulated, the data are used to inform the operation of an autonomous, multi-robot tokamak maintenance system using a neural network, and the resulting control policy is extracted into a Markov process such that it can be investigated using formal verification techniques.

This repo is intended for use on high-performance GPU infrastructure, as the learning is accelerated with CUDA. However, one can also execute the learning process on a CPU.

The learning is carried out with a Deep Q Network using a target-policy approach [1] and using priority replay memory [2] with a fingerprinting technique [3]. As this is a multi-agent problem with a shifting Q-function, we implemented a `fingerprint window' which enables the memory to forget about useless transitions from earlier in the learning process.

## Contents

This repo contains:

- A set of maintenance scenarios based on tokamak crashes, simulated by the JOREK-STARWALL simulation code [4,5].

- Gymnasium environments which define the structure of the tokamak maintenance problem.

- Modularised system logic which defines the logic of the robotic maintenance system and its environment.

- A multi-agent Deep Q Network (DQN) trainer which learns the ideal control policy for the scenario.

- A DQN -> Markov process translator, which allows the user to verify properties of the learnt policy.

## Requirements:

PyTorch and Gymnasium are required for learning. PyGame is required if you wish to render the system. The Storm model checker [6,7] is required to verify the properties of the control policy.

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

## Guide to the Input File

| Parameter                     | Default Value| Description |
| - | - | - |
| environment                   | TokamakMA-v2 | The environment version to use, from /tokamak |
| run\_id                       | None         | String to identify the run |
| scenario                      | None         | The scenario to use, from scenarios.py |
| system\_logic                 | engage\_MA   | The system logic to use, from /system_logic |
| multiagent                    | y            | Whether this is multi-agent or single-agent execution |
| nodes\_per\_layer             | 8            | For the DQN
| num\_hidden\_layers           | 2            | For the DQN
| batch\_size                   | 256          | For the priority replay memory
| buffer\_size                  | 100000       | For the priority replay memory
| memory\_sort\_frequency       | 5            | For the priority replay memory (in training episodes)
| optimisation\_frequency       | 10           | For the DQN (in environment steps)
| num\_training\_episodes       | 1000         | Training parameter
| max\_steps                    | 1000         | Training parameter
| epsilon\_decay\_type          | exponential  | The form of the epsilon (exploration rate) decay: exponential or linear |
| epsilon\_max                  | 0.95         | Exploration parameter
| epsilon\_min                  | 0.05         | Exploration parameter
| fingerprint\_window           | 500          | Lee-way on the fingerprint value, after which transtions will be low-priority |
| fingerprint\_mode             | episode      | Method of fingerprinting transitions: episode, optimisation_counter, or epsilon |
| reward\_sharing\_coefficient  | 0.5          | Degree to which agents share/mix rewards. 0.5 means each agent's reward is constructed from 50% of its 'own' reward and 50% of the sum of its colleagues' rewards |
| min\_epsilon\_time            | 0            | Exploration parameter
| max\_epsilon\_time            | 0            | Exploration parameter
| alpha                         | 0.0005       | Learning rate
| gamma                         | 0.5          | Discount factor
| tau                           | 0.05         | Target/policy network update parameter
| plot\_frequency               | 50           | Frequency of plotting progess, in training episodes
| checkpoint\_frequency         | 50           | Frequency of saving a checkpoint, in training episodes
| overwrite\_saved\_weights     | n            | Whether to overwrite a previously-saved weights file, should they have the same name ('n' gives the new file a random suffix) |
| evaluation\_weights\_file     | None         | File from which to load weights for evaluation |
| render\_evaluation            | n            | Whether to render the evaluation with PyGame |
| render\_evaluation\_deadlocks | y            | Whether to render deadlock traces after evaluation |
| num\_evaluation\_episodes     | 100          | Evaluation parameter |
| canonical\_fingerprint        | 1000         | The value of the fingerprint parameter to use when evaluating the learnt policy |
| render\_training              | n            | Whether to render training (a debug parameter; this will make training very slow |


 ### References:

[1] Shengbo Eben Li. ‘Deep Reinforcement Learning’. en. In: Reinforcement Learning for Sequential Decision and Optimal Control. Ed. by Shengbo Eben Li. Singapore: Springer Nature, 2023, pp. 365–402.

[2] Tom Schaul, John Quan, Ioannis Antonoglou and David Silver. Prioritized Experience Replay. en. arXiv:1511.05952 [cs]. Feb. 2016.

[3] Jakob Foerster, Nantas Nardelli, Gregory Farquhar, Triantafyllos Afouras, Philip H. S. Torr, Pushmeet Kohli and Shimon Whiteson. ‘Stabilising experience replay for deep multi-agent reinforcement learning’. In: Proceedings of the 34th International Conference on Machine Learning - Volume 70. ICML’17. Sydney, NSW, Australia: JMLR.org, Aug. 2017, pp. 1146–1155.

[4] M Hoelzl, GTA Huijsmans, SJP Pamela, M Becoulet, E Nardon, FJ Artola, B Nkonga, et al The JOREK non-linear extended MHD code and applications to large-scale instabilities and their control in magnetically confined fusion plasmas NF 61, 065001 (2021) 

[5] [www.jorek.eu](https://www.jorek.eu/)

[6] Christian Hensel, Sebastian Junges, Joost-Pieter Katoen, Tim Quatmann, and Matthias Volk, “The probabilistic model checker Storm,” Int. J. Softw. Tools Technol. Transf., no. 4, 2022.

[7] [www.stormchecker.org/](https://www.stormchecker.org/)

