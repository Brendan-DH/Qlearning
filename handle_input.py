
import sys
import select
import os
import scenarios
import importlib


def get_input_dict(input_dir="/inputs", print_inputs=True):

    input_dict = {
        "environment": "TokamakMA-v1",
        "run_id": None,
        "scenario": None,
        "system_logic": "blocking_MA",
        "multiagent": "y",
        "nodes_per_layer": 8,
        "num_hidden_layers": 2,
        "batch_size": 256,
        "buffer_size": 100000,
        "memory_sort_frequency": 5,
        "optimisation_frequency": 10,
        "num_training_episodes": 300,
        "max_steps": 200,
        "epsilon_decay_type": "exponential",
        "epsilon_max": 0.95,
        "epsilon_min": 0.05,
        "epsilon_window": 0.2,
        "reward_sharing_coefficient" : 0.5,
        "min_epsilon_time": 0,
        "max_epsilon_time": 0,
        "alpha": 0.0005,
        "gamma": 0.5,
        "tau": 0.05,
        "use_pseudorewards": "n",
        "plot_frequency": 50,
        "checkpoint_frequency": 50,
        "overwrite_saved_weights": "n",
        "evaluation_weights_file": None,
        "render_evaluation": "n",
        "render_evaluation_deadlocks": "y",
        "num_evaluation_episodes": 100,
        "evaluation_type": "mdp",
        "mc_order": "LIFO",
        "render_training": "n",
    }

    # this block prints out the default values if it doesn't detect them in "input_dir"
    if ("default_inputs.in" not in os.listdir(os.getcwd() + input_dir)):
        print(f"Saving default inputs to '{os.getcwd() + input_dir}/default_inputs.in'")
        with open(f"{os.getcwd() + input_dir}/default_inputs.in", "w") as file:
            file.write("# default input parameters for tokamak_trainer.py\n")
            for key, value in input_dict.items():
                file.write(f"{key} = {value}\n")
        file.close()

    # get input from stdin
    print("Attempting to read input file.")
    stdin = sys.stdin # get the input file from the standard input
    if (select.select([sys.stdin,],[],[],0.0)[0]):  # checks to make sure there IS any input
        for line in stdin:
            if (line[0] == "#" or len(line) == 0 or line.strip() == ""):  # skip comments and empty lines
                continue
            key, value = line.replace(" ", "").strip().split("=")
            if (key in input_dict.keys()):
                input_dict[key] = value  # overwrite the default value with what you read
            else:
                print(f"Warning: input variable '{key}' not understood, skipping.")
    else:
        print("No input file specified, exiting.")
        sys.exit(1)

    if print_inputs:
        dict_string = '\n'.join('{0}: {1}'.format(k, v)  for k,v in input_dict.items())
        print(f"Input dictionary:\n----\n{dict_string}\n----\n")

    return input_dict
