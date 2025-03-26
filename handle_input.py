
import sys
import select
import os
import scenarios
import importlib


def get_input_dict():

    input_dict = {
        "environment": "Tokamak-v15",
        "run_id": None,
        "scenario": None,
        "system_logic": "hybrid_system_tensor_logic",
        "nodes_per_layer": 256,
        "num_hidden_layers": 6,
        "batch_size": 256,
        "buffer_size": 50000,
        "memory_sort_frequency": 5,
        "num_training_episodes": 300,
        "max_steps": 200,
        "epsilon_decay_type": "exponential",
        "epsilon_max": 0.95,
        "epsilon_min": 0.05,
        "min_epsilon_time": 0,
        "max_epsilon_time": 0,
        "alpha": 0.05,
        "gamma": 0.8,
        "tau": 0.005,
        "use_pseudorewards": "n",
        "plot_frequency": 20,
        "checkpoint_frequency": 50,
        "overwrite_saved_weights": "n",
        "evaluation_weights_file": None,
        "render_evaluation": "n",
        "num_evaluation_episodes": 100,
        "evaluation_type": "mdp"
    }

    if ("default_inputs.in" not in os.listdir(os.getcwd() + "/inputs")):
        print(f"Saving default inputs to '{os.getcwd()}/inputs/default_inputs.in'")
        with open("inputs/default_inputs.in", "w") as file:
            file.write("# default input parameters for tokamak_trainer.py\n")
            for key, value in input_dict.items():
                file.write(f"{key} = {value}\n")
        file.close()

    # get input from stdin
    print("Attempting to read input file.")
    stdin = sys.stdin
    if (select.select([sys.stdin,],[],[],0.0)[0]):
        for line in stdin:
            if (line[0] == "#" or len(line) == 0):
                continue
            key, value = line.replace(" ", "").strip().split("=")
            if (key in input_dict.keys()):
                input_dict[key] = value
            else:
                print(f"Warning: input variable '{key}' not understood, skipping.")
    else:
        print("No input file specified, exiting.")
        sys.exit(1)

    dict_string = '\n'.join('{0}: {1}'.format(k, v)  for k,v in input_dict.items())
    print(f"Input dictionary:\n----\n{dict_string}\n----\n")

    return input_dict
