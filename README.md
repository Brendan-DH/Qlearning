
# A DQN for Tokamak Maintenance

This repo contains:

a.) A set of gymnasium environments which describe a tokamak maintenance scenario, and

b.) A DQN script which learns the ideal control policy for the scenario.

Pytorch and gymnasium are required. Pygame is technically also required, but it can be deleted from the imports of the gym environment you want to use, as long as you don't plan to render the system.

A set of sample_weights are provided for use with the following system parameters:

```python
num_robots = 3
size = 12
goal_locations = [11,5,6,0,3,2]
goal_probabilities = [0.1, 0.9, 0.5, 0.1, 0.9, 0.5]

env = gym.make("Tokamak-v4",
               num_robots=num_robots,
               size=size, num_goals=len(goal_locations),
               goal_locations=goal_locations,
               goal_probabilities = goal_probabilities,
               render_mode = None )
env_options = {"robot_locations" : [1,3,5]}
env.reset(options=env_options)
```

## Setting up the tokamak gymnasium environment

The repo contains a script `intitialise` which performs most of the set up.  Calling this will:

- load the correct python3.8 module (this should be commented out if not using `module`),
- create a python3.8 virtual environment,
- load the requirements.txt of this environment,
- register the relevant environment wtih gymnasium (the environment number should be changed to whatever is required).



