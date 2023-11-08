
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

The directory `/environments` contains various versions of the tokamak environment ("tokamakenv[version].py"). This environment must be registered to the local installation of gymnasium before they can be used in tokamak_DQN.py.

To register the environments:

1. Navigate to your gymnasium installation (e.g. `/home/[user]/anaconda3/lib/python3.8/site-packages/gymnasium`).
2. In `/envs`, create a directory called `tokamak`.
3. Inside `/envs/tokamak`, create a symlink to the tokamakenv that you want to use (e.g. tokamakenv4.py).
4. Also inside `/envs/tokamak`, create `__init__.py` and add the following line using the corresponding version number:
    
    ```python
    from gymnasium.envs.tokamak.tokamakenv4 import TokamakEnv4
    ```

5. Navigate back to `/envs` and open `__init__.py`. Add the following fragment, again using your chosen version number. Choose `max_episode_steps` as needed.
    
    ```python
    register(
     id='Tokamak-v4',
     entry_point='gymnasium.envs.tokamak:TokamakEnv4',
     max_episode_steps=500,
    )
    ```

Using the above snippets, gymnasium should now be able to make the "Tokamak-v4" environment from within tokamak_DQN.py or wherever else you need it.



