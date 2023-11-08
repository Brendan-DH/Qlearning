
# A DQN for Tokamak Maintenance

This repo contains:

a.) A set of gymnasium environments which describe a tokamak maintenance scenario, and

b.) A DQN script which learns the ideal control policy for the scenario.

Pytorch and gymnasium are required.

## Setting up the tokamak gymnasium environment

The directory `/environments` contains various versions of the tokamak environment ("tokamakenv[version].py"). This environment must be registered to the local installation of gymnasium before they can be used in tokamak_DQN.py.

To register the environments:

1. Navigate to your gymnasium installation (e.g. `/home/[user]/anaconda3/lib/python3.8/site-packages/gymnasium`).
2. In `/envs`, create a directory called `tokamak`.
3. Inside `/envs/tokamak`, create a symlink to the tokamakenv that you want to use (e.g. tokamakenv4.py).
4. Also inside `/envs/tokamak`, create `__init__.py` and add the following line:
    
    ```python
    from gymnasium.envs.tokamak.tokamakenv4 import TokamakEnv4
    ```

5. Navigate back to `/envs` and open `__init__.py`. Add the following fragment. Choose `max_episode_steps` as needed.
    
    ```python
    register(
     id='Tokamak-v4',
     entry_point='gymnasium.envs.tokamak:TokamakEnv4',
     max_episode_steps=500,
    )
    ```

Gymnasium should now be able to make the "Tokamak-v4" environment from within tokamak_DQN or wherever else you need it.



