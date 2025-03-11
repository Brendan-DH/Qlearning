from itertools import count
import numpy as np
import math

import torch
import sys
from collections import deque, OrderedDict


def evaluate_model_by_trial(dqn,
                            num_episodes,
                            env,
                            max_steps,
                            reset_options=None,
                            render=False):
    print("Evaluating...")

    if ("win" in sys.platform and render):
        print("Cannot render on windows...")
        render = False

    env.set_rendering(render)
    device = torch.device("cpu")
    print(f"Evaluation running on {device}.")

    steps = np.empty(num_episodes)
    deadlock_counter = 0
    deadlock_traces = deque([], maxlen=10)  # store last 10 deadlock traces

    for i in range(num_episodes):
        if (reset_options):
            obs_state, info = env.reset(options=reset_options.copy())
        else:
            obs_state, info = env.reset()

        states = [env.interpret_state_tensor(env.obs_state)]
        actions = []
        obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)
        rel_actions = ["move cc", "move_cw", "engage", "wait"]  # 0=counter-clockwise, 1=clockwise, 2=engage, 3=wait

        for t in count():

            # calculate action utilities and choose action
            action_utilities = dqn.forward(obs_tensor.unsqueeze(0))[0]  # why is this indexed?
            blocked = env.blocked_model(env, env.obs_state)
            print("blocked", blocked)
            print("blocked actions:",
                  [f"{math.floor(i / env.num_actions)}-{rel_actions[i % env.num_actions]} --- BLOCKED" if b else f"{math.floor(i / env.num_actions)}-{rel_actions[i % env.num_actions]}" for i, b in
                   enumerate(blocked)])
            action_utilities = torch.where(blocked, -1000, action_utilities)
            action = torch.argmax(action_utilities).item()

            # apply action to environment
            new_obs_state, reward, terminated, truncated, info = env.step(action)

            states.append(env.interpret_state_tensor(env.obs_state))
            actions.append(action)
            robot_no = math.floor(action / env.num_actions)
            rel_action = rel_actions[action % env.num_actions]
            print(f"{robot_no}-{rel_action}")

            obs_state = new_obs_state
            obs_tensor = torch.tensor(list(obs_state.values()), dtype=torch.float, device="cpu", requires_grad=False)

            done = terminated

            if (render):
                env.render_frame(states[-1], True)

            if (done or truncated or t > max_steps):
                # if (not torch.cuda.is_available()) : ticks.append(info["elapsed ticks"])
                # if (not torch.cuda.is_available()) : goal_resolutions.append(np.sum(info["goal_resolutions"]))
                if (int(num_episodes / 10) > 0 and i % int(num_episodes / 10) == 0):
                    print(f"{i}/{num_episodes} episodes complete")
                break

        if (not done):
            deadlock_traces.append(states)
            deadlock_counter += 1

        steps[i] = t

    print("Evaluation complete.")
    print(
        f"{'CONVERGENCE SUCCESSFUL' if deadlock_counter == 0 else 'FAILURE'} - Failed to complete {deadlock_counter} times")
    print(f"Percentage converged: {100 - (deadlock_counter * 100 / num_episodes)}")

    return states, actions, steps, deadlock_traces  # states, actions, ticks, steps
