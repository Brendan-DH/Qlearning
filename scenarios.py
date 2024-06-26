import DQN

small_case1 = DQN.system_parameters(
    size=12,
    robot_locations=[1, 2, 3],
    goal_locations=[11, 5, 7],
    goal_discovery_probabilities=[0.95, 0.95, 0.95],
    goal_completion_probabilities=[0.95, 0.95, 0.95],
    goal_checked=[0, 0, 0],
    goal_activations=[0, 0, 0],
    elapsed_ticks=0,
)


case_5goals = DQN.system_parameters(
    size=12,
    robot_locations=[1, 2, 7],
    goal_locations=[11, 3, 5, 4, 6],
    goal_discovery_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7],
    goal_completion_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7],
    goal_activations=[0, 0, 0, 0, 0],
    goal_checked=[0, 0, 0, 0, 0],
    elapsed_ticks=0,
)

case_7goals = DQN.system_parameters(
    size=12,
    robot_locations=[1, 5, 6],
    goal_locations=[11, 3, 5, 4, 10, 9, 7],
    goal_discovery_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    goal_completion_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    goal_activations=[0 for i in range(7)],
    goal_checked=[0 for i in range(7)],
    elapsed_ticks=0,
)


large_case_peaked = DQN.system_parameters(
    size=12,
    robot_locations=[1, 5, 6],
    goal_locations=[i for i in range(12)],
    goal_completion_probabilities=[
        0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95],
    goal_discovery_probabilities=[
        0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95],
    goal_activations=[0 for i in range(12)],
    goal_checked=[0 for i in range(12)],
    elapsed_ticks=0,
)

rects_id19_case_peaked = DQN.system_parameters(
    size=12,
    robot_locations=[1, 5, 6],
    goal_locations=[i for i in range(12)],
    goal_completion_probabilities=[0.6323834, 0.32501727, 0.30504792, 0.1, 0.51242514, 0.76339031, 0.85989944,
                                   0.89268024, 0.89840738, 0.89732351, 0.87692726, 0.76291351],
    goal_discovery_probabilities=[0.95 for i in range(12)],
    goal_activations=[0 for i in range(12)],
    goal_checked=[0 for i in range(12)],
    elapsed_ticks=0,
)