from dqn.dqn_collections import system_parameters

small_case1 = system_parameters(
    size=12,
    robot_locations=[1, 2, 3],
    goal_locations=[11, 5, 7],
    goal_discovery_probabilities=[0.95, 0.95, 0.95],
    goal_completion_probabilities=[0.95, 0.95, 0.95],
    goal_checked=[0, 0, 0],
    goal_activations=[0, 0, 0],
    elapsed_ticks=0,
)

case_5goals = system_parameters(
    size=12,
    robot_locations=[1, 2, 7],
    goal_locations=[11, 3, 5, 4, 6],
    goal_discovery_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7],
    goal_completion_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7],
    goal_activations=[0, 0, 0, 0, 0],
    goal_checked=[0, 0, 0, 0, 0],
    elapsed_ticks=0,
)

case_7goals = system_parameters(
    size=12,
    robot_locations=[1, 5, 6],
    goal_locations=[11, 3, 5, 4, 10, 9, 7],
    goal_discovery_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    goal_completion_probabilities=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    goal_activations=[0 for i in range(7)],
    goal_checked=[0 for i in range(7)],
    elapsed_ticks=0,
)

large_case_peaked = system_parameters(
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

seg20_case = system_parameters(
    size=20,
    robot_locations=[1, 5, 6],
    goal_locations=[i for i in range(20)],
    goal_completion_probabilities=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    goal_discovery_probabilities=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.7, 0.7, 0.3, 0.2, 0.7, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    goal_activations=[0 for i in range(20)],
    goal_checked=[0 for i in range(20)],
    elapsed_ticks=0,
)

rects_id19_case_peaked = system_parameters(
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

rects_id24 = system_parameters(size=20, robot_locations=[0, 7, 13],
                               goal_locations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                               goal_activations=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               goal_checked=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               goal_completion_probabilities=[0.8278440569619024, 0.922669854761093, 0.9765228479650532, 0.9881408238956123, 0.994422497414904, 0.9966218315330959, 0.9918902151544384,
                                                              0.9846729252811671, 0.971138401881295, 0.9564615460331654, 0.9214653420234962, 0.8414802238564458, 0.7554641207040974, 0.6363356852365863,
                                                              0.3819830279340779, 0.19999999999999996, 0.6867377557105544, 0.5728393073631655, 0.6419112208907641, 0.772361442642507],
                               goal_discovery_probabilities=[0.2044351823577408, 0.09182954747120214, 0.027879118041499333, 0.014082771623960355, 0.006623284319801586, 0.004011575054448599,
                                                             0.009630369504104352, 0.018200901228614043, 0.03427314776596226, 0.05170191408561609, 0.09325990634709827, 0.18824223417047056,
                                                             0.29038635666388435, 0.4318513737815537, 0.7338951543282825, 0.95, 0.37199891509371663, 0.5072533225062409, 0.42523042519221754,
                                                             0.27032078686202304],
                               elapsed_ticks=0)
