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

rects_21 = system_parameters(size=20,
                             robot_locations=[0, 7, 13],
                             goal_locations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                             goal_activations=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             goal_checked=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             goal_completion_probabilities=[0.9903497758879076, 0.9878980416757515, 0.9855220635392712, 0.975622469947035, 0.866107790806471, 0.930708099636183, 0.946287381902569,
                                                            0.8859712671955788, 0.7060459003115704, 0.09999999999999998, 0.21937200012809777, 0.1411699940314225, 0.9448718577998596,
                                                            0.9922939242471513,
                                                            0.9950572303869684, 0.9968548949687644, 0.9972032456334905, 0.9945329981684494, 0.9944547891001908, 0.993665782502539],
                             goal_discovery_probabilities=[0.00857797698852657, 0.010757296288220919, 0.01286927685398116, 0.02166891560263555, 0.11901529706091461, 0.061592800323392906,
                                                           0.04774454941993863,
                                                           0.10135887360392988, 0.2612925330563819, 0.8, 0.6938915554416909, 0.7634044497498467, 0.04900279306679149, 0.0068498451136432235,
                                                           0.004393572989361445,
                                                           0.0027956489166538036, 0.0024860038813417544, 0.004859557183600487, 0.004929076355385904, 0.005630415553298685],
                             elapsed_ticks=0)
