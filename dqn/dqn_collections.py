from collections import namedtuple, deque, OrderedDict


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'blocked'))

PriorityTransition = namedtuple('PriorityTransition',
                                ('state', 'action', 'next_state', 'reward', 'priority'))

DeltaTransition = namedtuple('DeltaTransition',
                             ('state', 'action', 'next_state', 'reward', 'blocked', 'delta'))

system_parameters = namedtuple("system_parameters",
                               ("size",
                                "robot_locations",
                                "goal_locations",
                                "goal_activations",
                                "goal_checked",
                                "goal_completion_probabilities",
                                "goal_discovery_probabilities",
                                "elapsed_ticks"
                                ))

#
# class hashdict(dict):
#     def __hash__(self):
#         return hash(frozenset(self))


class FiniteDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = OrderedDict()

    def __setitem__(self, key, value):
        if key not in self.data:
            self.data[key] = 0  # Initialize missing keys with 0, keeping the order
        self.data.move_to_end(key)  # Move the key to the end (most recent)
        self.data[key] = value

        if len(self.data) > self.max_size:
            self.data.popitem(last=False)  # Remove the oldest item (FIFO)

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = 0  # Set the default value to 0 if key is missing
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __repr__(self):
        return repr(self.data)