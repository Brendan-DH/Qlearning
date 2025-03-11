import random
import numpy as np
from collections import deque, OrderedDict

from dqn.dqn_collections import DeltaTransition

class PriorityMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.warning = False
        self.memory = deque([], maxlen=capacity)  # this could be a tensor
        self.max_priority = 1
        self.bounds = []
        self.prob_divisor = np.NaN

    def push(self, *args):
        """Save a transition"""
        # when a new transition is saved, it should have max priority:
        self.memory.appendleft(DeltaTransition(*args, self.max_priority))  # append at the high-prio part.
        # print("mem size:", len(self.memory))
        if len(self.memory) == self.capacity and not self.warning:
            print("REPLAY AT CAPACITY: " + str(len(self)))
            self.warning = True

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sort(self, batch_size, priority_coefficient):
        # sort the transitions according to priority, i.e. according to delta
        # higher rank = lower priority, so higher rank should be lower |delta|
        # i.e. lower rank should be higher delta, as such:

        if (len(self.memory) < batch_size):
            print(f"Memory too empty to sort {len(self.memory)}/{batch_size}")
            return

        items = [self.memory.pop() for i in range(len(self.memory))]  # pop everything?
        # print(items)
        items.sort(key=(lambda x: -x.delta))  # do the sorting (descending delta)
        self.memory = deque(items, maxlen=self.capacity)

        self.max_priority = 1

        # the divisor in the P equation
        self.prob_divisor = 1 / np.sum([((1 / (i + 1)) ** priority_coefficient) for i in range(len(items))])

        # re-calculate the bounds
        # do this very explicitly for now
        bounds = np.zeros(batch_size, dtype=int)
        start = 0
        # print("Iterating over bounds...")
        for i in range(len(bounds)):  # iterate over segments
            prob_in_segment = 0
            for j in range(start,
                           start + self.capacity):  # the (inclusive) start is the (exclusive) end of the previous bound
                priority = 1 / (j + 1)  # wary of div by 0
                prob_in_segment += (priority ** priority_coefficient) * self.prob_divisor
                if (prob_in_segment >= (1 / batch_size)):
                    # conservative boundaries (j rather than j+1); this means the boundaries contain less than 1/batch_size the probability
                    # this ensures that the boundaries won't overflow the size of the memory
                    # however, also ensure one tr per segment as empty segments will break early optimisations
                    bounds[i] = j if j > start else j + 1  # assign the END of this segment (exclusive)
                    start = j if j > start else j + 1
                    break  # move on to the next boundary

        # assign the uppermost boundry as the end of the memory
        # this is a bit of an approximation but the last segment is full of only the least important transitions anyway
        bounds[-1] = len(self.memory)

        self.bounds = bounds

    def update_priorities(self, index, delta):
        tr = self.memory[index]
        self.memory[index] = DeltaTransition(tr.state, tr.action, tr.next_state, tr.reward, delta)

    def __len__(self):
        return len(self.memory)