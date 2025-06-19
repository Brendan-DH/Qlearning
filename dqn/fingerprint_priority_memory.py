import random
import numpy as np
import sys 
from collections import deque, OrderedDict

from dqn.dqn_collections import FingerprintDeltaTransition

class FingerprintPriorityMemory(object):

    def __init__(self, capacity,fingerprint_type, fingerprint_window=1):
        self.capacity = capacity
        self.warning = False
        self.memory = deque([], maxlen=capacity)  # this could be a tensor
        self.max_priority = 1
        self.bounds = []
        self.prob_divisor = np.NaN
        self.memory_type = "fingerprint_priority"
        self.fingerprint_type = fingerprint_type
        self.fingerprint_window = fingerprint_window  # the epsilon window for the epsilon-greedy policy
        
        print(f"Initialised fingerprint priority memory with capacity {self.capacity} and fingerprint window {self.fingerprint_window}")

    def push(self, *args):
        """Save a transition"""
        # when a new transition is saved, it should have max priority:
        # print(*args)
        self.memory.appendleft(FingerprintDeltaTransition(*args, self.max_priority))  # append at the high-prio part.
        # print("mem size:", len(self.memory))
        if len(self.memory) == self.capacity and not self.warning:
            print("REPLAY AT CAPACITY: " + str(len(self)))
            self.warning = True

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sort(self, batch_size, priority_coefficient = 1, current_fingerprint = 1, window_direction = -1):
        # sort the transitions according to priority, i.e. according to delta
        # higher rank = lower priority, so higher rank should be lower |delta|
        # i.e. lower rank should be higher delta, as such:

        if (len(self.memory) < batch_size):
            return

        if self.fingerprint_type == "epsilon":
            for i in range(len(self.memory)):
                tr = self.memory[i]
                if tr.fingerprint > self.fingerprint_window + current_fingerprint:
                    self.memory[i] = FingerprintDeltaTransition(tr.state, tr.action,
                                                                tr.next_state, tr.reward, tr.blocked, tr.fingerprint, 0)
                    
        elif self.fingerprint_type == "episode" or self.fingerprint_type == "optimisation_counter":
            for i in range(len(self.memory)):
                tr = self.memory[i]
                if tr.fingerprint <  current_fingerprint - self.fingerprint_window:
                    # print("culling transition with fingerprint", tr.fingerprint)
                    self.memory[i] = FingerprintDeltaTransition(tr.state, tr.action,
                                                                tr.next_state, tr.reward, tr.blocked, tr.fingerprint, 0)
        else:
            print("Error with fingerprint window. Was the fingerprint type specified?")
            sys.exit(1)
            
        items = list(self.memory)
        items.sort(key=(lambda x: -x.delta))  # do the sorting (descending delta)
        self.memory = deque(items, maxlen=self.capacity)

        self.max_priority = 1

        # the divisor in the P equation
        # this is rank-based priority; the divisor 1/(sum of all priorities)
        # latex: \left( \sum_{i=0}^{N-1} \left( \frac{1}{i+1} \right)^\alpha \right)^{-1}
        self.prob_divisor = 1 / np.sum([((1 / (i + 1)) ** priority_coefficient) for i in range(len(items))])

        # re-calculate the bounds
        bounds = np.zeros(batch_size, dtype=int)
        start = 0
        # print("Iterating over bounds...")
        for i in range(len(bounds)):  # iterate over segments
            prob_in_segment = 0
            for j in range(start, start + self.capacity):  # the (inclusive) start is the (exclusive) end of the previous bound
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
        self.memory[index] = FingerprintDeltaTransition(tr.state, tr.action, tr.next_state, tr.reward, tr.blocked, tr.fingerprint, delta)

    def __len__(self):
        return len(self.memory)