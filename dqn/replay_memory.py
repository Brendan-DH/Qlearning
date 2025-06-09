import random
import numpy as np
from collections import deque, OrderedDict

from dqn.dqn_collections import Transition
import sys 

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.warning = False
        self.memory = deque([], maxlen=capacity)  # this could be a tensor
        self.max_priority = 1
        self.bounds = []
        self.prob_divisor = np.NaN
        self.memory_type = "replay"

    def push(self, *args):
        """Save a transition"""
        # when a new transition is saved, it should have max priority:
        # print(*args)
        self.memory.appendleft(Transition(*args))  # append at the high-prio part.

        trans = self.memory[0]  # the most recent transition

        if(trans.blocked[trans.action] ==1):
                        print(f"ADDED INCORRECTLY: action {trans.action} in transition {i} was blocked!!!!!!!!!!!!!!!!!!!")
                        print(trans)
                        sys.exit(1)
        
                            
        # print("mem size:", len(self.memory))
        if len(self.memory) == self.capacity and not self.warning:
            print("REPLAY AT CAPACITY: " + str(len(self)))
            self.warning = True

    def check_memory(self):
    
        for i in range(len(self.memory)):
            trans = self.memory[i]
            if(trans.blocked[trans.action] ==1):
                print(f"WARNING: action {trans.action} in transition {i} was blocked!!!!!!!!!!!!!!!!!!!")
                print(trans)
            return False

        # somehow the memory is being ruined

        print("Memory check passed, no blocked actions found.")
        return True

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)