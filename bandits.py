from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
            self.probas.sort()
            
            self.arms_cost=[np.random.random() for x in range(self.n)]
            self.arms_cost.sort()
            
            
        
            self.arms_reward=[1-x for x in self.arms_cost]
            #self.arms_cost=[np.random.uniform(0,self.arms_cost[x]) for x in range(self.n)]
            
            self.util = [((self.probas[_] * self.arms_reward[_]) - ((1 - self.probas[_]) * self.arms_cost[_])) for _ in range(self.n) ]
            
            self.random_values = [np.random.random() for x in range(self.num_of_steps)]
            
        else:
            self.probas = probas

        self.best_proba = max(self.util)
        

    def generate_reward(self, i, time):
        # The player selected the i-th machine.
        if self.random_values[time] < self.probas[i]:
            return 1
        else:
            return 0
