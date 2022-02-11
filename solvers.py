from __future__ import division

import numpy as np
import time
from scipy.stats import beta

from bandits import BernoulliBandit


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        
        self.time = 0  #time of run
        
        self.stopping_criteria = 0

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.util[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            self.time = _
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)
            
        

class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps

        self.estimates = [init_proba] * self.bandit.n  # Optimistic initialization

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one.
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class UCB(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n
        self.mean = [0] * self.bandit.n
        self.ucb = [0] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1
        
        #compute utility value
        
        for x in range(self.bandit.n):
            self.mean[x] = self.estimates[x]
            self.ucb[x] = self.mean[x] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x]))

        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: (self.ucb[x] * self.bandit.arms_reward[x]) - 
            ((1 - self.ucb[x]) * self.bandit.arms_cost[x]))
            
        r = self.bandit.generate_reward(i, self.time)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i
        
class UCB1(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n
        self.mean = [0] * self.bandit.n
        self.ucb = [0] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1
        
        #compute utility value
        
        for x in range(self.bandit.n):
            self.mean[x] = self.estimates[x]
            self.ucb[x] = self.mean[x] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x]))

        for i in range(self.bandit.n):
    
          if i > 0:
            
            if self.mean[i - 1] < self.mean[i] and self.ucb[i - 1] > self.ucb[i]:
                self.ucb[i - 1] = self.ucb[i]
                
            elif self.mean[i - 1] > self.mean[i] and self.ucb[i - 1] > self.ucb[i]:
                if (self.ucb[i - 1] - self.mean[i - 1]) > (self.ucb[i] - self.mean[i]):
                   self.ucb[i - 1] = self.ucb[i] 
                elif (self.ucb[i - 1] - self.mean[i - 1]) < (self.ucb[i] - self.mean[i]):
                   self.ucb[i] = self.ucb[i - 1] 
                else:
                   self.ucb[i - 1] = self.ucb[i] 
                
                          
            elif self.mean[i - 1] == self.mean[i] and self.ucb[i - 1] > self.ucb[i]:
                self.ucb[i- 1] = self.ucb[i] 
                
        
        for i in range(self.bandit.n - 1):
          if self.ucb[self.bandit.n - 1 - i] <= self.ucb[self.bandit.n - 1 - (i + 1)]:
            self.ucb[self.bandit.n - 1 - (i + 1)] = self.ucb[self.bandit.n - 1 - i] #- 0.05 
           

        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: (self.ucb[x] * self.bandit.arms_reward[x]) - 
            ((1 - self.ucb[x]) * self.bandit.arms_cost[x]))
            
        r = self.bandit.generate_reward(i, self.time)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        self.mean = [0] * self.bandit.n
        self.ucb = [0] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]
    
    @property
    def ucb_s(self):
        return self.ucb
        
    def run_one_step(self):
    
        #compute utility value
        
        for x in range(self.bandit.n):
            self.mean[x] = self._as[x] / float(self._as[x] + self._bs[x])
            self.ucb[x] = self.mean[x] + beta.std(self._as[x], self._bs[x]) * self.c 
            
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            key=lambda x: (self.ucb[x] * self.bandit.arms_reward[x]) - 
            ((1 - self.ucb[x]) * self.bandit.arms_cost[x]))
        
        r = self.bandit.generate_reward(i, self.time)

        # Update Gaussian posterior
        self._as[i] += r
        self._bs[i] += (1 - r)
        
        
        return i

class BayesianUCB1(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BayesianUCB1, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        self.mean = [0] * self.bandit.n
        self.rad = [0] * self.bandit.n #Stands for radiu - confidence interval 
        self.ucb = [0] * self.bandit.n
        self.lcb = [0] * self.bandit.n
        self.prev_lcb = [0] * self.bandit.n
        
        self.prev_i = 0

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]
    
    @property
    def ucb_s(self):
        return self.ucb
       
    def run_one_step(self):
    
        #compute utility value
        #if self.stopping_criteria == 0:
         for x in range(self.bandit.n):
            self.mean[x] = self._as[x] / float(self._as[x] + self._bs[x])
            self.rad[x] = (beta.std(self._as[x], self._bs[x]) * self.c)
            self.ucb[x] = self.mean[x] + self.rad[x] 
            self.lcb[x] = self.mean[x] - self.rad[x]
            
         
         for i in range(self.bandit.n):
    
          if i > 0:
          
            
            if self.mean[i - 1] < self.mean[i] and self.ucb[i - 1] > self.ucb[i]:
                self.ucb[i - 1] = self.ucb[i] #- 0.05
                
            elif self.mean[i - 1] > self.mean[i] and self.ucb[i - 1] > self.ucb[i]:
                if (self.ucb[i - 1] - self.mean[i - 1]) > (self.ucb[i] - self.mean[i]):
                   self.ucb[i - 1] = self.ucb[i] #- 0.05
                elif (self.ucb[i - 1] - self.mean[i - 1]) < (self.ucb[i] - self.mean[i]):
                   self.ucb[i] = self.ucb[i - 1] #+ 0.05
                else:
                   self.ucb[i - 1] = self.ucb[i] #- 0.05
                
                          
            elif self.mean[i - 1] == self.mean[i] and self.ucb[i - 1] > self.ucb[i]:
                self.ucb[i- 1] = self.ucb[i]  #- 0.05
                
         '''  
            
            if self.mean[i - 1] < self.mean[i] and self.rad[i - 1] > self.rad[i]:
                self.rad[i - 1] = self.rad[i] #- 0.05
                
            elif self.mean[i - 1] > self.mean[i] and self.rad[i - 1] > self.rad[i]:
                
                   self.rad[i - 1] = self.rad[i] #- 0.05
                
                          
            elif self.mean[i - 1] == self.mean[i] and self.rad[i - 1] > self.rad[i]:
                self.rad[i- 1] = self.rad[i]  #- 0.05
                 
        
         for i in range(self.bandit.n - 1):
          if self.rad[self.bandit.n - 1 - i] <= self.rad[self.bandit.n - 1 - (i + 1)]:
            self.rad[self.bandit.n - 1 - (i + 1)] = self.rad[self.bandit.n - 1 - i] #- 0.05 
         
         for i in range(self.bandit.n):
           self.ucb[i] = self.mean[i] + self.rad[i]
         '''
         for i in range(self.bandit.n - 1):
          if self.ucb[self.bandit.n - 1 - i] <= self.ucb[self.bandit.n - 1 - (i + 1)]:
            self.ucb[self.bandit.n - 1 - (i + 1)] = self.ucb[self.bandit.n - 1 - i] #- 0.05 
         
         i = max(
            range(self.bandit.n),
            key=lambda x: (self.ucb[x] * self.bandit.arms_reward[x]) - 
            ((1 - self.ucb[x]) * self.bandit.arms_cost[x])
         )
         
        # for a in range(self.bandit.n):
         '''
         if self.ucb[i] <= self.prev_lcb[self.prev_i]:
              #i = a
              self.stopping_criteria = 1
         self.prev_i = i
         self.prev_lcb = self.lcb
         '''
         
         r = self.bandit.generate_reward(i, self.time)

        # Update Gaussian posterior
         self._as[i] += r
         self._bs[i] += (1 - r)
         
         '''
        #else: 
         i = self.prev_i
         r = self.bandit.generate_reward(i, self.time)

         # Update Gaussian posterior
         self._as[i] += r
         self._bs[i] += (1 - r)
         '''
         return i

class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit)

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
'''

class UCB1(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1
        
        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
            
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i



class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
                self._as[x], self._bs[x]) * self.c
        )
        r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
'''        
        