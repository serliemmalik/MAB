from __future__ import division

import numpy as np
import time

import scipy
import scipy.stats as stats
import pdb

import random

# Writing to an excel 
# sheet using Python
import xlwt
from xlwt import Workbook
import xlrd
from xlutils.copy import copy

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
        self.actions = []  # A list of machine ids, 0 to bandit.n - 1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.ab = 0.
        
        
        self.t_ = 0
        self.t_ucbs = 0
        self.t_ucbs2 = 0
        self.q = 0
        self.p = 0
        self.s = 0
        self.tt = 0
        
        self.probass = [0 for i in range(self.bandit.n)] 
        
        
        
        self.counts_multiple = [0] * self.bandit.n
        self.actions_multiple = []  # A list of machine ids, 0 to bandit.n - 1.
        self.regret_multiple = 0.  # Cumulative regret.
        self.regrets_multiple = [0.]  # History of cumulative regret.
        self.ab_multiple = 0.
        self.utility_multiple = 0
        self.utili = 0
        self.arm_history_multiple =[[1,1] for x in range(self.bandit.n)]
        
        
        print("Likelhoods:      ", self.bandit.probas)
        
        print ("Arms Cost:  ", self.bandit.arms_cost)

        #self.arms_reward=[1-x for x in self.arms_cost]
        #self.arms_reward.sort()
        #self.arms_reward=[0.91,0.3,0.4,0.6,0.9]
        print ("Rewards: ", self.bandit.arms_reward)
        
              

        
        self.util = [0 for a in range(self.bandit.n)]
        self.util_multiple = [0 for a in range(self.bandit.n)]
        
        for xx in range(self.bandit.n):
          self.util[xx] = (self.bandit.probas[xx] * self.bandit.arms_reward[xx]) - (1 - self.bandit.probas[xx]) * self.bandit.arms_cost[xx] 
          print(self.util[xx])
        print("Util:    ",self.util) 
        self.best_util = max(self.util) 
        print("Best Util:  ", self.best_util)
      
        
    def update_regret(self, i):
        # i (int): index of the selected machine.
        #self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regret += (self.best_util - self.util[i])
        self.regrets.append(self.regret)
        
    def increment(self):
        self.t_ += 1
    
    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError
        

    def run(self, num_steps):
    
        assert self.bandit is not None
                    
        
        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n - 1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.ab = 0.
        self.utili = 0
        
        self.arm_history =[[1,1] for x in range(self.bandit.n)]
        
        #print(s.counts)
                        
        q = 0
        p = 0
                
        trial_times = 200
        #for a in range(trial_times):
        for _ in range(num_steps):
            i = self.run_one_step()
            
            #r = self.bandit.generate_reward(i)
            #r = np.random.random()
            #r = random.uniform(-1.0,1.0)
            '''
            workbook = xlrd.open_workbook('random_numbers.xls')
            worksheet = workbook.sheet_by_index(0)
            
            r = worksheet.cell(_,self.bandit.ran_col).value #rowx, colx
            '''
        
            r = self.bandit.random_values[_]
            
            #if r == 0:
            if r > self.bandit.probas[i]:
            #if r > self.util[i]:
                self.arm_history[i][1] += 1
                self.utili -= self.bandit.arms_cost[i]
                
                
                self.utility_multiple -= self.bandit.arms_cost[i]
                self.arm_history_multiple[i][1] += 1
                #self.regret_a[i]+= self.arms_cost[i]
            else:
                self.arm_history[i][0] += 1
                self.utili += self.bandit.arms_reward[i]
                
                
                self.utility_multiple += self.bandit.arms_reward[i]
                self.arm_history_multiple[i][0] += 1
           
            #print(self.arm_history)
 
            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)
            self.ab += 1
            
            self.t_ += 1
        
        

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

        print(self.estimates)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i
        
class UCB2(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB2, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n
        self.arm_history =[[init_proba,init_proba] for x in range(self.bandit.n)]
        self.ucbs_util = [init_proba] * self.bandit.n
        

    @property
    def estimated_probas(self):
        return [self.arm_history[i][0] / float(self.arm_history[i][0] + self.arm_history[i][1]) for i in range(self.bandit.n)]
    
    @property    
    def estimated_ucbs(self):
        return self.ucbs
        
    @property   
    def co(self):
       self.t_ucbs2 += 1
       return self.t_ucbs2
    

    def run_one_step(self):
    
        self.t += 1
        highest = -1
        highest_index = -1
       
        #print()
        for x in range(len(self.arm_history)):
        
            a=self.arm_history[x][0]
            b=self.arm_history[x][1]
            if a==0:
               a=1
            if b==0:
               b=1
            if self.t == 0:
               self.t = 1
        
            mean = a / (a + b)
            total_interactions = a + b
            sd = (a * b) / ((total_interactions + 1) * (total_interactions**2))
            ucb_a = mean + (4 * np.sqrt(sd))
            
            
            ucb = ucb_a * self.bandit.arms_reward[x] - ((1 - ucb_a) * self.bandit.arms_cost[x])
            #ucb = ucb_a 
            
            self.estimates[x] = mean
            
            self.ucbs_util[x] = ucb   
                  
        for x in range(len(self.ucbs_util)):           
            if self.ucbs_util[x] > highest:
               highest = self.ucbs_util[x]
               highest_index = x
            elif self.ucbs_util[x] == highest:
                chosen = random.choice((highest_index, x))
                highest = self.ucbs_util[chosen]
                highest_index = chosen 
           
            #print(mean)
        self.t_ucbs2 += 1
        return highest_index
        
        
class UCB3(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB3, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n
        self.arm_history =[[init_proba,init_proba] for x in range(self.bandit.n)]
        
        self.ucbs = [init_proba] * self.bandit.n
        self.ucbs_util = [init_proba] * self.bandit.n
        
        self.prev_ucb = [0 for x in range(len(self.arm_history))]
        

    @property
    def estimated_probas(self):
        return [self.arm_history[i][0] / float(self.arm_history[i][0] + self.arm_history[i][1]) for i in range(self.bandit.n)]
    
    @property    
    def estimated_ucbs(self):
        return self.ucbs
        
    @property    
    def estimated_ucbs_util(self):
        return self.ucbs_util
        
    @property   
    def co(self):
       self.t_ucbs += 1
       return self.t_ucbs
       
       
    def recursive_function(self, prev_u, av_list):
        
                 
        for i in range(len(av_list)):
           if i > 0:
              if prev_u[i - 1] <= prev_u[i] and i == len(prev_u):
                 break
              else: 
                 for i in range(len(prev_u)):
    
                   if i > 0:
            
                    if av_list[i - 1][0] < av_list[i][0] and av_list[i - 1][1] > av_list[i][1]:
                         av_list[i - 1][1] = av_list[i][1] 
               
                          
                    elif av_list[i - 1][0] > av_list[i][0] and av_list[i - 1][1] > av_list[i][1]:
                      if av_list[i - 1][1] - av_list[i - 1][0] > av_list[i][1] - av_list[i][0]:
                        av_list[i - 1][1] = av_list[i][1] 
                      elif av_list[i - 1][1] - av_list[i - 1][0] < av_list[i][1] - av_list[i][0]:
                        av_list[i][1] = av_list[i - 1][1]
                      else:
                        av_list[i - 1][1] = av_list[i][1]
                
                          
                    elif av_list[i - 1][0] == av_list[i][0] and av_list[i - 1][1] > av_list[i][1]:
                      av_list[i- 1][1] = av_list[i][1]
                    
                    for i in range(len(prev_u)):
        
                      prev_u[i] = av_list[i][1]
        return av_list
    
    def run_one_step(self):
        highest_between_two_arms = -1
        highest_between_two_arms_index = -1
    
        picked_arm_index=-1
        pick_arm_ucb=-1
        
        highest = -1
        highest_index = -1

        #av = newmethod(arm_history)
        
        av = [[0,0] for x in range(len(self.arm_history))]
        ucbs_util = [0 for x in range(len(self.arm_history))]
        
        for i in range(len(self.arm_history)):
        
           a=self.arm_history[i][0]
           b=self.arm_history[i][1]
           if a==0:
            a=1
           if b==0:
            b=1
           if self.t == 0:
            self.t = 1
       
           total_interactions = a + b
           
           mean = a / total_interactions
           sd = a * b / ((total_interactions + 1) * total_interactions ** 2)
           #sd = beta.std(a, b)
           ucb = mean + 4 * (np.sqrt(sd))
        
           av[i][0] = mean
           av[i][1] = ucb
           
           self.estimates[i] = av[i][0]
        
        for i in range(len(av)):
           if av[i][1] > self.prev_ucb[i] and self.prev_ucb[i] > 0 :
               av[i][1] = self.prev_ucb[i]
        
           
        
        for i in range(len(av)):
    
          if i > 0:
            
            if av[i - 1][0] < av[i][0] and av[i - 1][1] > av[i][1]:
                av[i - 1][1] = av[i][1]
                                 
                                 
            elif av[i - 1][0] > av[i][0] and av[i - 1][1] > av[i][1]:
                if av[i - 1][1] - av[i - 1][0] > av[i][1] - av[i][0]:
                   av[i - 1][1] = av[i][1]
                elif av[i - 1][1] - av[i - 1][0] < av[i][1] - av[i][0]:
                   av[i][1] - av[i - 1][1]
                else:
                   av[i - 1][1] = av[i][1] 
                          
                          
            elif av[i - 1][0] == av[i][0] and av[i - 1][1] > av[i][1]:
                av[i- 1][1] = av[i][1] 
                
                   
        for i in range(len(av) - 1):
          if av[len(av) - 1 - i][1] < av[len(av) - 1 - (i + 1)][1]:
           av[len(av) - 1 - (i + 1)][1] = av[len(av) - 1 - i][1] 
           
        for i in range(len(av)):
        
               self.prev_ucb[i] = av[i][1]
               
        #av = self.recursive_function(self.prev_ucb, av)
                               
        
        for i in range(len(av)):
           
           arm_ucb = (av[i][1] * self.bandit.arms_reward[i]) - ((1 - av[i][1]) * self.bandit.arms_cost[i])
           
           self.ucbs_util[i] = arm_ucb
 
        for x in range(len(self.ucbs_util)):           
            if self.ucbs_util[x] > highest:
               highest = self.ucbs_util[x]
               highest_index = x
            elif self.ucbs_util[x] == highest:
                chosen = random.choice((highest_index, x))
                highest = self.ucbs_util[chosen]
                highest_index = chosen 
                   
        self.t_ucbs += 1
        return highest_index 
 



class NewUCB1(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(NewUCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n
        #self.estimates = [0.1, 0.2, 0.4, 0.5, 0.5]

    @property
    def estimated_probas(self):
        return self.estimates
        
    def update_estimates(self, proba_index_i):
        for _ in range(self.bandit.n):
           if _ == proba_index_i :
              for a in range(self.bandit.n):
                 if a < proba_index_i :
                 
                   if self.estimates[a] <= self.estimates[proba_index_i]:
                   
                       print 
                   
                   elif self.estimates[a] > self.estimates[proba_index_i]:
                   
                       self.estimates[a] = self.estimates[proba_index_i]
                 
                 elif a > proba_index_i :
                 
                   if self.estimates[a] >= self.estimates[proba_index_i]:
                   
                       print
                   
                   elif self.estimates[a] < self.estimates[proba_index_i]:
                   
                       self.estimates[a] = self.estimates[proba_index_i]
           else:           
              print
 

    def run_one_step(self):
        self.t += 1

        if self.t == 1 :
        
           i = 4
           
        # Pick the best one with consideration of upper confidence bounds.
           b = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
           2 * np.log(self.t) / (1 + self.counts[x])))
           
           for _ in range(self.bandit.n):
               print#(self.estimates[_] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[_])))
        else: 
        
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
            
            for _ in range(self.bandit.n):
               print#(self.estimates[_] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[_])))
               
        
        print(self.estimates)
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        
        self.update_estimates(i)
        #print(self.estimates)
        #print("reward", r)
        #print("Trial num", self.t)
        #print("Counts ",self.counts)
        #print("Arm position ",i)
        #print("Estimates ", self.estimates)        

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
        
        for x in range(self.bandit.n):
            samp_list = self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
                self._as[x], self._bs[x]) * self.c
           
                
        #print(samp_list)

        return i


class NewBayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(NewBayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        self._util = [0.03, 0.07, 0.15, 0.25, 0.5]
        self._r = 1

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i])  for i in range(self.bandit.n)]

    def reward_result(self):
        return self.bandit.generate_reward(i)
    
    def estimated_eu(self):
        if self._r == 1:
           [(self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
           self._as[x], self._bs[x]) * self.c) * self._util[x] 
           for x in range(self.bandit.n)
           ]
        else:
           [(self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
           self._as[x], self._bs[x]) * self.c)
           for x in range(self.bandit.n)           
           ]
                
    
    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        
        if self._r == 1:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
           self._as[x], self._bs[x]) * self.c) * self._util[x] 
           for x in range(self.bandit.n)]
        else:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
           self._as[x], self._bs[x]) * self.c) * 1
           for x in range(self.bandit.n)]
        
        i = max(
            range(self.bandit.n),
            key=lambda x: samples[x]
        )
        self._r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += self._r
        self._bs[i] += (1 - self._r)
        
        #print(samples)
        
           
        return i


class BernoulliBayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BernoulliBayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        self._util = [0.03, 0.07, 0.15, 0.25, 0.5]
        self._r = 1

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i])  for i in range(self.bandit.n)]

    def reward_result(self):
        return self.bandit.generate_reward(i)
    
    def estimated_eu(self):
        if self._r == 1:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x])) + ((beta.std(
           self._as[x], self._bs[x]) * self.c)/ np.sqrt(1 + self.count[x]))
           for x in range(self.bandit.n)
           ]
        else:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x])) + ((beta.std(
           self._as[x], self._bs[x]) * self.c)/ np.sqrt(1 + self.count[x]))
           for x in range(self.bandit.n)
           ]
           
                
    
    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        
        if self._r == 1:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x])) + ((beta.std(
           self._as[x], self._bs[x]) * self.c)/ np.sqrt(1 + self.counts[x])) 
           for x in range(self.bandit.n)
           ]
        else:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x])) + ((beta.std(
           self._as[x], self._bs[x]) * self.c)/ np.sqrt(1 + self.counts[x]))
           for x in range(self.bandit.n)
           ]
        
        i = max(
            range(self.bandit.n),
            key=lambda x: samples[x]
        )
        
        self._r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += self._r
        self._bs[i] += (1 - self._r)
        
        #print(samples)
        
           
        return i

class NewBernoulliBayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(NewBernoulliBayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        self._util = [0.03, 0.07, 0.15, 0.25, 0.5]
        self._r = 0

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i])  for i in range(self.bandit.n)]

    def reward_result(self):
        return self.bandit.generate_reward(i)
                
    
    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        
        if self._r == 1:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x])) + ((beta.std(
           self._as[x], self._bs[x]) * self.c)/ np.sqrt(1 + self.counts[x])) * self._util[x] 
           for x in range(self.bandit.n)
           ]
        else:
           samples = [(self._as[x] / float(self._as[x] + self._bs[x])) + ((beta.std(
           self._as[x], self._bs[x]) * self.c)/ np.sqrt(1 + self.counts[x])) 
           for x in range(self.bandit.n)
           ]
        
        i = max(
            range(self.bandit.n),
            key=lambda x: samples[x]
        )
        
        self._r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += self._r
        self._bs[i] += (1 - self._r)
        
        #print(samples)
        
           
        return i


class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a=0, init_b=0):
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

        print(samples)
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
