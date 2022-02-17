import matplotlib  # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np

import pdb

# Writing to an excel 
# sheet using Python
import xlwt
from xlwt import Workbook
import xlrd
from xlutils.copy import copy

from bandits import BernoulliBandit
from solvers import Solver, EpsilonGreedy, UCB1, UCB2, UCB3, NewUCB1, BayesianUCB, NewBayesianUCB, BernoulliBayesianUCB, NewBernoulliBayesianUCB, ThompsonSampling



def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit
    
    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])        
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)
    
    # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x] for x in sorted_indices], 'x', markeredgewidth=2)
        #ax2.plot(range(b.n), [s.estimated_ucbs[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'$\theta$' + '(true reward probability)')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls=':', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)
    
    
    plt.savefig(figname)
    
def write_to_excel(row, column, data):
    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet1')
        
    sheet1.write(row, column, data)
      
      
    wb.save('data.xls')
 
def single_run(q,p, t_s, n): 

    # Workbook is created
    #wb = Workbook()
    # add_sheet is used to create sheet.
    #sheet1 = wb.add_sheet('Sheet1')
    
        # load the excel file
    rb = xlrd.open_workbook('data2.xls')
  
           # copy the contents of excel file
    wb__ = copy(rb)
  
          # open the first sheet
    w__sheet = wb__.get_sheet(0)
    
    w__sheet.write(0, 0, 'COUNT 1')
    w__sheet.write(0, 2, 'COUNT 2')
    w__sheet.write(0, 4, 'COUNT 3')
    w__sheet.write(0, 6, 'COUNT 4')
    w__sheet.write(0, 8, 'COUNT 5')
        
    w__sheet.write(0, 11, 'UTIL')
    w__sheet.write(0, 13, 'REGRET')
    w__sheet.write(0, 15, 'ESTIMATES 1')
    w__sheet.write(0, 17, 'ESTIMATES 2')
    w__sheet.write(0, 19, 'ESTIMATES 3')
    w__sheet.write(0, 21, 'ESTIMATES 4')
    w__sheet.write(0, 23, 'ESTIMATES 5')
    
    w__sheet.write(0, 27, 'UTIL 1')
    w__sheet.write(0, 29, 'UTIL 2')
    w__sheet.write(0, 31, 'UTIL 3')
    w__sheet.write(0, 33, 'UTIL 4')
    w__sheet.write(0, 35, 'UTIL 5')
    
    w__sheet.write(0, 39, 'PROBA 1')
    w__sheet.write(0, 40, 'PROBA 2')
    w__sheet.write(0, 41, 'PROBA 3')
    w__sheet.write(0, 42, 'PROBA 4')
    w__sheet.write(0, 43, 'PROBA 5')
    
    w__sheet.write(0, 45, 'COST 1')
    w__sheet.write(0, 46, 'COST 2')
    w__sheet.write(0, 47, 'COST 3')
    w__sheet.write(0, 48, 'COST 4')
    w__sheet.write(0, 49, 'COST 5')
    
    w__sheet.write(0, 51, 'REWARD 1')
    w__sheet.write(0, 52, 'REWARD 2')
    w__sheet.write(0, 53, 'REWARD 3')
    w__sheet.write(0, 54, 'REWARD 4')
    w__sheet.write(0, 55, 'REWARD 5')
    
    w__sheet.write(0, 57, 'UTIL 1')
    w__sheet.write(0, 58, 'UTIL 2')
    w__sheet.write(0, 59, 'UTIL 3')
    w__sheet.write(0, 60, 'UTIL 4')
    w__sheet.write(0, 61, 'UTIL 5')
    
    
    #q = 0
    #p = 0
    t_solvers = 0
    for s in t_s:
        t_solvers += 1
      #for a in range(10000):
        #print(a)        
        s.run(n)
        if(t_solvers == 1):
          q += 1
        #print(s.counts)
          w__sheet.write(q, 0, s.arm_history[0][0] + s.arm_history[0][1])
          w__sheet.write(q, 2, s.arm_history[1][0] + s.arm_history[1][1])
          w__sheet.write(q, 4, s.arm_history[2][0] + s.arm_history[2][1])
          w__sheet.write(q, 6, s.arm_history[3][0] + s.arm_history[3][1])
          w__sheet.write(q, 8, s.arm_history[4][0] + s.arm_history[4][1])
        
          w__sheet.write(q, 11, s.utili)
          w__sheet.write(q, 13, s.regret)
          
          w__sheet.write(q, 16, s.estimates[0])
          w__sheet.write(q, 18, s.estimates[1])
          w__sheet.write(q, 20, s.estimates[2])
          w__sheet.write(q, 22, s.estimates[3])
          w__sheet.write(q, 24, s.estimates[4])
          
          w__sheet.write(q, 27, s.ucbs_util[0])
          w__sheet.write(q, 29, s.ucbs_util[1])
          w__sheet.write(q, 31, s.ucbs_util[2])
          w__sheet.write(q, 33, s.ucbs_util[3])
          w__sheet.write(q, 35, s.ucbs_util[4])
          
        elif(t_solvers == 2):
          p += 1
        #print(s.counts)
          w__sheet.write(p, 0 + 1, s.counts[0])
          w__sheet.write(p, 2 + 1, s.counts[1])
          w__sheet.write(p, 4 + 1, s.counts[2])
          w__sheet.write(p, 6 + 1, s.counts[3])
          w__sheet.write(p, 8 + 1, s.counts[4])
        
          w__sheet.write(p, 11 + 1, s.utili)
          w__sheet.write(p, 13 + 1, s.regret)
          
          w__sheet.write(p, 16 + 1, s.estimates[0])
          w__sheet.write(p, 18 + 1, s.estimates[1])
          w__sheet.write(p, 20 + 1, s.estimates[2])
          w__sheet.write(p, 22 + 1, s.estimates[3])
          w__sheet.write(p, 24 + 1, s.estimates[4])
          
          w__sheet.write(p, 27 + 1, s.ucbs_util[0])
          w__sheet.write(p, 29 + 1, s.ucbs_util[1])
          w__sheet.write(p, 31 + 1, s.ucbs_util[2])
          w__sheet.write(p, 33 + 1, s.ucbs_util[3])
          w__sheet.write(p, 35 + 1, s.ucbs_util[4])
          
        
    wb__.save('data2.xls')  

def save_exp(t_s, n):
    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet1')
    
    sheet1.write(0, 0, 'COUNT 1')
    sheet1.write(0, 4, 'COUNT 2')
    sheet1.write(0, 8, 'COUNT 3')
    sheet1.write(0, 12, 'COUNT 4')
    sheet1.write(0, 16, 'COUNT 5')
        
    sheet1.write(0, 22, 'UTIL')
    sheet1.write(0, 24, 'REGRET')
    
    sheet1.write(0, 27, 'ESTIMATES 1')
    sheet1.write(0, 29, 'ESTIMATES 2')
    sheet1.write(0, 31, 'ESTIMATES 3')
    sheet1.write(0, 33, 'ESTIMATES 4')
    sheet1.write(0, 35, 'ESTIMATES 5')
    
    
    q = 0
    p = 0
    t_solvers = 0
    for s in t_s:
      t_solvers += 1
      print(t_solvers)
      for a in range(1000):
        #print(a)        
        s.run(n)
        if(t_solvers == 1):
          q += 1
          
          sheet1.write(q, 0, s.arm_history[0][0])
          sheet1.write(q, 1, s.arm_history[0][1])
          sheet1.write(q, 4, s.arm_history[1][0])
          sheet1.write(q, 5, s.arm_history[1][1])
          sheet1.write(q, 8, s.arm_history[2][0])
          sheet1.write(q, 9, s.arm_history[2][1])
          sheet1.write(q, 12, s.arm_history[3][0])
          sheet1.write(q, 13, s.arm_history[3][1])
          sheet1.write(q, 16, s.arm_history[4][0])
          sheet1.write(q, 17, s.arm_history[4][1])
          
        
          sheet1.write(q, 22, s.utili)
          sheet1.write(q, 24, s.regret)
          
          sheet1.write(q, 27, s.estimates[0])
          sheet1.write(q, 29, s.estimates[1])
          sheet1.write(q, 31, s.estimates[2])
          sheet1.write(q, 33, s.estimates[3])
          sheet1.write(q, 35, s.estimates[4])
          
          
        elif(t_solvers == 2):
          p += 1
          
          sheet1.write(p, 0 + 2, s.arm_history[0][0])
          sheet1.write(p, 1 + 2, s.arm_history[0][1])
          sheet1.write(p, 4 + 2, s.arm_history[1][0])
          sheet1.write(p, 5 + 2, s.arm_history[1][1])
          sheet1.write(p, 8 + 2, s.arm_history[2][0])
          sheet1.write(p, 9 + 2, s.arm_history[2][1])
          sheet1.write(p, 12 + 2, s.arm_history[3][0])
          sheet1.write(p, 13 + 2, s.arm_history[3][1])
          sheet1.write(p, 16 + 2, s.arm_history[4][0])
          sheet1.write(p, 17 + 2, s.arm_history[4][1])
          
        
          sheet1.write(p, 22 + 1, s.utili)
          sheet1.write(p, 24 + 1, s.regret)
          
          sheet1.write(p, 27 + 1, s.estimates[0])
          sheet1.write(p, 29 + 1, s.estimates[1])
          sheet1.write(p, 31 + 1, s.estimates[2])
          sheet1.write(p, 33 + 1, s.estimates[3])
          sheet1.write(p, 35 + 1, s.estimates[4])
       
        
    wb.save('data__0.xls')   
    
    
def experiment(q, p, K, N, F, aa):
    """ 
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of slot machiens.
        N (int): number of time steps to try.
    """

    b = BernoulliBandit(K, F, aa, N)
    print ("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    print()
    print ("The best machine has index: {} and proba: {}".format(
           max(range(K), key=lambda i: b.probas[i]), max(b.probas)))
           

    
    
    test_solvers = [
        # EpsilonGreedy(b, 0),
        # EpsilonGreedy(b, 1),
        #EpsilonGreedy(b, 0.1),        
        #UCB1(b),
        UCB2(b),
        UCB3(b),
        #NewUCB1(b),
        #BayesianUCB(b, 3, 1, 1),
        #NewBayesianUCB(b, 3, 1, 1),
        #BernoulliBayesianUCB(b, 3, 1, 1),
        #NewBernoulliBayesianUCB(b, 3, 1, 1),
        #ThompsonSampling(b, 1, 1)
        ]
    names = [
        # 'Full-exploitation',
        # 'Full-exploration',
        #r'$\epsilon$' + '-Greedy',
        #'UCB1',
        'UCB2',
        'UCB3',
        #'newUCB1',
        #'Bayesian UCB',
        #'NewBayesianUCB',
        #'BernoulliBayesianUCB',
        #'NewBernoulliBayesianUCB',
        #'Thompson Sampling'
        ]

    #Run experiment and write all results into excel file per single run
    single_run(q, p, test_solvers, N)
    
    #for s in test_solvers: 
    #   s.run(N)
    
    #Run the experiment and save the experiment into excel file for 
    #10,000 times
    #save_exp(test_solvers, N)
    
        # load the excel file
    rb = xlrd.open_workbook('data2.xls')
  
           # copy the contents of excel file
    wb__ = copy(rb)
  
          # open the first sheet
    w__sheet = wb__.get_sheet(0)
           
        #print(s.counts)
         
    w__sheet.write(p + 1, 39, b.probas[0])
    w__sheet.write(p + 1, 40, b.probas[1])
    w__sheet.write(p + 1, 41, b.probas[2])
    w__sheet.write(p + 1, 42, b.probas[3])
    w__sheet.write(p + 1, 43, b.probas[4])
    
    w__sheet.write(p + 1, 45, b.arms_cost[0])
    w__sheet.write(p + 1, 46, b.arms_cost[1])
    w__sheet.write(p + 1, 47, b.arms_cost[2])
    w__sheet.write(p + 1, 48, b.arms_cost[3])
    w__sheet.write(p + 1, 49, b.arms_cost[4])
    
    w__sheet.write(p + 1, 51, b.arms_reward[0])
    w__sheet.write(p + 1, 52, b.arms_reward[1])
    w__sheet.write(p + 1, 53, b.arms_reward[2])
    w__sheet.write(p + 1, 54, b.arms_reward[3])
    w__sheet.write(p + 1, 55, b.arms_reward[4])
    
    w__sheet.write(p + 1, 57, b.util[0])
    w__sheet.write(p + 1, 58, b.util[1])
    w__sheet.write(p + 1, 59, b.util[2])
    w__sheet.write(p + 1, 60, b.util[3])
    w__sheet.write(p + 1, 61, b.util[4])         

        
    wb__.save('data2.xls')

       
    print ("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    print()
    print ("The best machine has index: {} and proba: {}".format(
           max(range(K), key=lambda i: b.probas[i]), max(b.probas)))
    
    plot_results(test_solvers, names, "results_K{}_N{}.png".format(K, N))

if __name__ == '__main__':
  
  
    wb = Workbook()
    worksheet = wb.add_sheet('Sheet 1')
    wb.save('data2.xls')
    #wb.save('data__0.xls')
    for i in range(1000):
      
      experiment(i, i, 5, 10000, 'DAAAA',i)
    
