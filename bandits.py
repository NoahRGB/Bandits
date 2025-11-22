import random
import math

import matplotlib.pyplot as plt

from bandit import Bandit

def argmax(d):
    # returns a random key in the dict among those that have the max value
    max_val = max(list(d.values())) 
    maxs = [key for (key, val) in d.items() if val == max_val]
    return random.choice(maxs) 

def plot(x, y_vals, y_labels=[], x_title="", y_title="", y_lims=None, x_lims=None):
    for i in range(0, len(y_vals)):
        plt.plot(x, y_vals[i], label=y_labels[i])
    plt.legend()
    if y_lims: plt.ylim(y_lims[0], y_lims[1])
    if x_lims: plt.xlim(x_lims[0], x_lims[1])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


def k_armed_bandits(k, steps, epsilon, quiet=True, Q1=0, a=None):
    # returns the average reward recieved over steps and the percentage
    # of the time that the optimal action was chosen
    q = {} # true action rewards 
    Q = {} # current estimates of action rewards
    N = {} # number of times each action is chosen
    R = [] # stores all previous rewards recieved
    for i in range(0, k):
        Q[i] = [Q1] 
        N[i] = 0
        q[i] = random.gauss(0, 1) # random reward in normal dist with mean 0, stdev=1
    optimal_action = argmax(q)
    for i in range(0, steps):
        if random.random() < epsilon: # epsilon% of the time, choose a random action
            A = random.choice(list(q.keys())) 
        else: # otherwise choose one of the best actions so far
            A = argmax({key: val[-1] for (key,val) in Q.items()})

        R.append(random.gauss(q[A], 1))
        N[A] += 1

        
        if not a: # Q[A] is updated using sample averages
            Q[A].append(Q[A][-1] + ((R[i] - Q[A][-1]) / N[A]))
        else:
            Q[A].append(Q[A][-1] + a * (R[i] - Q[A][-1]))
   
    if not quiet:
        for i in range(0, k):
            print(f"Bandit {i+1} --> Q: {Q[i]}, N: {N[i]}, q: {q[i]}")
    return sum(R) / steps, (sum([value for (key, value) in N.items() if key==optimal_action]) / steps) * 100

def run_bandit_trials(trials, k, epsilons, min_step=100, max_step=1000, step_step=100, quiet=True):
    epsilon_reward = {epsilon: [] for epsilon in epsilons} 
    epsilon_optimal_count = {epsilon: [] for epsilon in epsilons}    
    step_range = [1] + [i for i in range(min_step, max_step+1, step_step)]
    if min_step == max_step == 1: step_range = [1]

    for epsilon in epsilon_reward:
        for steps in step_range: 
            if not quiet: print(f"ε = {epsilon}, steps = {steps}")
            avg_reward = 0
            avg_optimal_percentage = 0
            for i in range(0, trials):
                reward, optimal_percentage = k_armed_bandits(k, steps, epsilon, quiet=True) 
                avg_reward += reward
                avg_optimal_percentage += optimal_percentage
            epsilon_reward[epsilon].append(avg_reward / trials)
            epsilon_optimal_count[epsilon].append(avg_optimal_percentage / trials)
    
    plot(step_range, list(epsilon_reward.values()), [f"ε=0.0", f"ε=0.01", f"ε=0.1"], "Step sizes", "Average reward recieved")
    plot(step_range, list(epsilon_optimal_count.values()), [f"ε=0.0", f"ε=0.01", f"ε=0.1"], "Step sizes", "Percentage of the time the optimal action was chosen", y_lims=[0, 100])


if __name__ == "__main__":
    #run_bandit_trials(2000, 10, [0.0, 0.01, 0.1], quiet=False)
    #run_bandit_trials(10000, 10, [0.1], quiet=False)
    #run_bandit_trials(1, 10, [0.1], quiet=False, min_step=1, max_step=5, step_step=1)

    b1 = Bandit(random.gauss(1, 0), 0)
    
