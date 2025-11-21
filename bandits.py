import random
import math

import matplotlib.pyplot as plt

def argmax(d):
    # returns a random key in the dict among those that have the max value
    max_val = max(list(d.values())) 
    maxs = [key for (key, val) in d.items() if val == max_val]
    return random.choice(maxs) 

def k_armed_bandits(k, steps, epsilon, quiet=True):
    # returns the average reward recieved over steps and the percentage
    # of the time that the optimal action was chosen
    reward_sum = 0
    optimal_action_sum = 0
    q = {} # true action rewards 
    Q = {} # current estimates of action rewards
    N = {} # number of times each action is chosen
    R = [] # stores all previous rewards
    for i in range(0, k):
        Q[i] = [0]
        N[i] = 0
        q[i] = random.gauss(0, 1) # random reward in normal dist with mean 0, stdev=1
    optimal_action = argmax(q)
    
    for i in range(0, steps):
        if random.random() < epsilon: # epsilon% of the time, choose a random action
            A = random.choice(list(q.keys())) 
        else: # otherwise choose one of the best actions so far
            A = argmax(Q)

        optimal_action_sum += (A == optimal_action)
        R.append(random.gauss(q[A], 1))
        reward_sum += R[i] 
        N[A] += 1

        # Q[A] is updated using sample averages
        # Q[A].append(Q[A][len(Q[A])-2] + ((R[i] - Q[A][len(Q[A])-2] / N[A])))
        a = 0.2
        r_sum = 0
        n = len(Q[A])
        for j in range(0, n):
            r_sum += a * math.pow((1 - a), (n-1)) * R[j]
        Q[A].append(math.pow((1 - a), n) * Q[A][0] + r_sum)
    
    if not quiet:
        for i in range(0, k):
            print(f"Bandit {i+1} --> Q: {Q[i]}, N: {N[i]}, q: {q[i]}")
    return sum(R) / steps, (optimal_action_sum / steps) * 100

def run_bandit_trials(trials, k, epsilons, min_step=100, max_step=1000, step_step=100, quiet=True):
    epsilon_reward = {epsilon: [] for epsilon in epsilons} 
    epsilon_optimal_count = {epsilon: [] for epsilon in epsilons}    
    step_range = [1] + [i for i in range(min_step, max_step+1, step_step)]

    for epsilon in epsilon_reward:
        for steps in step_range: 
            if not quiet: print(f"ε = {epsilon}, steps = {steps}")
            avg_reward = 0
            avg_optimal_percentage = 0
            for i in range(0, trials):
                reward, optimal_percentage = k_armed_bandits(k, steps, epsilon) 
                avg_reward += reward
                avg_optimal_percentage += optimal_percentage
            epsilon_reward[epsilon].append(avg_reward / trials)
            epsilon_optimal_count[epsilon].append(avg_optimal_percentage / trials)
    
    for epsilon in epsilon_reward:
        plt.plot(step_range, epsilon_reward[epsilon], label=f"ε = {epsilon}")
    plt.legend()
    plt.xlabel("Step sizes")
    plt.ylabel("Average reward recieved")
    plt.show()
    
    for epsilon in epsilon_optimal_count:
        plt.plot(step_range, epsilon_optimal_count[epsilon], label=f"ε = {epsilon}")
    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.xlabel("Step sizes")
    plt.ylabel("Percentage of the time the optimal action was chosen")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_bandit_trials(2000, 10, [0.0, 0.01, 0.1], quiet=False)
    #run_bandit_trials(1, 10, [0.1], quiet=False)

    
