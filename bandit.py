import random

class KArmedBandit:
    def __init__(self, k, Q1, epsilon, true_rewards=[]):
        self.N, self.Q, self.q = {}, {}, {}
        if len(true_rewards) == 0:
            true_rewards = [random.gauss(0, 1) for arm in k]
        for arm in range(0, k):
           self.N[arm] = 0
           self.Q[arm] = [Q1]
           self.q[arm] = true_rewards[arm]
        self.epsilon = epsilon

    def run(self):
        if random.random() < self.epsilon:
            action = 
        return 
