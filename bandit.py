import random

def argmax(d):
    # returns a random key in the dict among those that have the max value
    max_val = max(list(d.values())) 
    maxs = [key for (key, val) in d.items() if val == max_val]
    return random.choice(maxs) 

class KArmedBandit:
    def __init__(self, k, Q1, epsilon, step_size=None, true_rewards=[]):
        self.N, self.Q, self.q = {}, {}, {}
        self.k = k
        self.Q1 = Q1
        self.rewards = []
        self.step_size = step_size
        self.epsilon = epsilon
        for arm in range(0, self.k):
           self.N[arm] = 0
           self.Q[arm] = [self.Q1]
           self.q[arm] = true_rewards[arm] if len(true_rewards) > 0 else random.gauss(0, 1)

    def run(self):
        if random.random() < self.epsilon:
            action = random.randint(0, self.k-1)
        else:
            action = argmax({key: vals[-1] for (key, vals) in self.Q.items()})
        self.N[action] += 1
        self.rewards.append(random.gauss(self.q[action], 1))
        self.update_Q(action)
        return self.rewards[-1]

    def update_Q(self, action):
        if not self.step_size:
            self.Q[action].append(self.Q[action][-1] + ((self.rewards[-1] - self.Q[action][-1]) / self.N[action]))
        else:
            self.Q[action].append(self.Q[action][-1] + self.step_size * (self.rewards[-1] - self.Q[action][-1]))

    def __str__(self):
        string = f"{self.k}-armed bandit with Q1={self.Q1}, ε={self.epsilon}"
        for i in range(0, self.k):
            string += f"\nArm {i} --> N={self.N[i]}, Q={self.Q[i][-1]}, q={self.q[i]}"
        string += f"\nTotal reward = {sum(self.rewards)}"
        return string 
