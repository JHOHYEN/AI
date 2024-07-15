import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.epsilon = 0.1
        self.gamma = 0.99
        self.alpha = 0.01
        
        if self.mode == "q_learning":
            self.alpha = 0.2
    
        elif self.mode == "mc_control":
            self.alpha = 0.001
            self.step_result = list()

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space        
        """
        if self.mode == "q_learning" or self.mode == 'test_mode' or self.mode == "mc_control":       
            self.epsilon = self.epsilon / 10
            policy_state = np.ones(self.n_actions) * self.epsilon / self.n_actions
            action_greedy = np.argmax(self.Q[state])
            policy_state[action_greedy] = 1 - self.epsilon + (self.epsilon / self.n_actions)
            action = np.random.choice(self.n_actions, p=policy_state)
            return action
        #return np.random.choice(self.n_actions)
        #return action
        
    def step(self, state, action, reward, next_state, done):
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.mode == "mc_control":
            if done:
                rewards = defaultdict(lambda: np.zeros(self.n_actions))
                for q_result in reversed(self.step_result):
                    state, action, reward = q_result
                    rewards[state][action] = reward + self.gamma * rewards[state][action]
                    self.Q[state][action] += self.alpha * (rewards[state][action] - self.Q[state][action])
                self.step_result = []
            else:
                self.step_result.append((state, action, reward))
                
        elif self.mode == "q_learning":
            Q_value = reward + self.gamma * self.Q[next_state][action]
            self.Q[state][action] += self.alpha * (Q_value - self.Q[state][action])
        
