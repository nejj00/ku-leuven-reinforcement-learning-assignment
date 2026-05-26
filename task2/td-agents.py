import numpy as np
import random

from task2.tabular_agent import TabularMinihackAgent

class TDAgent(TabularMinihackAgent):

    def learn(self, state, action, next_state, next_action, reward):
        raise NotImplementedError()

    def act(self, state, reward=0, step=0):
        # TODO: Implement act() for TD agents.

        # ### YOUR SOLUTION STARTS HERE
        encoded_state = self.state_encoder(state)
        action = self.choose_next_action(encoded_state)
        
        if self.last_action is not None and self.last_state is not None:
            self.learn(self.last_state, self.last_action, encoded_state, action, reward)
        
        self.last_state = encoded_state
        self.last_action = action
        
        return action
        # ### END OF YOUR SOLUTION

    def onEpisodeEnd(self,*args, **kwargs):
        # TODO: Implement onEpisodeEnd() for TD agents.

        # ### YOUR SOLUTION STARTS HERE
        encoded_state = self.state_encoder(args[0])
        action = args[1]
        reward = args[2]
        print(f"State: {encoded_state}, Action: {action}, Reward: {reward}")
        
        self.learn(self.last_state, self.last_action, encoded_state, action, reward)
        
        self.last_state = None
        self.last_action = None
        # ### END OF YOUR SOLUTION

class QLearning(TDAgent):
    def learn(self, state, action, next_state, next_action, reward):
        # TODO: Implement the Q-learning update rule.

        # ### YOUR SOLUTION STARTS HERE
        self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])
        # ### END OF YOUR SOLUTION

    def choose_next_action(self, state):
        # TODO: Implement epsilon-greedy action selection, disabling exploration when not learning.

        # ### YOUR SOLUTION STARTS HERE
        if self.learning and random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))
        # ### END OF YOUR SOLUTION

class SARSAOnPolicyAgent(TDAgent):
    def learn(self, state, action, next_state, next_action, reward):
        # TODO: Implement the SARSA update rule.
        # ### YOUR SOLUTION STARTS HERE
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action])
        # ### END OF YOUR SOLUTION
    
