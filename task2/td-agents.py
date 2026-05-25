import numpy as np
import random

from task2.tabular_agent import TabularMinihackAgent

class TDAgent(TabularMinihackAgent):

    def learn(self, state, action, next_state, next_action, reward):
        raise NotImplementedError()

    def act(self, state, reward=0, step=0):
        # TODO: Implement act() for TD agents.

        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def onEpisodeEnd(self,*args, **kwargs):
        # TODO: Implement onEpisodeEnd() for TD agents.

        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

class QLearning(TDAgent):
    def learn(self, state, action, next_state, next_action, reward):
        # TODO: Implement the Q-learning update rule.

        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def choose_next_action(self, state):
        # TODO: Implement epsilon-greedy action selection, disabling exploration when not learning.

        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

class SARSAOnPolicyAgent(TDAgent):
    def learn(self, state, action, next_state, next_action, reward):
        # TODO: Implement the SARSA update rule.

        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION
    
