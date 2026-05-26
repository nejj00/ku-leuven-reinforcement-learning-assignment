from collections import defaultdict

import numpy as np

from task2.tabular_agent import TabularMinihackAgent

class MCAgent(TabularMinihackAgent):
    def __init__(
        self,
        id,
        action_space,
        epsilon=0.1,
        alpha=0.1,
        gamma=1.0,
        initialization_value=0.0,
        epsilon_schedule=False,
        epsilon_end=0.0,
        epsilon_decay_episodes=1,
    ):
        super().__init__(
            id,
            action_space,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            initialization_value=initialization_value,
            epsilon_schedule=epsilon_schedule,
            epsilon_end=epsilon_end,
            epsilon_decay_episodes=epsilon_decay_episodes,
        )
        self.counts = defaultdict(lambda: np.ones(self.action_space.n))
        self.last_episode = []

    def act(self, state, reward=0, step=0):
        # TODO: Implement act() for the Monte Carlo agent.

        # ### YOUR SOLUTION STARTS HERE
        if self.last_action is not None and self.last_state is not None:
            self.last_episode.append((self.last_state, self.last_action, reward))

        encoded_state = self.state_encoder(state)
        self.last_state = encoded_state
        action = self.choose_next_action(encoded_state)
        self.last_action = action

        return action
        # ### END OF YOUR SOLUTION

    def onEpisodeEnd(self, state, action, reward, episode):
        # TODO: Implement onEpisodeEnd() for the Monte Carlo agent.

        # ### YOUR SOLUTION STARTS HERE
        self.last_episode.append((self.last_state, self.last_action, reward))
        self.learn()
        self.last_episode.clear()
        self.last_state = None
        self.last_action = None
        # ### END OF YOUR SOLUTION

    def learn(self):
        # TODO: Implement the Monte Carlo return and Q-table update.

        # ### YOUR SOLUTION STARTS HERE
        G = 0
        for state, action, reward in reversed(self.last_episode):
            G = self.gamma * G + reward
            self.counts[state][action] += 1

            self.q_table[state][action] += (G - self.q_table[state][action]) / self.counts[state][action]
        # ### END OF YOUR SOLUTION
