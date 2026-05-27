import numpy as np
import random
from collections import defaultdict

from commons import AbstractAgent

blank = 32

class TabularMinihackAgent(AbstractAgent):

    def __init__(
        self,
        id,
        action_space,
        alpha=0.1,
        gamma=1.,
        epsilon=0.1,
        initialization_value=0.,
        epsilon_schedule=False,
        epsilon_end=0.0,
        epsilon_decay_episodes=1,
    ):
        super().__init__(id, action_space)
        self._state_map = {}
        self.q_table = defaultdict(lambda: initialization_value * np.ones(self.action_space.n))
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = max(int(epsilon_decay_episodes), 1)
        self.alpha = alpha
        self.gamma = gamma
        self.last_state = None
        self.last_action = None

    def checkpoint_data(self):
        return {
            "state_map": dict(self._state_map),
            "q_table": {state: values.copy() for state, values in self.q_table.items()},
            "initialization_value": float(self.q_table.default_factory()[0]),
        }

    def load_checkpoint_data(self, checkpoint):
        initialization_value = checkpoint.get("initialization_value", 0.0)
        self._state_map = dict(checkpoint["state_map"])
        self.q_table = defaultdict(
            lambda: initialization_value * np.ones(self.action_space.n),
            {
                int(state): np.array(values, dtype=float)
                for state, values in checkpoint["q_table"].items()
            },
        )

    def state_encoder(self, observation):
        c = observation["chars"]
        coords = np.argwhere(c != blank)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        c = c[x_min:x_max + 1, y_min:y_max + 1]
        c = np.reshape(c, [-1])
        key = tuple(c)
        if key not in self._state_map:
            self._state_map[key] = len(self._state_map)
        return self._state_map[key]
    
    def choose_next_action(self, state):
        # TODO: Implement epsilon-greedy action selection.

        # ### YOUR SOLUTION STARTS HERE
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))
        # ### END OF YOUR SOLUTION

    def update_epsilon(self, episode):
        if not self.epsilon_schedule:
            return
        # TODO: Implement the linear epsilon decay schedule.

        # ### YOUR SOLUTION STARTS HERE
        if  episode < self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_start - ((episode + 1) * ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay_episodes))
        else:
            self.epsilon = self.epsilon_end
        # ### END OF YOUR SOLUTION
