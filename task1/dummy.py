import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List
from nle import nethack
import minihack_env as me

from commons import AbstractAgent


# TASK 1.1: Implement a variable-size Grid-World environment.
class GridWorldEnv(gym.Env):
    """
    Variable-size (n x m) Grid-World implemented as a Gymnasium environment.

    Environment rules:
    - start at (0, 0)
    - goal at (n-1, m-1)
    - actions: 0=up, 1=down, 2=right, 3=left
    - reward: -1 per step, even if hitting a wall

    Gymnasium API requirements:
    - self.action_space is fixed to gymnasium.spaces.Discrete(4)
    - self.observation_space is fixed to gymnasium.spaces.MultiDiscrete([n, m])
    - reset() must return: (observation, info)
    - step(action) must return:
        (observation, reward, terminated, truncated, info)

    Observation:
    - the agent position as (row, col)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, n: int = 5, m: int = 5, render_mode: str | None = None):
        super().__init__()
        self.n = n
        self.m = m
        self.render_mode = render_mode

        # Fixed Gymnasium spaces.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.n, self.m])

        # TODO: Initialize the environment state.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment to the initial state.

        Return:
            observation, info
        """
        # TODO: Implement reset.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def step(self, action):
        """
        Execute one action.

        Rules:
        - Update the agent position based on the action.
        - Prevent movement off-grid (position unchanged).
        - Apply reward = -1.
        - Set terminated when the agent reaches the goal.
        - truncated should always be False in this environment.

        Return:
            observation, reward, terminated, truncated, info
        """
        # TODO: Implement the transition dynamics.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def render(self):
        """
        Return a string representation of the grid using:
        - "." for empty cells
        - "A" for the agent
        - "G" for the goal
        """
        # TODO: Implement render.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION


# TASK 1.2: Implement a stochastic variant of the GridWorld environment.
class SlipperyGridWorldEnv(gym.Env):
    """
    Stochastic variable-size (n x m) Grid-World implemented as a Gymnasium environment.

    Same layout and rewards as GridWorldEnv:
    - start at (0, 0)
    - goal at (n-1, m-1)
    - actions: 0=up, 1=down, 2=right, 3=left
    - reward: -1 per step, even if hitting a wall

    Transition model:
    - with probability p_success, execute the intended action
    - with probability (1 - p_success) / 2, execute the action obtained by
      turning left relative to the intended direction
    - with probability (1 - p_success) / 2, execute the action obtained by
      turning right relative to the intended direction

    Action encoding:
    - 0 = up
    - 1 = down
    - 2 = right
    - 3 = left

    Left/right turns are defined relative to the intended direction:
    - up    -> left, right
    - down  -> right, left
    - right -> up, down
    - left  -> down, up

    Gymnasium API requirements:
    - self.action_space is fixed to gymnasium.spaces.Discrete(4)
    - self.observation_space is fixed to gymnasium.spaces.MultiDiscrete([n, m])
    - use self.np_random for randomness
    - reset() must return: (observation, info)
    - step(action) must return:
        (observation, reward, terminated, truncated, info)

    Reset options:
    - If options contains {"start_pos": (row, col)}, reset the agent there.
    - Otherwise reset to the default start state (0, 0).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        n: int = 5,
        m: int = 5,
        p_success: float = 0.8,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.p_success = p_success
        self.render_mode = render_mode

        # Fixed Gymnasium spaces.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.n, self.m])

        # TODO: Initialize the environment state.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment to the initial state.

        If options contains {"start_pos": (row, col)}, use that as the initial
        position. Otherwise use the default start state (0, 0).

        Return:
            observation, info
        """
        # TODO: Implement reset.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def step(self, action):
        """
        Execute one action using the stochastic transition model.

        Rules:
        - Sample the executed action using self.np_random according to the
          transition model described in the class docstring.
        - Apply the executed action to the grid.
        - Prevent movement off-grid (position unchanged).
        - Apply reward = -1.
        - Set terminated when the agent reaches the goal.
        - truncated should always be False in this environment.

        Return:
            observation, reward, terminated, truncated, info
        """
        # TODO: Implement the stochastic transition dynamics.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def render(self):
        """
        Return a string representation of the grid using:
        - "." for empty cells
        - "A" for the agent
        - "G" for the goal
        """
        # TODO: Implement render.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION


# TASK 1.3: Implement a simple scripted agent for the GridWorld environment.
class GridWorldHardCodedAgent(AbstractAgent):
    """
    Hardcoded policy for the GridWorld environment.

    Requirements:
    - act(state, reward=0, **kwargs) must return an action compatible with
      the environment action space.
    - The policy must move down until the goal row is reached, then move
      right until the goal column is reached.

    Assumed action mapping:
    - 0=up, 1=down, 2=right, 3=left
    """

    def __init__(self, id, action_space, goal_pos):
        super().__init__(id=id, action_space=action_space)
        self.goal_pos = goal_pos

    def act(self, state, reward=0, **kwargs):
        # TODO: Implement the hardcoded policy.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

# TASK 1.4: Implement a scripted MiniHack agent for the empty room environment.
class MiniHackEmptyHardCodedAgent(AbstractAgent):
    """
    Fixed agent on empty room MiniHack environment.

    Requirements:
    - act(state, reward=0, **kwargs) must return a valid action from the
      MiniHack environment action space.
    - Use state["chars"] to inspect the map.
    - state["chars"] contains ASCII codes, not characters.
    """

    def __init__(self, id, action_space, goal_pos):
        super().__init__(id=id, action_space=action_space)
        self.goal_pos = goal_pos

        # TODO: Initialize any state needed by act().
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def act(self, state, reward=0, **kwargs):
        # TODO: Implement a policy that navigates to the goal.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION


# TASK 1.5: Implement a scripted MiniHack agent for the cliff environment.
class MiniHackCliffHardCodedAgent(AbstractAgent):
    """
    Fixed agent on the MiniHack cliff environment.

    Requirements:
    - act(state, reward=0, **kwargs) must return a valid action from the
      MiniHack environment action space.
    - Use state["chars"] to inspect the map.
    - state["chars"] contains ASCII codes, not characters.
    """

    def __init__(self, id, action_space, goal_pos):
        super().__init__(id=id, action_space=action_space)
        self.goal_pos = goal_pos

        # TODO: Initialize any state needed by act().
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION

    def act(self, state, reward=0, **kwargs):
        # TODO: Implement a policy that navigates to the goal while avoiding hazards.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION


# TASK 1.6: Implement the generic interaction loop between an agent and an environment.
class AbstractRLTask:
    def __init__(self, env, agent):
        """
        This class abstracts the concept of an agent interacting with an environment.

        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """
        self.env = env
        self.agent = agent

    def interact(self, n_episodes, max_steps_per_episode=None) -> List[float]:
        """
        Execute n_episodes of interaction between the agent and the environment.

        :param n_episodes: number of episodes
        :param max_steps_per_episode: optional episode step limit
        :return: a list of raw (undiscounted) per-episode returns, one per episode

        Requirements:
        - Call env.reset() at the beginning of each episode.
        - Repeatedly:
            action = agent.act(state, reward=reward, step=step)
            next_state, reward, terminated, truncated, info = env.step(action)
        - Accumulate the raw (undiscounted) episode return.
        - If max_steps_per_episode is not None, truncate the episode when the
          limit is reached.
        - Call agent.onEpisodeEnd(state, action, reward, episode) at the end of
          each episode.
        - Return the list of per-episode returns.
        """
        # TODO: Implement the interaction loop.
        # ### YOUR SOLUTION STARTS HERE
        raise NotImplementedError()
        # ### END OF YOUR SOLUTION


if __name__ == "__main__":
    pass