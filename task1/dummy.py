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
        self.agent_pos = (0, 0)
        self.goal_pos = (self.n - 1, self.m - 1)
        # ### END OF YOUR SOLUTION

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment to the initial state.

        Return:
            observation, info
        """
        # TODO: Implement reset.
        # ### YOUR SOLUTION STARTS HERE
        super().reset(seed=seed)
        
        self.agent_pos = (0, 0)
        
        observation = self.agent_pos
        info = {}
        
        return observation, info
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
        direction = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, 1),   # right
            3: (0, -1)    # left
        }
        
        move = direction.get(action, (0, 0))
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if 0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.m:
            self.agent_pos = new_pos
        
        observation = self.agent_pos        
        reward = -1
        terminated = self.agent_pos == self.goal_pos
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info
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
        grid = ""
        
        for i in range(self.n):
            for j in range(self.m):
                if (i, j) == self.agent_pos:
                    grid += "A"
                elif (i, j) == self.goal_pos:
                    grid += "G"
                else:
                    grid += "."
            grid += "\n"
            
        return grid
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
        self.agent_pos = (0, 0)
        self.goal_pos = (self.n - 1, self.m - 1)
        self.np_random = np.random.RandomState()
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
        if options and "start_pos" in options:
            self.agent_pos = options["start_pos"]
        else:
            self.agent_pos = (0, 0)

        self.np_random = np.random.RandomState(seed)
                    
        observation = self.agent_pos
        info = {}
        
        return observation, info
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
        direction = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, 1),   # right
            3: (0, -1)    # left
        }
        
        left_turn = {
            0: 3,  # up -> left
            1: 2,  # down -> right
            2: 0,  # right -> up
            3: 1   # left -> down
        }
        
        right_turn = {
            0: 2,  # up -> right
            1: 3,  # down -> left
            2: 1,  # right -> down
            3: 0   # left -> up
        }
        
        executed_action = self.np_random.choice([action, left_turn[action], right_turn[action]], p=[self.p_success, (1 - self.p_success) / 2, (1 - self.p_success) / 2])
        move = direction.get(executed_action, (0, 0))
        
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if 0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.m:
            self.agent_pos = new_pos
        
        observation = self.agent_pos        
        reward = -1
        terminated = self.agent_pos == self.goal_pos
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info
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
        grid = ""
        
        for i in range(self.n):
            for j in range(self.m):
                if (i, j) == self.agent_pos:
                    grid += "A"
                elif (i, j) == self.goal_pos:
                    grid += "G"
                else:
                    grid += "."
            grid += "\n"
            
        return grid
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
        row, col = state
        goal_row, goal_col = self.goal_pos
        if row < goal_row:
            return 1
        elif col < goal_col:
            return 2
        else:
            return 0        
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
        self.agent_pos = (0, 0)
        # ### END OF YOUR SOLUTION

    def act(self, state, reward=0, **kwargs):
        # TODO: Implement a policy that navigates to the goal.
        # ### YOUR SOLUTION STARTS HERE
        chars = state["chars"]
        self.agent_pos = np.argwhere(chars == ord('@'))[0]
        
        if chars[self.agent_pos[0]][self.agent_pos[1] + 1] != ord('|'):
            return me.ACTIONS.index(nethack.CompassCardinalDirection.E)
        elif chars[self.agent_pos[0] + 1][self.agent_pos[1]] != ord('-'):
            return me.ACTIONS.index(nethack.CompassCardinalDirection.S)
        else:
            return me.ACTIONS.index(nethack.CompassCardinalDirection.N)
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
        self.agent_pos = (0, 0)
        # ### END OF YOUR SOLUTION

    def act(self, state, reward=0, **kwargs):
        # TODO: Implement a policy that navigates to the goal while avoiding hazards.
        # ### YOUR SOLUTION STARTS HERE
        chars = state["chars"]
        print(f"State chars: {chars}")
        self.agent_pos = np.argwhere(chars == ord('@'))[0]
        
        if chars[self.agent_pos[0]][self.agent_pos[1] + 1] == ord('L'):
            return me.ACTIONS.index(nethack.CompassCardinalDirection.N)
        elif chars[self.agent_pos[0]][self.agent_pos[1] + 1] != ord('|'):
            return me.ACTIONS.index(nethack.CompassCardinalDirection.E)
        elif chars[self.agent_pos[0] + 1][self.agent_pos[1]] != ord('-'):
            return me.ACTIONS.index(nethack.CompassCardinalDirection.S)
        else:
            return me.ACTIONS.index(nethack.CompassCardinalDirection.N)
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
        episode_returns = []
        for episode in range(n_episodes):
            state, info = self.env.reset()
            episode_return = 0
            step = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                action = self.agent.act(state, reward=episode_return, step=step)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                state = next_state
                step += 1
                
                if max_steps_per_episode is not None and step >= max_steps_per_episode:
                    truncated = True
            
            self.agent.onEpisodeEnd(state, action, episode_return, episode)
            episode_returns.append(episode_return)
        
        return episode_returns
        # ### END OF YOUR SOLUTION


if __name__ == "__main__":
    pass