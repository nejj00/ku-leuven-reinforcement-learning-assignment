# tests/test1.py

##################
## INSTRUCTIONS ##
##################

# Run from the repository root with:
#   docker compose run --rm py pytest tests/test1.py
#
# To run a single test:
#   docker compose run --rm py pytest tests/test1.py::test_grid_world_spaces_reset_and_render
#
# To run with more detailed output:
#   docker compose run --rm py pytest -vv tests/test1.py
#
# To run all tests in the repository:
#   docker compose run --rm py pytest tests/

######################
## END INSTRUCTIONS ##
######################

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

###################
## CONFIGURATION ##
###################

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from task1.dummy import (
    GridWorldEnv,
    SlipperyGridWorldEnv,
    GridWorldHardCodedAgent,
    MiniHackCliffHardCodedAgent,
    MiniHackEmptyHardCodedAgent,
    AbstractRLTask,
)
import minihack_env as me
from nle import nethack

#######################
## END CONFIGURATION ##
#######################


###################
## TEST UTILITIES ##
###################

def _as_pos(obs):
    arr = np.asarray(obs, dtype=int).reshape(-1)
    assert arr.size == 2
    return tuple(arr.tolist())


def _unwrap_reset(reset_out):
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _unwrap_step(step_out):
    if len(step_out) == 5:
        return step_out
    obs, reward, terminated, info = step_out
    return obs, reward, terminated, False, info


def _find_agent_pos(observation):
    chars = observation["chars"]
    coords = np.argwhere(chars == ord("@"))
    if coords.size == 0:
        return None
    return tuple(coords[0])


BLOCKED = {ord("|"), ord("-"), ord("#"), ord(" ")}

CARDINAL_ACTION_TO_DELTA = {
    me.ACTIONS.index(nethack.CompassCardinalDirection.N): (-1, 0),
    me.ACTIONS.index(nethack.CompassCardinalDirection.S): (1, 0),
    me.ACTIONS.index(nethack.CompassCardinalDirection.E): (0, 1),
    me.ACTIONS.index(nethack.CompassCardinalDirection.W): (0, -1),
}


def _assert_safe_cardinal_move(obs, action, action_space):
    assert action_space.contains(action)
    assert action in CARDINAL_ACTION_TO_DELTA

    pos = _find_agent_pos(obs)
    assert pos is not None

    dr, dc = CARDINAL_ACTION_TO_DELTA[action]
    nr, nc = pos[0] + dr, pos[1] + dc
    chars = obs["chars"]

    assert 0 <= nr < chars.shape[0]
    assert 0 <= nc < chars.shape[1]
    assert chars[nr, nc] not in BLOCKED


########################
## TESTS FOR TASK 1.1 ##
########################

@pytest.mark.parametrize("n,m", [(2, 2), (3, 4), (5, 5)])
def test_grid_world_spaces_reset_and_render(n, m):
    env = GridWorldEnv(n=n, m=m)

    assert isinstance(env, gym.Env)

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4

    assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
    np.testing.assert_array_equal(env.observation_space.nvec, np.array([n, m]))

    obs, info = env.reset(seed=123)

    assert isinstance(info, dict)
    assert _as_pos(obs) == (0, 0)
    assert env.observation_space.contains(np.asarray(obs))

    rendered = env.render()
    assert isinstance(rendered, str)
    assert rendered.count("A") == 1
    assert rendered.count("G") == 1
    assert len(rendered.splitlines()) == n


@pytest.mark.parametrize("n,m", [(2, 2), (3, 4), (5, 5)])
def test_grid_world_public_dynamics_to_goal(n, m):
    env = GridWorldEnv(n=n, m=m)
    goal = (n - 1, m - 1)

    obs, info = env.reset()
    assert isinstance(info, dict)
    assert _as_pos(obs) == (0, 0)

    for action in (0, 3):
        obs, reward, terminated, truncated, info = env.step(action)
        assert _as_pos(obs) == (0, 0)
        assert reward == -1
        assert terminated is False
        assert truncated is False
        assert isinstance(info, dict)
        assert env.observation_space.contains(np.asarray(obs))

    actions = [1] * (n - 1) + [2] * (m - 1)
    for i, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == -1
        assert truncated is False
        assert isinstance(info, dict)
        assert env.observation_space.contains(np.asarray(obs))
        assert terminated is (i == len(actions))

    assert _as_pos(obs) == goal


def test_grid_world_action_mapping_from_interior():
    env = GridWorldEnv(n=3, m=4)

    expected_after_action = {
        0: (0, 1),
        1: (2, 1),
        2: (1, 2),
        3: (1, 0),
    }

    for action, expected_pos in expected_after_action.items():
        env.reset()
        env.step(1)
        env.step(2)

        obs, reward, terminated, truncated, info = env.step(action)

        assert _as_pos(obs) == expected_pos
        assert reward == -1
        assert terminated is False
        assert truncated is False
        assert isinstance(info, dict)


############################
## END TESTS FOR TASK 1.1 ##
############################


########################
## TESTS FOR TASK 1.2 ##
########################

@pytest.mark.parametrize("n,m,p_success", [(3, 4, 1.0), (4, 4, 0.8)])
def test_slippery_grid_world_spaces_reset_and_render(n, m, p_success):
    env = SlipperyGridWorldEnv(n=n, m=m, p_success=p_success)

    assert isinstance(env, gym.Env)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4

    assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
    np.testing.assert_array_equal(env.observation_space.nvec, np.array([n, m]))

    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)
    assert _as_pos(obs) == (0, 0)
    assert env.observation_space.contains(np.asarray(obs))

    rendered = env.render()
    assert isinstance(rendered, str)
    assert rendered.count("A") == 1
    assert rendered.count("G") == 1
    assert len(rendered.splitlines()) == n


def test_slippery_grid_world_reset_start_pos_option():
    env = SlipperyGridWorldEnv(n=4, m=5, p_success=0.8)

    obs, info = env.reset(seed=7, options={"start_pos": (2, 3)})
    assert isinstance(info, dict)
    assert _as_pos(obs) == (2, 3)
    assert env.observation_space.contains(np.asarray(obs))


def test_slippery_grid_world_p1_matches_deterministic_dynamics():
    env = SlipperyGridWorldEnv(n=3, m=4, p_success=1.0)

    obs, info = env.reset(seed=0)
    assert _as_pos(obs) == (0, 0)

    obs, reward, terminated, truncated, info = env.step(1)
    assert _as_pos(obs) == (1, 0)
    assert reward == -1
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(2)
    assert _as_pos(obs) == (1, 1)
    assert reward == -1
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(0)
    assert _as_pos(obs) == (0, 1)
    assert reward == -1
    assert terminated is False
    assert truncated is False

    obs, reward, terminated, truncated, info = env.step(3)
    assert _as_pos(obs) == (0, 0)
    assert reward == -1
    assert terminated is False
    assert truncated is False


def test_slippery_grid_world_p1_reaches_goal_like_grid_world():
    env = SlipperyGridWorldEnv(n=3, m=4, p_success=1.0)
    goal = (2, 3)

    obs, _ = env.reset(seed=0)

    actions = [1, 1, 2, 2, 2]
    for i, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == -1
        assert truncated is False
        assert isinstance(info, dict)
        assert terminated is (i == len(actions))

    assert _as_pos(obs) == goal


def _rollout_slippery(env, actions, seed, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    out = [_as_pos(obs)]
    for a in actions:
        obs, reward, terminated, truncated, _ = env.step(a)
        out.append((_as_pos(obs), reward, terminated, truncated))
        if terminated or truncated:
            break
    return out


def test_slippery_grid_world_same_seed_same_trajectory():
    actions = [1, 2, 2, 0, 3, 1, 2, 2]

    env1 = SlipperyGridWorldEnv(n=5, m=5, p_success=0.75)
    env2 = SlipperyGridWorldEnv(n=5, m=5, p_success=0.75)

    traj1 = _rollout_slippery(env1, actions, seed=42)
    traj2 = _rollout_slippery(env2, actions, seed=42)

    assert traj1 == traj2


def test_slippery_grid_world_empirical_distribution_from_interior():
    env = SlipperyGridWorldEnv(n=4, m=4, p_success=0.8)

    counts = {
        (0, 1): 0,
        (1, 0): 0,
        (1, 2): 0,
    }

    n_trials = 4000
    for seed in range(n_trials):
        obs, _ = env.reset(seed=seed, options={"start_pos": (1, 1)})
        obs, reward, terminated, truncated, info = env.step(0)
        pos = _as_pos(obs)

        assert reward == -1
        assert truncated is False
        assert isinstance(info, dict)
        assert pos in counts
        counts[pos] += 1

    p_up = counts[(0, 1)] / n_trials
    p_left = counts[(1, 0)] / n_trials
    p_right = counts[(1, 2)] / n_trials

    assert abs(p_up - 0.8) < 0.05
    assert abs(p_left - 0.1) < 0.04
    assert abs(p_right - 0.1) < 0.04


def test_slippery_grid_world_observation_always_in_space():
    env = SlipperyGridWorldEnv(n=4, m=4, p_success=0.65)

    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(np.asarray(obs))

    for action in [0, 1, 2, 3] * 10:
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(np.asarray(obs))
        assert reward == -1
        assert isinstance(info, dict)
        if terminated or truncated:
            break


############################
## END TESTS FOR TASK 1.2 ##
############################


########################
## TESTS FOR TASK 1.3 ##
########################

@pytest.mark.parametrize("n,m", [(2, 2), (3, 4), (5, 5)])
def test_grid_world_hardcoded_agent_policy_and_rollout(n, m):
    env = GridWorldEnv(n=n, m=m)
    goal = (n - 1, m - 1)

    agent = GridWorldHardCodedAgent(
        id="agent-1",
        action_space=env.action_space,
        goal_pos=goal,
    )

    assert agent.act((0, 0), reward=0) == 1
    assert agent.act((n - 1, 0), reward=0) == 2
    assert env.action_space.contains(agent.act(goal, reward=0))

    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    optimal_steps = (n - 1) + (m - 1)

    while not (terminated or truncated):
        row, col = _as_pos(obs)
        action = agent.act(obs, reward=0)
        assert env.action_space.contains(action)

        if row < goal[0]:
            assert action == 1
        elif col < goal[1]:
            assert action == 2

        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == -1
        assert isinstance(info, dict)
        steps += 1

        assert steps <= optimal_steps + 1

    assert _as_pos(obs) == goal
    assert steps == optimal_steps


############################
## END TESTS FOR TASK 1.3 ##
############################


########################
## TESTS FOR TASK 1.4 ##
########################

def test_minihack_empty_hardcoded_agent_finishes_safely():
    env = me.get_minihack_environment(me.EMPTY_ROOM)
    obs = _unwrap_reset(env.reset())

    agent = MiniHackEmptyHardCodedAgent(
        id="empty-agent",
        action_space=env.action_space,
        goal_pos=None,
    )

    reward = 0
    terminated = False
    truncated = False
    prev_pos = _find_agent_pos(obs)

    max_steps = obs["chars"].shape[0] * obs["chars"].shape[1]

    for _ in range(max_steps):
        action = agent.act(obs, reward=reward)
        _assert_safe_cardinal_move(obs, action, env.action_space)

        obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))
        assert isinstance(info, dict)

        new_pos = _find_agent_pos(obs)
        if new_pos is not None and prev_pos is not None:
            assert new_pos != prev_pos
            prev_pos = new_pos

        if terminated or truncated:
            break

    assert terminated or truncated, "Agent did not finish the episode within budget."


############################
## END TESTS FOR TASK 1.4 ##
############################


########################
## TESTS FOR TASK 1.5 ##
########################

def test_minihack_cliff_hardcoded_agent_finishes_safely():
    env = me.get_minihack_environment(me.CLIFF)
    obs = _unwrap_reset(env.reset())

    agent = MiniHackCliffHardCodedAgent(
        id="cliff-agent",
        action_space=env.action_space,
        goal_pos=None,
    )

    reward = 0
    terminated = False
    truncated = False
    prev_pos = _find_agent_pos(obs)

    max_steps = 2 * obs["chars"].shape[0] * obs["chars"].shape[1]

    for _ in range(max_steps):
        action = agent.act(obs, reward=reward)
        _assert_safe_cardinal_move(obs, action, env.action_space)

        obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))
        assert isinstance(info, dict)

        new_pos = _find_agent_pos(obs)
        if new_pos is not None and prev_pos is not None:
            assert new_pos != prev_pos
            prev_pos = new_pos

        if terminated or truncated:
            break

    assert terminated or truncated, "Agent did not finish the episode within budget."


############################
## END TESTS FOR TASK 1.5 ##
############################


########################
## TESTS FOR TASK 1.6 ##
########################

class _FixedLengthEnv(gym.Env):
    def __init__(self, horizon=3):
        super().__init__()
        self.horizon = horizon
        self.t = 0
        self.reset_calls = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(horizon + 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.reset_calls += 1
        return 0, {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self.horizon
        return self.t, 1.0, terminated, False, {}


class _RecordingAgent:
    def __init__(self):
        self.episode_ends = []

    def act(self, state, reward=0, **kwargs):
        return 0

    def onEpisodeEnd(self, state, action, reward, episode):
        self.episode_ends.append((state, action, reward, episode))


def test_abstract_rl_task_raw_returns_and_multiple_episodes():
    env = _FixedLengthEnv(horizon=3)
    agent = _RecordingAgent()
    task = AbstractRLTask(env, agent)

    returns = task.interact(n_episodes=3)
    expected_episode_return = 3.0  # 3 steps × reward 1.0, no discounting

    assert len(returns) == 3
    assert np.allclose(returns, [expected_episode_return] * 3)
    assert env.reset_calls == 3
    assert len(agent.episode_ends) == 3


def test_abstract_rl_task_respects_max_steps_per_episode():
    env = _FixedLengthEnv(horizon=10)
    agent = _RecordingAgent()
    task = AbstractRLTask(env, agent)

    returns = task.interact(n_episodes=2, max_steps_per_episode=2)

    assert len(returns) == 2
    assert np.allclose(returns, [2.0, 2.0])
    assert env.reset_calls == 2
    assert len(agent.episode_ends) == 2


############################
## END TESTS FOR TASK 1.6 ##
############################