##################
## INSTRUCTIONS ##
##################

# Run from the repository root with:
#   docker compose run --rm py pytest tests/test2.py
#
# To run only correctness tests:
#   docker compose run --rm py pytest tests/test2.py -m correctness
#
# To run only performance tests:
#   docker compose run --rm py pytest tests/test2.py -m performance
#
# To run a single test:
#   docker compose run --rm py pytest tests/test2.py::test_q_learning_improves_on_empty_room
#
# To run with more detailed output, use the -vv flag:
#   docker compose run --rm py pytest -vv tests/test2.py
#
# To run all tests in the repository:
#   docker compose run --rm py pytest tests/


######################
## END INSTRUCTIONS ##
######################

import importlib.util
import os
import pickle
import random
import sys
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

###################
## CONFIGURATION ##
###################

# Ensure repo root is on sys.path so local modules are importable when running pytest.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import minihack_env as me

TRAINING_TEST_THRESHOLDS = {
    me.EMPTY_ROOM: {
        "q_learning": 1,
        "sarsa": 1,
        "monte_carlo": 1,
    },
    me.CLIFF: {
        "q_learning": 1,
        "sarsa": 1,
        "monte_carlo": 1,
    },
    me.ROOM_WITH_MONSTER: {
        "q_learning": 0.5,
        "sarsa": 0.5,
        "monte_carlo": 0.5,
    },
}

MAX_EPISODE_STEPS = 50
EVAL_SEED = 0

def load_module(module_name: str, relative_path: str):
    module_path = Path(REPO_ROOT) / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

td_agents = load_module("td_agents", "task2/td-agents.py")
mc_agents = load_module("mc_agents", "task2/mc-agents.py")

QLearning = td_agents.QLearning
SARSAOnPolicyAgent = td_agents.SARSAOnPolicyAgent
TDAgent = td_agents.TDAgent
MCAgent = mc_agents.MCAgent

PRETRAINED_TABULAR_EVAL_CONFIGS = {
    "q_learning": {
        me.EMPTY_ROOM: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "q_learning" / f"{me.EMPTY_ROOM}.pkl",
            "episodes": 1,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.EMPTY_ROOM]["q_learning"],
        },
        me.CLIFF: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "q_learning" / f"{me.CLIFF}.pkl",
            "episodes": 1,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.CLIFF]["q_learning"],
        },
        me.ROOM_WITH_MONSTER: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "q_learning" / f"{me.ROOM_WITH_MONSTER}.pkl",
            "episodes": 10,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.ROOM_WITH_MONSTER]["q_learning"],
        },
    },
    "sarsa": {
        me.EMPTY_ROOM: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "sarsa" / f"{me.EMPTY_ROOM}.pkl",
            "episodes": 1,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.EMPTY_ROOM]["sarsa"],
        },
        me.CLIFF: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "sarsa" / f"{me.CLIFF}.pkl",
            "episodes": 1,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.CLIFF]["sarsa"],
        },
        me.ROOM_WITH_MONSTER: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "sarsa" / f"{me.ROOM_WITH_MONSTER}.pkl",
            "episodes": 10,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.ROOM_WITH_MONSTER]["sarsa"],
        },
    },
    "monte_carlo": {
        me.EMPTY_ROOM: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "monte_carlo" / f"{me.EMPTY_ROOM}.pkl",
            "episodes": 1,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.EMPTY_ROOM]["monte_carlo"],
        },
        me.CLIFF: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "monte_carlo" / f"{me.CLIFF}.pkl",
            "episodes": 1,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.CLIFF]["monte_carlo"],
        },
        me.ROOM_WITH_MONSTER: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "tabular" / "monte_carlo" / f"{me.ROOM_WITH_MONSTER}.pkl",
            "episodes": 10,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "threshold": TRAINING_TEST_THRESHOLDS[me.ROOM_WITH_MONSTER]["monte_carlo"],
        },
    },
}


def load_tabular_checkpoint(path: Path):
    assert path.exists(), f"Missing pretrained checkpoint: {path}"
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert "q_table" in checkpoint, f"Checkpoint at {path} does not contain 'q_table'."
    assert "state_map" in checkpoint, f"Checkpoint at {path} does not contain 'state_map'."
    return checkpoint


def make_env_for_checkpoint(env_id, checkpoint, max_episode_steps):
    return me.get_minihack_environment(
        env_id,
        observation_mode=checkpoint.get("observation_mode", "coords"),
        max_episode_steps=max_episode_steps,
        reward_win=1,
        reward_lose=0,
        penalty_step=0,
        penalty_time=0,
    )


def seed_evaluation_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)


def evaluate_agent(env, agent, n_episodes, max_steps_per_episode):
    previous_learning = agent.learning
    agent.learning = False
    returns = []
    env_factory = getattr(env, "_eval_factory", None)

    try:
        for episode in range(n_episodes):
            episode_seed = EVAL_SEED + episode
            current_env = env_factory() if env_factory is not None else env

            try:
                seed_evaluation_rngs(episode_seed)
                current_env.action_space.seed(episode_seed)
                if hasattr(current_env, "observation_space") and current_env.observation_space is not None:
                    current_env.observation_space.seed(episode_seed)

                state, _ = current_env.reset(seed=episode_seed)
                reward = None
                episode_return = 0.0
                step = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action = agent.act(state, reward=reward, step=step)
                    next_state, reward, terminated, truncated, _ = current_env.step(action)
                    episode_return += reward
                    state = next_state
                    step += 1

                    if max_steps_per_episode is not None and step >= max_steps_per_episode:
                        truncated = True

                agent.onEpisodeEnd(state, action, reward, episode)
                returns.append(episode_return)
            finally:
                if current_env is not env:
                    current_env.close()
    finally:
        agent.learning = previous_learning

    return returns


def load_pretrained_tabular_agent(agent_class, algorithm_name, env_id, config):
    checkpoint = load_tabular_checkpoint(config["checkpoint"])
    assert checkpoint["algorithm"] == algorithm_name
    assert checkpoint["env_id"] == env_id

    env = make_env_for_checkpoint(env_id, checkpoint, config["max_episode_steps"])
    agent = agent_class(
        id=f"eval-{algorithm_name}-{env_id}",
        action_space=env.action_space,
        epsilon=0.0,
        initialization_value=checkpoint.get("initialization_value", 0.0),
    )
    agent.load_checkpoint_data(checkpoint)
    env._eval_factory = lambda: make_env_for_checkpoint(env_id, checkpoint, config["max_episode_steps"])
    return env, agent

def run_training(env, agent, n_episodes, max_steps_per_episode):
    returns = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        reward = None
        episode_return = 0.0
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(state, reward=reward, step=step)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state
            step += 1

            if max_steps_per_episode is not None and step >= max_steps_per_episode:
                truncated = True

        agent.onEpisodeEnd(state, action, reward, episode)
        returns.append(episode_return)

    return returns

def make_empty_room_training_env():
    return me.get_minihack_environment(
        me.EMPTY_ROOM,
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_win=1,
        reward_lose=0,
        penalty_step=0,
        penalty_time=0,
        observation_mode="coords",
    )

def make_cliff_training_env():
    return me.get_minihack_environment(
        me.CLIFF,
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_win=1,
        reward_lose=0,
        penalty_step=0,
        penalty_time=0,
        observation_mode="coords",
    )

def make_room_with_monster_training_env():
    return me.get_minihack_environment(
        me.ROOM_WITH_MONSTER,
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_win=1,
        reward_lose=0,
        penalty_step=0,
        penalty_time=0,
        observation_mode="coords",
    )

def get_threshold(environment_name, algorithm_name):
    return TRAINING_TEST_THRESHOLDS[environment_name][algorithm_name]

class SpyTDAgent(TDAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learn_calls = []

    def learn(self, state, action, next_state, next_action, reward):
        self.learn_calls.append((state, action, next_state, next_action, reward))


class SpyMCAgent(MCAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learn_call_count = 0
        self.learn_call_snapshots = []

    def learn(self):
        self.learn_call_count += 1
        self.learn_call_snapshots.append(list(self.last_episode))

@pytest.fixture
def action_space():
    return gym.spaces.Discrete(4)

#######################
## END CONFIGURATION ##
#######################

##########################################################################
## CORRECTNESS TESTS                                                    ##
## These tests check the correctness of your algorithms implementations ##               
##########################################################################

def test_monte_carlo_backward_returns_and_incremental_average(action_space: gym.Space):
    # Checks that Monte Carlo updates compute backward returns correctly and apply incremental averaging with visit counts.
    agent = MCAgent(id="mc", action_space=action_space, gamma=0.9, epsilon=0.1)
    agent.last_episode = [
        (0, 1, 1.0),
        (2, 3, 2.0),
        (0, 1, 3.0),
    ]

    agent.learn()

    g_last = 3.0
    g_mid = 2.0 + agent.gamma * g_last
    g_first = 1.0 + agent.gamma * g_mid

    q_01_after_last_visit = 0.0 + (g_last - 0.0) / 2.0
    q_01_final = q_01_after_last_visit + (g_first - q_01_after_last_visit) / 3.0
    q_23_final = 0.0 + (g_mid - 0.0) / 2.0

    assert np.isclose(agent.q_table[0][1], q_01_final)
    assert np.isclose(agent.q_table[2][3], q_23_final)
    assert agent.counts[0][1] == 3.0
    assert agent.counts[2][3] == 2.0

def test_mc_agent_on_episode_end_updates_q_table(action_space: gym.Space):
    # Checks that Monte Carlo agents append the last transition, update the Q-table, and clear episode memory at episode end.
    agent = MCAgent(id="mc", action_space=action_space, gamma=1.0, epsilon=0.0)
    agent.last_state = 0
    agent.last_action = 2
    agent.last_episode = [(1, 1, 4.0)]

    agent.onEpisodeEnd(state=None, action=None, reward=3.0, episode=0)

    expected_last = 3.0
    expected_first = 4.0 + expected_last

    q_after_last = 0.0 + (expected_last - 0.0) / 2.0
    q_after_first = 0.0 + (expected_first - 0.0) / 2.0

    assert np.isclose(agent.q_table[0][2], q_after_last)
    assert np.isclose(agent.q_table[1][1], q_after_first)
    assert agent.last_episode == []

def test_sarsa_update_rule(action_space: gym.Space):
    # Checks that the SARSA update uses the value of the actually selected next action.
    agent = SARSAOnPolicyAgent(id="sarsa", action_space=action_space, alpha=0.5, gamma=0.9, epsilon=0.1)
    state = 0
    action = 1
    next_state = 1
    next_action = 3
    reward = 5.0

    agent.q_table[state] = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    agent.q_table[next_state] = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)

    old_q = agent.q_table[state][action]
    expected = old_q + agent.alpha * (reward + agent.gamma * agent.q_table[next_state][next_action] - old_q)

    agent.learn(state=state, action=action, next_state=next_state, next_action=next_action, reward=reward)

    assert np.isclose(agent.q_table[state][action], expected)

def test_q_learning_update_rule(action_space: gym.Space):
    # Checks that the Q-learning update matches the expected one-step TD target with a max over next actions.
    agent = QLearning(id="q-learning", action_space=action_space, alpha=0.5, gamma=0.9, epsilon=0.1)
    state = 0
    action = 2
    next_state = 1
    reward = 10.0

    agent.q_table[state] = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    agent.q_table[next_state] = np.array([4.0, 5.0, 6.0, 7.0], dtype=float)

    old_q = agent.q_table[state][action]
    expected = old_q + agent.alpha * (reward + agent.gamma * np.max(agent.q_table[next_state]) - old_q)

    agent.learn(state=state, action=action, next_state=next_state, next_action=None, reward=reward)

    assert np.isclose(agent.q_table[state][action], expected)

def test_choose_next_action_is_greedy_when_not_learning(action_space: gym.Space):
    # Checks that epsilon-greedy action selection becomes fully greedy when learning is disabled.
    agent = QLearning(id="q-learning", action_space=action_space, epsilon=0.9)
    state = 0
    agent.q_table[state] = np.array([1.0, 4.0, 2.0, 3.0], dtype=float)
    agent.learning = False

    with patch.object(action_space, "sample", side_effect=AssertionError("should not explore")):
        assert agent.choose_next_action(state) == 1

def test_choose_next_action_explores_when_random_draw_is_below_epsilon(action_space: gym.Space):
    # Checks that epsilon-greedy action selection explores when the random draw falls below epsilon.
    agent = QLearning(id="q-learning", action_space=action_space, epsilon=0.5)
    state = 0
    agent.q_table[state] = np.array([10.0, 9.0, 8.0, 7.0], dtype=float)
    agent.learning = True

    with patch.object(td_agents.random, "uniform", return_value=0.1):
        with patch.object(action_space, "sample", return_value=3):
            assert agent.choose_next_action(state) == 3

def test_choose_next_action_exploits_when_random_draw_is_above_epsilon(action_space: gym.Space):
    # Checks that epsilon-greedy action selection exploits the best Q-value when the random draw exceeds epsilon.
    agent = QLearning(id="q-learning", action_space=action_space, epsilon=0.5)
    state = 0
    agent.q_table[state] = np.array([10.0, 11.0, 8.0, 7.0], dtype=float)
    agent.learning = True

    with patch.object(td_agents.random, "uniform", return_value=0.9):
        assert agent.choose_next_action(state) == 1

def test_linear_epsilon_decay_updates_as_expected(action_space: gym.Space):
    # Checks that the optional linear epsilon schedule follows the expected values and saturates at epsilon_end.
    agent = QLearning(
        id="q-learning",
        action_space=action_space,
        epsilon=1.0,
        epsilon_schedule=True,
        epsilon_end=0.2,
        epsilon_decay_episodes=4,
    )

    agent.update_epsilon(episode=0)
    assert np.isclose(agent.epsilon, 0.8)

    agent.update_epsilon(episode=1)
    assert np.isclose(agent.epsilon, 0.6)

    agent.update_epsilon(episode=2)
    assert np.isclose(agent.epsilon, 0.4)

    agent.update_epsilon(episode=3)
    assert np.isclose(agent.epsilon, 0.2)

    agent.update_epsilon(episode=10)
    assert np.isclose(agent.epsilon, 0.2)

def test_mc_agent_act_buffers_transitions_and_does_not_learn_during_episode(action_space: gym.Space):
    # Checks that MC's act() buffers the previous transition into last_episode on each step
    # and does NOT call learn() during the episode (MC only learns at episode end).
    agent = SpyMCAgent(id="spy-mc", action_space=action_space, epsilon=0.0)

    with patch.object(agent, "state_encoder", side_effect=[10, 20, 30]):
        with patch.object(agent, "choose_next_action", side_effect=[1, 2, 3]):
            first_action = agent.act({"chars": np.array([[64]])}, reward=None)
            second_action = agent.act({"chars": np.array([[64]])}, reward=5.0)
            third_action = agent.act({"chars": np.array([[64]])}, reward=2.0)

    assert first_action == 1
    assert second_action == 2
    assert third_action == 3
    assert agent.learn_call_count == 0
    assert agent.last_episode == [(10, 1, 5.0), (20, 2, 2.0)]
    assert agent.last_state == 30
    assert agent.last_action == 3


def test_mc_agent_on_episode_end_triggers_learn_and_clears_buffer(action_space: gym.Space):
    # Checks that MC's onEpisodeEnd() appends the final transition, calls learn() exactly once,
    # and clears the episode buffer so the next episode starts from scratch.
    agent = SpyMCAgent(id="spy-mc", action_space=action_space, epsilon=0.0)
    agent.last_state = 7
    agent.last_action = 3
    agent.last_episode = [(1, 0, 2.0)]

    agent.onEpisodeEnd(state=None, action=None, reward=-1.0, episode=0)

    assert agent.learn_call_count == 1
    assert agent.learn_call_snapshots[0] == [(1, 0, 2.0), (7, 3, -1.0)]
    assert agent.last_episode == []
    assert agent.last_state is None
    assert agent.last_action is None


def test_td_agent_act_triggers_learning_from_previous_transition(action_space: gym.Space):
    # Checks that TD agents call the learning rule on the previous transition when a new state is observed.
    agent = SpyTDAgent(id="spy-td", action_space=action_space, epsilon=0.0)

    with patch.object(agent, "state_encoder", side_effect=[10, 20]):
        with patch.object(agent, "choose_next_action", side_effect=[1, 2]):
            first_action = agent.act({"chars": np.array([[64]])}, reward=None)
            second_action = agent.act({"chars": np.array([[64]])}, reward=5.0)

    assert first_action == 1
    assert second_action == 2
    assert agent.learn_calls == [(10, 1, 20, 2, 5.0)]

def test_td_agent_on_episode_end_triggers_final_update(action_space: gym.Space):
    # Checks that TD agents perform the final learning update when the episode ends.
    agent = SpyTDAgent(id="spy-td", action_space=action_space, epsilon=0.0)
    agent.last_state = 7
    agent.last_action = 3

    with patch.object(agent, "state_encoder", return_value=11):
        agent.onEpisodeEnd({"chars": np.array([[64]])}, None, -1.0, 0)

    assert agent.learn_calls == [(7, 3, 11, None, -1.0)]

###########################
## END CORRECTNESS TESTS ##
###########################

###########################################################################################
## PERFORMANCE TESTS                                                                     ##
## The outcome of these tests will depend on BOTH the correctness of your implementation ##
## and the choice of hyperparameters that you used to train the weights.                 ##
###########################################################################################

def test_monte_carlo_improves_on_empty_room():
    # Checks that the pretrained Monte Carlo agent reaches the configured average return on the empty-room environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["monte_carlo"][me.EMPTY_ROOM]
    env, agent = load_pretrained_tabular_agent(MCAgent, "monte_carlo", me.EMPTY_ROOM, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_monte_carlo_improves_on_cliff():
    # Checks that the pretrained Monte Carlo agent reaches the configured average return on the cliff environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["monte_carlo"][me.CLIFF]
    env, agent = load_pretrained_tabular_agent(MCAgent, "monte_carlo", me.CLIFF, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_monte_carlo_improves_on_room_with_monster():
    # Checks that the pretrained Monte Carlo agent reaches the configured average return on the room-with-monster environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["monte_carlo"][me.ROOM_WITH_MONSTER]
    env, agent = load_pretrained_tabular_agent(MCAgent, "monte_carlo", me.ROOM_WITH_MONSTER, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_sarsa_improves_on_empty_room():
    # Checks that the pretrained SARSA agent reaches the configured average return on the empty-room environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["sarsa"][me.EMPTY_ROOM]
    env, agent = load_pretrained_tabular_agent(SARSAOnPolicyAgent, "sarsa", me.EMPTY_ROOM, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_sarsa_improves_on_cliff():
    # Checks that the pretrained SARSA agent reaches the configured average return on the cliff environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["sarsa"][me.CLIFF]
    env, agent = load_pretrained_tabular_agent(SARSAOnPolicyAgent, "sarsa", me.CLIFF, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_sarsa_improves_on_room_with_monster():
    # Checks that the pretrained SARSA agent reaches the configured average return on the room-with-monster environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["sarsa"][me.ROOM_WITH_MONSTER]
    env, agent = load_pretrained_tabular_agent(SARSAOnPolicyAgent, "sarsa", me.ROOM_WITH_MONSTER, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_q_learning_improves_on_empty_room():
    # Checks that the pretrained Q-learning agent reaches the configured average return on the empty-room environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["q_learning"][me.EMPTY_ROOM]
    env, agent = load_pretrained_tabular_agent(QLearning, "q_learning", me.EMPTY_ROOM, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_q_learning_improves_on_cliff():
    # Checks that the pretrained Q-learning agent reaches the configured average return on the cliff environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["q_learning"][me.CLIFF]
    env, agent = load_pretrained_tabular_agent(QLearning, "q_learning", me.CLIFF, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_q_learning_improves_on_room_with_monster():
    # Checks that the pretrained Q-learning agent reaches the configured average return on the room-with-monster environment.
    config = PRETRAINED_TABULAR_EVAL_CONFIGS["q_learning"][me.ROOM_WITH_MONSTER]
    env, agent = load_pretrained_tabular_agent(QLearning, "q_learning", me.ROOM_WITH_MONSTER, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], config["max_episode_steps"])
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

###########################
## END PERFORMANCE TESTS ##                                                                           
###########################

CORRECTNESS_TEST_NAMES = [
    "test_monte_carlo_backward_returns_and_incremental_average",
    "test_mc_agent_on_episode_end_updates_q_table",
    "test_sarsa_update_rule",
    "test_q_learning_update_rule",
    "test_choose_next_action_is_greedy_when_not_learning",
    "test_choose_next_action_explores_when_random_draw_is_below_epsilon",
    "test_choose_next_action_exploits_when_random_draw_is_above_epsilon",
    "test_linear_epsilon_decay_updates_as_expected",
    "test_mc_agent_act_buffers_transitions_and_does_not_learn_during_episode",
    "test_mc_agent_on_episode_end_triggers_learn_and_clears_buffer",
    "test_td_agent_act_triggers_learning_from_previous_transition",
    "test_td_agent_on_episode_end_triggers_final_update",
]

PERFORMANCE_TEST_NAMES = [
    "test_monte_carlo_improves_on_empty_room",
    "test_monte_carlo_improves_on_cliff",
    "test_monte_carlo_improves_on_room_with_monster",
    "test_sarsa_improves_on_empty_room",
    "test_sarsa_improves_on_cliff",
    "test_sarsa_improves_on_room_with_monster",
    "test_q_learning_improves_on_empty_room",
    "test_q_learning_improves_on_cliff",
    "test_q_learning_improves_on_room_with_monster",
]

for test_name in CORRECTNESS_TEST_NAMES:
    globals()[test_name] = pytest.mark.correctness(globals()[test_name])

for test_name in PERFORMANCE_TEST_NAMES:
    globals()[test_name] = pytest.mark.performance(globals()[test_name])
