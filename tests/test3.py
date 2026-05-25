##################
## INSTRUCTIONS ##
##################

# Run from the repository root with:
#   docker compose run --rm py pytest tests/test3.py
#
# To run only correctness tests:
#   docker compose run --rm py pytest tests/test3.py -m correctness
#
# To run only performance tests:
#   docker compose run --rm py pytest tests/test3.py -m performance
#
# To run a single test:
#   docker compose run --rm py pytest tests/test3.py::test_dqn_pretrained_empty_room
#
# To run with more detailed output, use the -vv flag::
#   docker compose run --rm py pytest -vv tests/test3.py
#
# To run all tests in the repository:
#   docker compose run --rm py pytest tests/

######################
## END INSTRUCTIONS ##
######################

import math
import os
import random
import sys
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure repo root is on sys.path so local modules are importable when running pytest.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import minihack_env as me
from task3.dqn import DQNAgent
from task3.ppo import PPOAgent, PPOTask

###################
## CONFIGURATION ##
###################

PRETRAINED_DEEP_EVAL_CONFIGS = {
    "dqn": {
        me.EMPTY_ROOM: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "dqn" / f"{me.EMPTY_ROOM}.pt",
            "episodes": 1,
            "max_episode_steps": 50,
            "threshold": 1,
        },
        me.ROOM_WITH_MONSTER: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "dqn" / f"{me.ROOM_WITH_MONSTER}.pt",
            "episodes": 10,
            "max_episode_steps": 50,
            "threshold": 0.5,
        },
        me.CLIFF: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "dqn" / f"{me.CLIFF}.pt",
            "episodes": 1,
            "max_episode_steps": 50,
            "threshold": 1,
        },
    },
    "ppo": {
        me.EMPTY_ROOM: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "ppo" / f"{me.EMPTY_ROOM}.pt",
            "episodes": 1,
            "max_episode_steps": 50,
            "threshold": 1,
        },
        me.ROOM_WITH_MONSTER: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "ppo" / f"{me.ROOM_WITH_MONSTER}.pt",
            "episodes": 10,
            "max_episode_steps": 50,
            "threshold": 0.5,
        },
        me.CLIFF: {
            "checkpoint": Path(REPO_ROOT) / "trained_weights" / "ppo" / f"{me.CLIFF}.pt",
            "episodes": 1,
            "max_episode_steps": 50,
            "threshold": 1,
        },
    },
}

MAX_EPISODE_STEPS = 50
EVAL_SEED = 0

@pytest.fixture
def action_space():
    return gym.spaces.Discrete(2)


def load_checkpoint(path: Path):
    assert path.exists(), f"Missing pretrained checkpoint: {path}"
    checkpoint = torch.load(path, map_location="cpu")
    assert "model_state_dict" in checkpoint, f"Checkpoint at {path} does not contain 'model_state_dict'."
    assert "state_encoding" in checkpoint, f"Checkpoint at {path} does not contain 'state_encoding'."
    return checkpoint


def make_env_for_checkpoint(env_id, checkpoint, max_episode_steps):
    return me.get_minihack_environment(
        env_id,
        observation_mode=checkpoint["state_encoding"],
        max_episode_steps=max_episode_steps,
        obs_crop_w=checkpoint.get("obs_crop_w", 5),
        obs_crop_h=checkpoint.get("obs_crop_h", 5),
        obs_crop_pad=checkpoint.get("obs_crop_pad", 0),
    )


def seed_evaluation_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_pretrained_dqn(env_id, config):
    checkpoint = load_checkpoint(config["checkpoint"])
    assert checkpoint["env_id"] == env_id

    env = make_env_for_checkpoint(env_id, checkpoint, MAX_EPISODE_STEPS)
    agent = DQNAgent(
        id=f"eval-dqn-{env_id}",
        action_space=env.action_space,
        state_encoding=checkpoint["state_encoding"],
        device="cpu",
    )

    observation, _ = env.reset(seed=EVAL_SEED)
    encoded_state = agent.state_encoder(observation)
    agent._maybe_initialize_networks(encoded_state)
    agent.q_network.load_state_dict(checkpoint["model_state_dict"])
    agent.target_network.load_state_dict(checkpoint["model_state_dict"])
    env._eval_factory = lambda: make_env_for_checkpoint(env_id, checkpoint, MAX_EPISODE_STEPS)
    return env, agent


def load_pretrained_ppo(env_id, config):
    checkpoint = load_checkpoint(config["checkpoint"])
    assert checkpoint["env_id"] == env_id

    env = make_env_for_checkpoint(env_id, checkpoint, MAX_EPISODE_STEPS)
    agent = PPOAgent(
        id=f"eval-ppo-{env_id}",
        action_space=env.action_space,
        state_encoding=checkpoint["state_encoding"],
        device="cpu",
    )

    observation, _ = env.reset(seed=EVAL_SEED)
    encoded_state = agent.state_encoder(observation)
    agent._maybe_initialize_networks(encoded_state)
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    env._eval_factory = lambda: make_env_for_checkpoint(env_id, checkpoint, MAX_EPISODE_STEPS)
    return env, agent

#######################
## END CONFIGURATION ##
#######################

########################
## CORRECTNESS TESTS  ##
########################

def test_dqn_soft_target_update(action_space: gym.Space):
    # Checks that DQN target-network updates implement the expected soft-update interpolation.
    agent = DQNAgent(id="dqn", action_space=action_space, tau=0.25, device="cpu")
    agent.q_network = nn.Linear(1, 1, bias=False)
    agent.target_network = nn.Linear(1, 1, bias=False)

    with torch.no_grad():
        agent.q_network.weight.fill_(2.0)
        agent.target_network.weight.fill_(10.0)

    agent.update_target_network()

    expected = 0.25 * 2.0 + 0.75 * 10.0
    assert torch.isclose(agent.target_network.weight[0, 0], torch.tensor(expected))


def test_dqn_loss_matches_expected_mse(action_space: gym.Space):
    # Checks that the DQN loss matches the expected MSE between current Q-values and frozen TD targets.
    agent = DQNAgent(
        id="dqn",
        action_space=action_space,
        gamma=0.9,
        batch_size=2,
        learning_rate=1e-3,
        device="cpu",
    )
    agent.q_network = nn.Linear(2, 2, bias=False)
    agent.target_network = nn.Linear(2, 2, bias=False)
    agent.optimizer = torch.optim.SGD(agent.q_network.parameters(), lr=0.0)

    with torch.no_grad():
        agent.q_network.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        agent.target_network.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    batch = {
        "states": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "actions": torch.tensor([0, 1], dtype=torch.long),
        "rewards": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "next_states": torch.tensor([[1.0, 1.0], [2.0, 1.0]], dtype=torch.float32),
        "dones": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }
    agent.replay_buffer.sample = lambda batch_size, device: batch

    loss = agent.learn()

    expected_current_q = np.array([1.0, 4.0], dtype=float)
    expected_next_q = np.array([3.0, 4.0], dtype=float)
    expected_targets = np.array([1.0 + 0.9 * 3.0, 2.0], dtype=float)
    expected_loss = np.mean((expected_current_q - expected_targets) ** 2)

    assert np.isclose(loss, expected_loss)

class DummyPolicy(nn.Module):
    def __init__(self, initial_logprob):
        super().__init__()
        self.logprob_param = nn.Parameter(torch.tensor(float(initial_logprob), dtype=torch.float32))

def make_rollout(rewards, values, dones):
    rollout = []
    for reward, value, done in zip(rewards, values, dones):
        rollout.append(
            {
                "state": np.array([0.0, 0.0], dtype=np.float32),
                "action": 0,
                "logprob": 0.0,
                "value": float(value),
                "reward": float(reward),
                "done": float(done),
            }
        )
    return rollout

def compute_expected_gae(rewards, values, dones, gamma, gae_lambda, next_value):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages

def test_ppo_gae_recursion_affects_policy_update_as_expected(action_space: gym.Space):
    # Checks that PPO's GAE recursion produces the expected policy update on a controlled rollout.
    agent = PPOAgent(
        id="ppo",
        action_space=action_space,
        gamma=0.9,
        gae_lambda=0.8,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.0,
        max_grad_norm=1e9,
        normalize_advantages=False,
        num_minibatches=1,
        update_epochs=1,
        device="cpu",
    )
    agent.policy = DummyPolicy(initial_logprob=math.log(1.1))
    agent.optimizer = torch.optim.SGD(agent.policy.parameters(), lr=1.0)

    def fake_evaluate_actions(self, states, actions):
        batch_size = actions.shape[0]
        logprobs = self.policy.logprob_param.expand(batch_size)
        entropy = torch.zeros(batch_size, dtype=torch.float32)
        values = torch.zeros(batch_size, dtype=torch.float32)
        return logprobs, entropy, values

    agent.evaluate_actions = MethodType(fake_evaluate_actions, agent)
    task = PPOTask(env=None, agent=agent)

    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.5, 0.4, 0.2], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    next_value = 0.0
    rollout = make_rollout(rewards, values, dones)

    expected_advantages = compute_expected_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
        next_value=next_value,
    )
    expected_mean_advantage = expected_advantages.mean()

    initial_param = agent.policy.logprob_param.item()
    task.update_policy(rollout, next_value=next_value)
    updated_param = agent.policy.logprob_param.item()

    expected_updated_param = initial_param + math.exp(initial_param) * expected_mean_advantage
    assert np.isclose(updated_param, expected_updated_param, atol=1e-5)


def test_ppo_clipping_blocks_gradient_when_ratio_exceeds_clip_range(action_space: gym.Space):
    # Checks that PPO clipping prevents further policy updates when the probability ratio exceeds the clip range.
    agent = PPOAgent(
        id="ppo",
        action_space=action_space,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.0,
        normalize_advantages=False,
        num_minibatches=1,
        update_epochs=1,
        device="cpu",
    )
    agent.policy = DummyPolicy(initial_logprob=math.log(1.5))
    agent.optimizer = torch.optim.SGD(agent.policy.parameters(), lr=1.0)

    def fake_evaluate_actions(self, states, actions):
        batch_size = actions.shape[0]
        logprobs = self.policy.logprob_param.expand(batch_size)
        entropy = torch.zeros(batch_size, dtype=torch.float32)
        values = torch.zeros(batch_size, dtype=torch.float32)
        return logprobs, entropy, values

    agent.evaluate_actions = MethodType(fake_evaluate_actions, agent)
    task = PPOTask(env=None, agent=agent)

    rollout = make_rollout(
        rewards=np.array([1.0], dtype=np.float32),
        values=np.array([0.0], dtype=np.float32),
        dones=np.array([1.0], dtype=np.float32),
    )

    initial_param = agent.policy.logprob_param.item()
    task.update_policy(rollout, next_value=0.0)
    updated_param = agent.policy.logprob_param.item()

    assert np.isclose(updated_param, initial_param, atol=1e-7)

###########################
## END CORRECTNESS TESTS ##
###########################

###########################################################################################
## PERFORMANCE TESTS                                                                     ##
## The outcome of these tests will depend on BOTH the correctness of your implementation ##
## and the choice of hyperparameters that you used to train the weights.                 ##
###########################################################################################

def test_dqn_pretrained_empty_room():
    # Checks that the pretrained DQN network achieves the configured average return on the empty-room environment.
    config = PRETRAINED_DEEP_EVAL_CONFIGS["dqn"][me.EMPTY_ROOM]
    env, agent = load_pretrained_dqn(me.EMPTY_ROOM, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], MAX_EPISODE_STEPS)
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]


def test_dqn_pretrained_cliff():
    # Checks that the pretrained DQN network achieves the configured average return on the cliff environment.
    config = PRETRAINED_DEEP_EVAL_CONFIGS["dqn"][me.CLIFF]
    env, agent = load_pretrained_dqn(me.CLIFF, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], MAX_EPISODE_STEPS)
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]


def test_dqn_pretrained_room_with_monster():
    # Checks that the pretrained DQN network achieves the configured average return on the room-with-monster environment.
    config = PRETRAINED_DEEP_EVAL_CONFIGS["dqn"][me.ROOM_WITH_MONSTER]
    env, agent = load_pretrained_dqn(me.ROOM_WITH_MONSTER, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], MAX_EPISODE_STEPS)
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_ppo_pretrained_empty_room():
    # Checks that the pretrained PPO network achieves the configured average return on the empty-room environment.
    config = PRETRAINED_DEEP_EVAL_CONFIGS["ppo"][me.EMPTY_ROOM]
    env, agent = load_pretrained_ppo(me.EMPTY_ROOM, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], MAX_EPISODE_STEPS)
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_ppo_pretrained_cliff():
    # Checks that the pretrained PPO network achieves the configured average return on the cliff environment.
    config = PRETRAINED_DEEP_EVAL_CONFIGS["ppo"][me.CLIFF]
    env, agent = load_pretrained_ppo(me.CLIFF, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], MAX_EPISODE_STEPS)
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]

def test_ppo_pretrained_room_with_monster():
    # Checks that the pretrained PPO network achieves the configured average return on the room-with-monster environment.
    config = PRETRAINED_DEEP_EVAL_CONFIGS["ppo"][me.ROOM_WITH_MONSTER]
    env, agent = load_pretrained_ppo(me.ROOM_WITH_MONSTER, config)
    try:
        returns = evaluate_agent(env, agent, config["episodes"], MAX_EPISODE_STEPS)
    finally:
        env.close()
    assert float(np.mean(returns)) >= config["threshold"]
    
###########################
## END PERFORMANCE TESTS ##
###########################

CORRECTNESS_TEST_NAMES = [
    "test_dqn_soft_target_update",
    "test_dqn_loss_matches_expected_mse",
    "test_ppo_gae_recursion_affects_policy_update_as_expected",
    "test_ppo_clipping_blocks_gradient_when_ratio_exceeds_clip_range",
]

PERFORMANCE_TEST_NAMES = [
    "test_dqn_pretrained_empty_room",
    "test_dqn_pretrained_cliff",
    "test_dqn_pretrained_room_with_monster",
    "test_ppo_pretrained_empty_room",
    "test_ppo_pretrained_cliff",
    "test_ppo_pretrained_room_with_monster",
]

for test_name in CORRECTNESS_TEST_NAMES:
    globals()[test_name] = pytest.mark.correctness(globals()[test_name])

for test_name in PERFORMANCE_TEST_NAMES:
    globals()[test_name] = pytest.mark.performance(globals()[test_name])
