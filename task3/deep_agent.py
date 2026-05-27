from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from commons import AbstractAgent, get_crop_array_from_observation, get_crop_chars_from_observation
from task1.dummy import AbstractRLTask

AGENT_CHAR = ord("@")
MAX_CHAR_CODE = 255.0
MAX_GLYPH_CODE = 6000.0


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, copy=True),
                int(action),
                float(reward),
                np.array(next_state, copy=True),
                float(done),
            )
        )

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            "actions": torch.tensor(actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_states": torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device),
        }


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class ConvQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        channels, height, width = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            encoder_dim = self.encoder(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class BaseDeepMinihackAgent(AbstractAgent):
    def __init__(
        self,
        id,
        action_space,
        state_encoding="coords",
        gamma=0.99,
        learning_rate=0,
        device=None,
    ):
        super().__init__(id, action_space)
        self.state_encoding = state_encoding
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._last_coord_encoding = None

    def state_encoder(self, observation):
        if self.state_encoding == "coords":
            chars = get_crop_chars_from_observation(observation)
            coords = np.argwhere(chars == AGENT_CHAR)
            if coords.size == 0:
                if self._last_coord_encoding is not None:
                    return self._last_coord_encoding.copy()
                raise ValueError("Could not locate the agent character '@' in observation['chars'].")

            row, col = coords[0]
            height, width = chars.shape

            # Use coordinates relative to the non-empty cropped room instead of
            # the full 21x79 screen. When the room is surrounded by walls, this
            # removes the constant screen offset and yields a more meaningful
            # local position signal.
            interior_row = max(row - 1, 0)
            interior_col = max(col - 1, 0)
            interior_height = max(height - 2, 1)
            interior_width = max(width - 2, 1)
            encoding = np.array(
                [
                    interior_row / max(interior_height - 1, 1),
                    interior_col / max(interior_width - 1, 1),
                ],
                dtype=np.float32,
            )
            self._last_coord_encoding = encoding
            return encoding

        if self.state_encoding == "relative-coords":
            chars = get_crop_chars_from_observation(observation)
            agent_coords = np.argwhere(chars == AGENT_CHAR)
            goal_coords = np.argwhere(chars == ord(">"))
            if agent_coords.size == 0:
                if self._last_coord_encoding is not None:
                    return self._last_coord_encoding.copy()
                raise ValueError("Could not locate the agent character '@' in observation['chars'].")
            if goal_coords.size == 0:
                raise ValueError("Could not locate the goal character '>' in observation['chars'].")

            agent_row, agent_col = agent_coords[0]
            goal_row, goal_col = goal_coords[0]
            height, width = chars.shape

            interior_agent_row = max(agent_row - 1, 0)
            interior_agent_col = max(agent_col - 1, 0)
            interior_goal_row = max(goal_row - 1, 0)
            interior_goal_col = max(goal_col - 1, 0)
            interior_height = max(height - 2, 1)
            interior_width = max(width - 2, 1)

            encoding = np.array(
                [
                    (interior_goal_row - interior_agent_row) / max(interior_height - 1, 1),
                    (interior_goal_col - interior_agent_col) / max(interior_width - 1, 1),
                ],
                dtype=np.float32,
            )
            self._last_coord_encoding = encoding
            return encoding

        if self.state_encoding == "pixel":
            pixels = observation["pixel"]
            return np.transpose(pixels, (2, 0, 1)).astype(np.float32) / 255.0

        if self.state_encoding == "pixel_crop":
            pixels = observation["pixel_crop"]
            return np.transpose(pixels, (2, 0, 1)).astype(np.float32) / 255.0

        if self.state_encoding == "chars":
            chars = observation["chars"].astype(np.float32)
            return chars[None, :, :] / MAX_CHAR_CODE

        if self.state_encoding == "chars_crop":
            chars = observation.get("chars_crop")
            if chars is None:
                chars = get_crop_chars_from_observation(observation)
            chars = chars.astype(np.float32)
            return chars[None, :, :] / MAX_CHAR_CODE

        if self.state_encoding == "glyphs":
            glyphs = observation["glyphs"].astype(np.float32)
            return glyphs[None, :, :] / MAX_GLYPH_CODE

        if self.state_encoding == "glyphs_crop":
            glyphs = observation.get("glyphs_crop")
            if glyphs is None:
                glyphs = get_crop_array_from_observation(observation, "glyphs")
            glyphs = glyphs.astype(np.float32)
            return glyphs[None, :, :] / MAX_GLYPH_CODE

        raise ValueError(
            "Unsupported state_encoding. Expected one of: 'coords', 'relative-coords', 'chars', 'chars_crop', 'glyphs', 'glyphs_crop', 'pixel', 'pixel_crop'."
        )

    def encoded_state_shape(self, encoded_state):
        return tuple(encoded_state.shape)


class DeepMinihackAgent(BaseDeepMinihackAgent):
    def __init__(
        self,
        id,
        action_space,
        state_encoding="coords",
        gamma=0.99,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=1000,
        batch_size=32,
        buffer_size=10_000,
        learning_starts=1_000,
        train_frequency=1,
        target_network_frequency=100,
        tau=1.0,
        device=None,
    ):
        super().__init__(
            id=id,
            action_space=action_space,
            state_encoding=state_encoding,
            gamma=gamma,
            learning_rate=learning_rate,
            device=device,
        )
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.q_network = None
        self.target_network = None
        self.optimizer = None

        self.last_state = None
        self.last_action = None
        self.global_step = 0

    def build_q_network(self, encoded_state):
        if encoded_state.ndim == 1:
            return MLPQNetwork(encoded_state.shape[0], self.action_space.n)
        if encoded_state.ndim == 3:
            return ConvQNetwork(encoded_state.shape, self.action_space.n)
        raise ValueError(f"Unsupported encoded_state shape: {encoded_state.shape}")

    def _maybe_initialize_networks(self, encoded_state):
        if self.q_network is not None:
            return

        self.q_network = self.build_q_network(encoded_state).to(self.device)
        self.target_network = self.build_q_network(encoded_state).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def epsilon(self):
        if not self.learning:
            return 0.0

        progress = min(self.global_step, self.epsilon_decay_steps)
        slope = (self.epsilon_end - self.epsilon_start) / max(self.epsilon_decay_steps, 1)
        return max(self.epsilon_start + slope * progress, self.epsilon_end)

    def q_values(self, encoded_state):
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_network(state_tensor).squeeze(0)

    def choose_next_action(self, encoded_state):
        if random.random() < self.epsilon():
            return self.action_space.sample()
        return int(torch.argmax(self.q_values(encoded_state)).item())

    def observe_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.global_step += 1

        enough_data = len(self.replay_buffer) >= max(self.batch_size, self.learning_starts)
        should_train = self.global_step % self.train_frequency == 0
        if self.learning and enough_data and should_train:
            self.learn()

        should_update_target = self.global_step % self.target_network_frequency == 0
        if self.learning and should_update_target:
            self.update_target_network()

    def act(self, state, reward=0, step=0):
        encoded_state = self.state_encoder(state)
        self._maybe_initialize_networks(encoded_state)

        if self.learning and reward is not None and self.last_state is not None:
            self.observe_transition(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=encoded_state,
                done=False,
            )

        action = self.choose_next_action(encoded_state)
        self.last_state = encoded_state
        self.last_action = action
        return action

    def onEpisodeEnd(self, state, action, reward, episode):
        encoded_state = self.state_encoder(state)
        self._maybe_initialize_networks(encoded_state)

        if self.learning and reward is not None and self.last_state is not None:
            self.observe_transition(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=encoded_state,
                done=True,
            )

        self.last_state = None
        self.last_action = None

    def learn(self):
        raise NotImplementedError()

    def update_target_network(self):
        # TODO: implement soft updates of the target network parameters 
        # ### YOUR SOLUTION STARTS HERE
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # ### END OF YOUR SOLUTION

class DeepRLTask(AbstractRLTask):
    def __init__(self, env, agent, seed=None):
        super().__init__(env, agent)
        self._pending_reset_seed = seed

        if seed is not None:
            self.env.action_space.seed(seed)
            if hasattr(self.env, "observation_space") and self.env.observation_space is not None:
                self.env.observation_space.seed(seed)

    def _reset_env(self):
        if self._pending_reset_seed is not None:
            seed = self._pending_reset_seed
            self._pending_reset_seed = None
            return self.env.reset(seed=seed)
        return self.env.reset()

    def interact(self, n_episodes, max_steps_per_episode=None, log_every=None):
        returns = []

        for episode in range(n_episodes):
            state, _ = self._reset_env()
            reward = None
            episode_return = 0.0
            step = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.agent.act(state, reward=reward, step=step)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_return += reward
                state = next_state
                step += 1

                if max_steps_per_episode is not None and step >= max_steps_per_episode:
                    truncated = True

            self.agent.onEpisodeEnd(state, action, reward, episode)
            returns.append(episode_return)

            completed_episodes = episode + 1
            if log_every is not None and completed_episodes % log_every == 0:
                bar_width = 30
                progress = completed_episodes / max(n_episodes, 1)
                filled = int(bar_width * progress)
                bar = "#" * filled + "-" * (bar_width - filled)
                window = returns[-min(10, len(returns)) :]
                avg_return = float(np.mean(window))
                print(
                    f"\r[{bar}] {completed_episodes}/{n_episodes} "
                    f"avg_return(last {len(window)})={avg_return:.2f}",
                    end="",
                    flush=True,
                )

        if log_every is not None:
            print()

        return returns

    def visualize_episode(self, max_number_steps=None):
        self.agent.learning = False

        state, _ = self._reset_env()
        reward = None
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self.agent.act(state, reward=reward, step=step)
            state, reward, terminated, truncated, _ = self.env.step(action)
            step += 1

            if max_number_steps is not None and step >= max_number_steps:
                break
