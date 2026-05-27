import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from task3.deep_agent import BaseDeepMinihackAgent, DeepRLTask
import minihack_env as me


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)


class ConvActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        channels, height, width = input_shape
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            encoder_dim = self.encoder(dummy).shape[1]

        self.actor = nn.Sequential(
            layer_init(nn.Linear(encoder_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(encoder_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.actor(features), self.critic(features)


class PPOAgent(BaseDeepMinihackAgent):
    def __init__(
        self,
        id,
        action_space,
        state_encoding="coords",
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.8,
        clip_coef=1,
        clip_vloss=True,
        ent_coef=0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=1,
        num_minibatches=8,
        num_steps=64,
        anneal_lr=False,
        target_kl=None,
        normalize_advantages=True,
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
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.target_kl = target_kl
        self.normalize_advantages = normalize_advantages

        self.policy = None
        self.optimizer = None

        self.last_logprob = None
        self.last_value = None

    def build_actor_critic(self, encoded_state):
        if encoded_state.ndim == 1:
            return MLPActorCritic(encoded_state.shape[0], self.action_space.n)
        if encoded_state.ndim == 3:
            return ConvActorCritic(encoded_state.shape, self.action_space.n)
        raise ValueError(f"Unsupported encoded_state shape: {encoded_state.shape}")

    def _maybe_initialize_networks(self, encoded_state):
        if self.policy is not None:
            return
        self.policy = self.build_actor_critic(encoded_state).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)

    def get_action_and_value(self, encoded_state, action=None):
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.policy(state_tensor)
        distribution = Categorical(logits=logits)

        if action is None:
            if self.learning:
                action = distribution.sample()
            else:
                action = torch.argmax(logits, dim=1)
        else:
            action = action.view(-1)

        logprob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, logprob, entropy, value.squeeze(-1)

    def act(self, state, reward=0, step=0):
        encoded_state = self.state_encoder(state)
        self._maybe_initialize_networks(encoded_state)

        with torch.no_grad():
            action, logprob, _, value = self.get_action_and_value(encoded_state)

        self.last_logprob = float(logprob.item())
        self.last_value = float(value.item())
        return int(action.item())

    def evaluate_actions(self, states, actions):
        logits, values = self.policy(states)
        distribution = Categorical(logits=logits)
        logprobs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return logprobs, entropy, values.squeeze(-1)


class PPOTask(DeepRLTask):
    def interact(self, n_episodes, max_steps_per_episode=None, log_every=None):
        returns = []
        completed_episodes = 0
        update_count = 0

        state, _ = self._reset_env()
        episode_return = 0.0
        episode_step = 0
        done = False

        encoded_state = self.agent.state_encoder(state)
        self.agent._maybe_initialize_networks(encoded_state)

        while completed_episodes < n_episodes:
            rollout = []

            if self.agent.anneal_lr:
                total_updates = max(int(np.ceil(n_episodes / max(self.agent.num_steps, 1))), 1)
                frac = 1.0 - min(update_count, total_updates - 1) / total_updates
                self.agent.optimizer.param_groups[0]["lr"] = frac * self.agent.learning_rate

            for _ in range(self.agent.num_steps):
                action = self.agent.act(state, step=episode_step)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_step += 1

                if max_steps_per_episode is not None and episode_step >= max_steps_per_episode:
                    done = True

                rollout.append(
                    {
                        "state": self.agent.state_encoder(state),
                        "action": action,
                        "logprob": self.agent.last_logprob,
                        "value": self.agent.last_value,
                        "reward": reward,
                        "done": float(done),
                    }
                )

                episode_return += reward
                state = next_state

                if done:
                    returns.append(episode_return)
                    completed_episodes += 1
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
                    self.agent.onEpisodeEnd(state, action, reward, completed_episodes - 1)
                    if completed_episodes >= n_episodes:
                        break
                    state, _ = self._reset_env()
                    episode_return = 0.0
                    episode_step = 0

            with torch.no_grad():
                if done:
                    next_value = 0.0
                else:
                    next_encoded_state = self.agent.state_encoder(state)
                    _, _, _, value = self.agent.get_action_and_value(next_encoded_state)
                    next_value = float(value.item())

            self.update_policy(rollout, next_value)
            update_count += 1

        if log_every is not None:
            print()

        return returns

    def update_policy(self, rollout, next_value):
        states = torch.tensor(
            np.stack([transition["state"] for transition in rollout]),
            dtype=torch.float32,
            device=self.agent.device,
        )
        actions = torch.tensor(
            [transition["action"] for transition in rollout],
            dtype=torch.long,
            device=self.agent.device,
        )
        old_logprobs = torch.tensor(
            [transition["logprob"] for transition in rollout],
            dtype=torch.float32,
            device=self.agent.device,
        )
        rewards = torch.tensor(
            [transition["reward"] for transition in rollout],
            dtype=torch.float32,
            device=self.agent.device,
        )
        dones = torch.tensor(
            [transition["done"] for transition in rollout],
            dtype=torch.float32,
            device=self.agent.device,
        )
        values = torch.tensor(
            [transition["value"] for transition in rollout],
            dtype=torch.float32,
            device=self.agent.device,
        )

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rollout))):
            if t == len(rollout) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
                
            # TODO: Compute the GAE advantage estimate for step t.
            # ### YOUR SOLUTION STARTS HERE
            delta = rewards[t] + self.agent.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = delta + self.agent.gamma * self.agent.gae_lambda * next_non_terminal * last_gae
            last_gae = advantages[t]
            # ### END OF YOUR SOLUTION

        returns = advantages + values

        batch_size = len(rollout)
        minibatch_size = max(batch_size // self.agent.num_minibatches, 1)
        batch_indices = np.arange(batch_size)

        for _ in range(self.agent.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_indices = batch_indices[start:end]

                mb_states = states[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_logprobs = old_logprobs[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                mb_values = values[minibatch_indices]

                if self.agent.normalize_advantages and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_logprobs, entropy, new_values = self.agent.evaluate_actions(mb_states, mb_actions)
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = log_ratio.exp()
    
                # TODO: compute the PPO clipped objective loss
                # ### YOUR SOLUTION STARTS HERE
                policy_loss = -torch.min(ratio * mb_advantages, torch.clamp(ratio, 1 - self.agent.clip_coef, 1 + self.agent.clip_coef) * mb_advantages).mean()
                # ### END OF YOUR SOLUTION

                if self.agent.clip_vloss:
                    value_loss_unclipped = (new_values - mb_returns) ** 2
                    value_clipped = mb_values + torch.clamp(
                        new_values - mb_values,
                        -self.agent.clip_coef,
                        self.agent.clip_coef,
                    )
                    value_loss_clipped = (value_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = policy_loss - self.agent.ent_coef * entropy_loss + self.agent.vf_coef * value_loss
                
                self.agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.agent.max_grad_norm)
                self.agent.optimizer.step()

                if self.agent.target_kl is not None:
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    if approx_kl > self.agent.target_kl:
                        return


def make_ppo_minihack_task(
    env_id=me.EMPTY_ROOM,
    state_encoding="coords",
    max_episode_steps=100000,
    seed=None,
    **agent_kwargs,
):
    env_keys = {
        "obs_crop_w",
        "obs_crop_h",
        "obs_crop_pad",
        "size",
        "random",
        "reward_win",
        "reward_lose",
        "penalty_step",
        "penalty_time",
        "wrapper_goal_reward",
        "wrapper_negative_step_reward",
        "wrapper_dead_negative_reward",
    }
    env_kwargs = {key: agent_kwargs.pop(key) for key in list(agent_kwargs.keys()) if key in env_keys}

    env = me.get_minihack_environment(
        env_id,
        observation_mode=state_encoding,
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )
    agent = PPOAgent(
        id=f"ppo-{state_encoding}",
        action_space=env.action_space,
        state_encoding=state_encoding,
        **agent_kwargs,
    )
    return PPOTask(env=env, agent=agent, seed=seed)

