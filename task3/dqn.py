from task3.deep_agent import DeepMinihackAgent, DeepRLTask
import minihack_env as me
import torch.nn.functional as F
import torch


class DQNAgent(DeepMinihackAgent):
    """
    Teaching-oriented DQN agent for MiniHack.

    The reusable logic lives in `DeepMinihackAgent`:
    - observation encoding
    - epsilon-greedy exploration
    - replay buffer management
    - target network updates

    """

    def learn(self):
        batch = self.replay_buffer.sample(self.batch_size, self.device)

        # TODO: compute the DQN loss 
        # ### YOUR SOLUTION STARTS HERE
        print(f"Batch: {batch}")
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = F.mse_loss(q_values, target_q_values)
        # ### END OF YOUR SOLUTION

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def make_dqn_minihack_task(
    env_id=me.EMPTY_ROOM,
    state_encoding="coords",
    max_episode_steps=200,
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
    agent = DQNAgent(
        id=f"dqn-{state_encoding}",
        action_space=env.action_space,
        state_encoding=state_encoding,
        **agent_kwargs,
    )
    return DeepRLTask(env=env, agent=agent, seed=seed)

