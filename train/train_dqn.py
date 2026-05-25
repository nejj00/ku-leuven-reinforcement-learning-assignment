import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import minihack_env as me
from task3.dqn import make_dqn_minihack_task

ENVIRONMENTS = [
    me.EMPTY_ROOM,
    me.ROOM_WITH_MONSTER,
    me.CLIFF,
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train the DQN agent on one of the MiniHack environments.")

    # DO NOT CHANGE THESE ARGUMENTS!!!
    parser.add_argument(
        "--state-encoding",
        default="pixel_crop",
        choices=["coords", "relative-coords", "chars", "chars_crop", "glyphs", "glyphs_crop", "pixel", "pixel_crop"],
        help="Observation encoding used by the DQN agent.",
    )
    parser.add_argument("--obs-crop-w", type=int, default=3, help="Width of the MiniHack cropped observation in tiles.")
    parser.add_argument("--obs-crop-h", type=int, default=3, help="Height of the MiniHack cropped observation in tiles.")
    parser.add_argument("--obs-crop-pad", type=int, default=0, help="Padding used for the MiniHack cropped observation.")
    parser.add_argument("--max-episode-steps", type=int, default=50, help="Maximum steps per episode.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print a simple progress bar every N completed episodes. Use 0 to disable logging.",
    )
    parser.add_argument(
        "--save-dir",
        default="trained_weights",
        help="Directory where the trained DQN checkpoint will be saved.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    # END "DO NOT CHANGE"

    # CHANGE TO TRAIN ON DIFFERENT ENVIRONMENTS [EMPTY_ROOM, ROOM_WITH_MONSTER, CLIFF]
    parser.add_argument("--env", default=me.EMPTY_ROOM, choices=ENVIRONMENTS, help="MiniHack environment to use.")

    # CHANGE TO TUNE HYPERPARAMETERS
    parser.add_argument("--episodes", type=int, default=0, help="Number of training episodes.")
    parser.add_argument("--learning-rate", type=float, default=0, help="Optimizer learning rate.")
    parser.add_argument("--epsilon-start", type=float, default=0, help="Initial epsilon for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-end", type=float, default=0, help="Final epsilon for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=0, help="Number of steps used to decay epsilon.")
    parser.add_argument("--batch-size", type=int, default=0, help="Replay-buffer batch size.")
    parser.add_argument("--buffer-size", type=int, default=0, help="Replay-buffer capacity.")
    parser.add_argument("--learning-starts", type=int, default=0, help="Number of transitions before DQN starts learning.")
    parser.add_argument("--train-frequency", type=int, default=0, help="Train the Q-network every N environment steps.")
    parser.add_argument("--target-network-frequency", type=int, default=0, help="Update the target network every N environment steps.",)
    parser.add_argument("--tau", type=float, default=0, help="Soft-update coefficient for the target network.")
    # END "CHANGE TO TUNE HYPERPARAMETERS"

    return parser.parse_args()

def save_checkpoint(task, args):
    save_dir = Path(args.save_dir) / "dqn"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.env}.pt"
    checkpoint = {
        "algorithm": "dqn",
        "env_id": args.env,
        "state_encoding": args.state_encoding,
        "obs_crop_w": args.obs_crop_w,
        "obs_crop_h": args.obs_crop_h,
        "obs_crop_pad": args.obs_crop_pad,
        "max_episode_steps": args.max_episode_steps,
        "model_state_dict": task.agent.q_network.state_dict(),
    }
    torch.save(checkpoint, save_path)
    return save_path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    task = make_dqn_minihack_task(
        env_id=args.env,
        state_encoding=args.state_encoding,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        obs_crop_w=args.obs_crop_w,
        obs_crop_h=args.obs_crop_h,
        obs_crop_pad=args.obs_crop_pad,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        target_network_frequency=args.target_network_frequency,
        tau=args.tau,
    )

    returns = task.interact(
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_episode_steps,
        log_every=args.log_every if args.log_every > 0 else None,
    )

    print(f"Finished training on {args.env} with state encoding '{args.state_encoding}'.")
    print(f"Number of episodes: {len(returns)}")
    print(f"Average return over all episodes: {float(np.mean(returns)):.3f}")
    print(f"Average return over last 10 episodes: {float(np.mean(returns[-10:])):.3f}")
    print("Last 10 returns:", returns[-10:])
    save_path = save_checkpoint(task, args)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
