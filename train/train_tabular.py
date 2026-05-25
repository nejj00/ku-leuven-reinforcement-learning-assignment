import argparse
import importlib.util
import pickle
from pathlib import Path
import random
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import minihack_env as me

ENVIRONMENTS = [
    me.EMPTY_ROOM,
    me.ROOM_WITH_MONSTER,
    me.CLIFF,
]

def load_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


td_agents = load_module("td_agents", "task2/td-agents.py")
mc_agents = load_module("mc_agents", "task2/mc-agents.py")
task1_dummy = load_module("task1_dummy", "task1/dummy.py")

AbstractRLTask = task1_dummy.AbstractRLTask


ALGORITHMS = {
    "q_learning": td_agents.QLearning,
    "sarsa": td_agents.SARSAOnPolicyAgent,
    "monte_carlo": mc_agents.MCAgent,
}


def parse_args():

    parser = argparse.ArgumentParser(description="Train a tabular RL agent on one of the MiniHack environments.")

    # DO NOT CHANGE THESE ARGUMENTS!!!
    parser.add_argument("--max-episode-steps", type=int, default=50, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for NumPy.")
    parser.add_argument("--save-dir", default="trained_weights", help="Directory where the trained tabular checkpoint will be saved.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    # END "DO NOT CHANGE"

    # CHANGE TO TRAIN ON DIFFERENT ENVIRONMENTS [EMPTY_ROOM, ROOM_WITH_MONSTER, CLIFF]
    parser.add_argument("--env", default=me.EMPTY_ROOM, choices=ENVIRONMENTS, help="MiniHack environment to use.")

    # CHANGE TO TRAIN DIFFERENT ALGORITHMS [q_learning, sarsa, monte_carlo]
    parser.add_argument("--algorithm", default="q_learning", choices=ALGORITHMS, help="Tabular RL algorithm to train.")

    # CHANGE TO TUNE HYPERPARAMETERS
    parser.add_argument("--episodes", type=int, default=0, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0, help="Learning rate.")
    parser.add_argument("--epsilon", type=float, default=0, help="Exploration rate.")
    # NOTE: --epsilon-schedule is False by default. It becomes True only if you pass --epsilon-schedule on the command line, which enables linear epsilon decay over episodes (if implemented).
    parser.add_argument("--epsilon-schedule", action="store_true", help="Enable linear epsilon decay over episodes.")
    parser.add_argument("--epsilon-end", type=float, default=0, help="Minimum epsilon value reached by the linear decay schedule.",)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=0, help="Number of episodes used by the linear epsilon decay schedule.",)
    parser.add_argument("--initialization-value", type=float, default=0.0, help="Initial Q-value for unseen states.")
    # END "CHANGE TO TUNE HYPERPARAMETERS"

    return parser.parse_args()

def get_agent_coords(observation):
    chars = observation["chars"]
    coords = np.argwhere(chars == ord("@"))
    if coords.size == 0:
        return None
    return tuple(int(x) for x in coords[0])


def save_checkpoint(agent, args):
    save_dir = Path(args.save_dir) / "tabular" / args.algorithm
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.env}.pkl"
    checkpoint = {
        "algorithm": args.algorithm,
        "env_id": args.env,
        "observation_mode": "coords",
        "max_episode_steps": args.max_episode_steps,
        **agent.checkpoint_data(),
    }
    with save_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    return save_path


def build_agent(agent_class, args, action_space):
    return agent_class(
        id=args.algorithm,
        action_space=action_space,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_schedule=args.epsilon_schedule,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        initialization_value=args.initialization_value,
    )


def main():

    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = me.get_minihack_environment(
        args.env,
        max_episode_steps=args.max_episode_steps,
        observation_mode="coords",
    )
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)

    agent_class = ALGORITHMS[args.algorithm]
    agent = build_agent(agent_class, args, env.action_space)

    # Reuse the interaction loop implemented in Task 1.
    task = AbstractRLTask(env=env, agent=agent)
    returns = task.interact(
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_episode_steps,
    )

    print(f"Finished training {args.algorithm} on {args.env}.")
    print(f"Number of episodes: {len(returns)}")
    print(f"Average return over all episodes: {float(np.mean(returns)):.3f}")
    print(f"Average return over last 10 episodes: {float(np.mean(returns[-10:])):.3f}")
    save_path = save_checkpoint(agent, args)
    print(f"Saved checkpoint to {save_path}")
    env.close()


if __name__ == "__main__":
    main()
