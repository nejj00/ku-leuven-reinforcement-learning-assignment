##################
## INSTRUCTIONS ##
##################

# Run from the repository root with:
#   docker compose run --rm pypython visualize_env.py
#
# To remove black borders from the displayed image:
#   docker compose run --rm py python visualize_env.py --trim-borders
#
# To visualize a specific environment:
#   docker compose run --rm py python visualize_env.py --env empty-room # (or room-with-monster, or cliff-minihack)
#
# To visualize all environments:
#   docker compose run --rm py python visualize_env.py --env all
#
# To use cropped pixel observations:
#   docker compose run --rm py python visualize_env.py --observation-mode pixel_crop
#
# You can combine the above options as needed, e.g.:
#   docker compose run --rm py python visualize_env.py --env empty-room --trim-borders

######################
## END INSTRUCTIONS ##
######################

import argparse
import matplotlib.pyplot as plt
import minihack_env as me

ENVIRONMENTS = [
    me.EMPTY_ROOM,
    me.ROOM_WITH_MONSTER,
    me.CLIFF,
]

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MiniHack environments as pixel observations.")
    parser.add_argument(
        "--env",
        default=me.EMPTY_ROOM,
        choices=ENVIRONMENTS + ["all"],
        help="Environment to visualize.",
    )
    parser.add_argument(
        "--observation-mode",
        default="pixel",
        choices=["pixel", "pixel_crop"],
        help="Pixel observation type to display.",
    )
    parser.add_argument(
        "--trim-borders",
        action="store_true",
        help="Crop fully black image borders before displaying the observation.",
    )
    return parser.parse_args()


def get_image(env_id, observation_mode):
    env = me.get_minihack_environment(
        env_id,
        observation_mode=observation_mode,
        max_episode_steps=1,
    )
    observation, _ = env.reset()
    image = observation[observation_mode]
    env.close()
    return image

def trim_black_borders(image):
    non_black = image.any(axis=2)
    rows = non_black.any(axis=1)
    cols = non_black.any(axis=0)

    if not rows.any() or not cols.any():
        return image

    row_indices = rows.nonzero()[0]
    col_indices = cols.nonzero()[0]
    return image[
        row_indices[0] : row_indices[-1] + 1,
        col_indices[0] : col_indices[-1] + 1,
    ]


def maybe_trim_image(image, trim_borders):
    if trim_borders:
        return trim_black_borders(image)
    return image


def show_single_environment(env_id, observation_mode, trim_borders):
    image = get_image(env_id, observation_mode)
    image = maybe_trim_image(image, trim_borders)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_all_environments(observation_mode,  trim_borders):
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for axis, env_id in zip(axes, ENVIRONMENTS):
        image = get_image(env_id, observation_mode)
        image = maybe_trim_image(image, trim_borders)
        axis.imshow(image)
        axis.axis("off")

    axes[-1].axis("off")
    figure.suptitle(f"MiniHack environments ({observation_mode})")
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    if args.env == "all":
        show_all_environments(
            observation_mode=args.observation_mode,
            trim_borders=args.trim_borders,
        )
    else:
        show_single_environment(
            env_id=args.env,
            observation_mode=args.observation_mode,
            trim_borders=args.trim_borders,
        )


if __name__ == "__main__":
    main()
