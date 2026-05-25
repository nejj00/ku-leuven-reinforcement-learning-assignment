import gymnasium as gym
import minihack
from nle import nethack


ACTIONS = tuple(nethack.CompassCardinalDirection)

EMPTY_ROOM = "empty-room"
ROOM_WITH_MONSTER = "room-with-monster"
CLIFF = "cliff-minihack"


des_empty_room = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|-----|
|.....|
|.....|
|.....|
|.....|
|.....|
|-----|
ENDMAP
REGION: (0,0,6,6), lit, "ordinary"
STAIR:(5,5),down
BRANCH: (1,1,1,1),(2,2,2,2)
"""
des_cliff="""
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
-------------
|...........|
|...........|
|.LLLLLLLLL.|
-------------
ENDMAP
BRANCH: (1,3,1,3), (2,2,2,2)
REGION: (0,0,11,4), lit, "ordinary"
STAIR:(11,3),down

"""

des_monster = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|-----|
|.....|
|.....|
|.....|
|.....|
|.....|
|-----|
ENDMAP
REGION: (0,0,6,6), lit, "ordinary"
STAIR:(5,5),down
BRANCH: (1,1,1,1),(2,2,2,2)
MONSTER: ('Z', "ghoul"), (3,3)
"""

def get_minihack_environment(id, **kwargs):
    add_pixels = kwargs["add_pixel"] if "add_pixel" in kwargs else False
    observation_mode = kwargs["observation_mode"] if "observation_mode" in kwargs else None
    max_episode_steps = kwargs["max_episode_steps"] if "max_episode_steps" in kwargs else 1000
    obs_crop_w = kwargs["obs_crop_w"] if "obs_crop_w" in kwargs else 5
    obs_crop_h = kwargs["obs_crop_h"] if "obs_crop_h" in kwargs else 5
    obs_crop_pad = kwargs["obs_crop_pad"] if "obs_crop_pad" in kwargs else 0
    reward_win = kwargs["reward_win"] if "reward_win" in kwargs else 1
    reward_lose = kwargs["reward_lose"] if "reward_lose" in kwargs else 0
    penalty_step = kwargs["penalty_step"] if "penalty_step" in kwargs else 0
    penalty_time = kwargs["penalty_time"] if "penalty_time" in kwargs else 0
    if observation_mode is None:
        observation_mode = "pixel" if add_pixels else "coords"

    if observation_mode in {"coords", "relative-coords", "chars"}:
        obs = ["chars"]
    elif observation_mode == "chars_crop":
        obs = ["chars", "chars_crop"]
    elif observation_mode in {"glyphs", "glyphs_crop"}:
        obs = ["chars", "glyphs"]
        if observation_mode == "glyphs_crop":
            obs.append("glyphs_crop")
    elif observation_mode == "pixel":
        obs = ["chars", "pixel"]
    elif observation_mode == "pixel_crop":
        obs = ["chars", "pixel_crop"]
    else:
        raise ValueError(
            "Unsupported observation_mode. Expected one of: 'coords', 'relative-coords', 'chars', 'chars_crop', 'glyphs', 'glyphs_crop', 'pixel', 'pixel_crop'."
        )

    if id == EMPTY_ROOM:
        des_file = des_empty_room
        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs,
            obs_crop_w=obs_crop_w,
            obs_crop_h=obs_crop_h,
            obs_crop_pad=obs_crop_pad,
            reward_win=reward_win,
            reward_lose=reward_lose,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
        )
    elif id == CLIFF:
        des_file = des_cliff

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs,
            obs_crop_w=obs_crop_w,
            obs_crop_h=obs_crop_h,
            obs_crop_pad=obs_crop_pad,
            reward_win=reward_win,
            reward_lose=reward_lose,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
        )
    elif id == ROOM_WITH_MONSTER:
        des_file = des_monster

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs,
            obs_crop_w=obs_crop_w,
            obs_crop_h=obs_crop_h,
            obs_crop_pad=obs_crop_pad,
            reward_win=reward_win,
            reward_lose=reward_lose,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
        )
    else:
        raise Exception("Environment %s not found" % str(id))

    return env
