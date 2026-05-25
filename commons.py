import numpy as np

blank = 32


def _get_non_empty_char_bounds(observation):
    chars = observation["chars"]
    coords = np.argwhere(chars != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max


def get_crop_chars_from_observation(observation):
    chars = observation["chars"]
    x_min, y_min, x_max, y_max = _get_non_empty_char_bounds(observation)
    chars = chars[x_min:x_max + 1, y_min:y_max + 1]
    return chars


def get_crop_array_from_observation(observation, key):
    array = observation[key]
    x_min, y_min, x_max, y_max = _get_non_empty_char_bounds(observation)
    return array[x_min:x_max + 1, y_min:y_max + 1]


size_pixel = 16
def get_crop_pixel_from_observation(observation):
    x_min, y_min, x_max, y_max = _get_non_empty_char_bounds(observation)
    non_empty_pixels = observation["pixel"][x_min * size_pixel : (x_max + 1) * size_pixel, y_min * size_pixel : (y_max + 1) * size_pixel]
    return non_empty_pixels

class AbstractAgent():

    def __init__(self, id, action_space):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning = True

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        raise NotImplementedError()


    def onEpisodeEnd(self,*args, **kwargs):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass
