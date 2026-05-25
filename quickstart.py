import minihack_env as me
import matplotlib.pyplot as plt
import commons
# mpl.use('MacOSX')   #uncomment this in some MacOSX machines for matplotlib

# How to get a minihack environment from the minihack_env utility.
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id)
state = env.reset()
print("Initial state", state)
next_state = env.step(1)
print("Next State", next_state)

# How to get a minihack environment with also pixels states
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id, add_pixel=True)
state = env.reset()
print("Initial state", state)
plt.imshow(state[0]["pixel"]) # CHECK: for some reason there is a tuple around the observation
plt.show()

# Crop representations to non-empty part
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id, add_pixel=True)
state = env.reset()
print("Initial state", commons.get_crop_chars_from_observation(state[0])) # CHECK: for some reason there is a tuple around the observation
plt.imshow(commons.get_crop_pixel_from_observation(state[0])) # CHECK: for some reason there is a tuple around the observation
plt.show()