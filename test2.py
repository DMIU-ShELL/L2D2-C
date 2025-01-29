from gym.wrappers import RecordVideo
import gym
import minihack
from nle import nethack

# Wrap the environment with the Monitor for recording
MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (nethack.Command.OPEN, nethack.Command.KICK)

name = "MiniHack-MultiRoom-N2-v0"
env = gym.make(name, observation_keys=("pixel_crop",), actions=NAVIGATE_ACTIONS)
# Wrap the environment with RecordVideo
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode="rgb_array")  # Ensure rendering happens

env.close()