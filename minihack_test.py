import minihack
import gym

env = gym.make("MiniHack-River-v0")
state = env.reset()
print(state)

state, reward, done, info = env.step(env.action_space.sample())
#print(state, reward, done, info)
env.render()
