import gymnasium as gym
import composuite

ROBOT = 'IIWA'
OBJ = 'Hollowbox'
OBSTACLE = 'None'
TASK = 'PickPlace'
CONTROLLER = 'joint'
HORIZON = 500



env = composuite.make(ROBOT, OBJ, OBSTACLE, TASK, CONTROLLER, HORIZON, use_task_id_obs=True)
state, done = env.reset()
print(state, done)

output = env.step(env.action_space.sample())
print(len(output))
#print(state, reward, done, truncated, info)