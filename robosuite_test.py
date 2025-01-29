import robosuite
import robosuite.environments as environments
from robosuite.controllers import load_controller_config
from robosuite import make
from robosuite.wrappers.gym_wrapper import GymWrapper

controller_config = load_controller_config(default_controller="JOINT_POSITION")
lift_env = make(
    "Lift",
    robots=["IIWA"],
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True
)
lift_env = GymWrapper(lift_env)

# Reset the environment
lift_env.reset()

# You can now interact with the environment
action = lift_env.action_space.sample()
state, reward, done, truncated, info = lift_env.step(action)



stack_env = make(
    "Stack",
    robots=["IIWA"],
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True
)
stack_env = GymWrapper(stack_env)

# Reset the environment
stack_env.reset()

# You can now interact with the environment
action = stack_env.action_space.sample()
state, reward, done, truncated, info = stack_env.step(action)


pickplace_env = make(
    "PickPlace",
    robots=["IIWA"],
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True
)
pickplace_env = GymWrapper(pickplace_env)

# Reset the environment
pickplace_env.reset()

# You can now interact with the environment
action = lift_env.action_space.sample()
state, reward, done, truncated, info = lift_env.step(action)

