#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import copy
from .atari_wrapper import *
import multiprocessing as mp
import sys
from .bench import Monitor
from ..utils import *
import uuid

class BaseTask:
    def __init__(self):
        pass

    def set_monitor(self, env, log_dir):
        if log_dir is None:
            return env
        mkdir(log_dir)
        return Monitor(env, '%s/%s' % (log_dir, uuid.uuid4()))

    def reset(self):
        #print 'base task reset called'
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        #print 'base task step called'
        #print self.env
        #print done
        if done:
            next_state = self.env.reset()
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class ClassicalControl(BaseTask):
    def __init__(self, name, max_steps=200, log_dir=None):
        # Removed from name = "Cart_Pole-v0"
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class DynamicGrid(BaseTask):
    def __init__(self, name, max_steps=200, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        import dynamic_grid
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.env = self.set_monitor(self.env, log_dir)

        # total number of unique tasks in this class instance. note, the actual
        # environment (wrapped by this class) have many more task variation
        num_tasks = 3
        change_matrix = np.eye(num_tasks).astype(np.bool)
        self.tasks = self.env.unwrapped.unwrapped.random_tasks(change_matrix)
        self.current_task = self.tasks[0]

    def reset_task(self, task_info):
        self.set_task(task_info)
        return self.env.reset()

    def set_task(self, task_info):
        msg = '`{0}` parameter should be included in `task_info` in the DynamicGrid env'
        assert 'goal_location' in task_info.keys(), msg.format('goal_location')
        assert 'transition_dynamics' in task_info.keys(), msg.format('transition_dynamics')
        assert 'permute_input' in task_info.keys(), msg.format('permute_input')
        self.env.unwrapped.unwrapped.set_task(task_info)
        self.current_task = task_info

    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=True):
        if requires_task_label:
            tasks_label = np.eye(len(self.tasks))
            tasks = copy.deepcopy(self.tasks)
            for task, label in zip(tasks, tasks_label):
                task['task_label'] = label
            return tasks
        else:
            return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        tasks_idx = np.random.randint(low=0, high=len(self.tasks), size=(num_tasks,))
        if requires_task_label:
            all_tasks = copy.deepcopy(self.tasks)
            tasks_label = np.eye(len(all_tasks)).astype(np.float32)
            tasks = []
            for idx in tasks_idx:
                task = all_tasks[idx]
                task['task_label'] = tasks_label[idx]
                tasks.append(task)
            return tasks
        else:
            tasks = [self.tasks[idx] for idx in tasks_idx]
            return tasks

class CTgraph(BaseTask):
    def __init__(self, name, env_config_path, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        from gym_CTgraph import CTgraph_env
        from gym_CTgraph.CTgraph_conf import CTgraph_conf
        from gym_CTgraph.CTgraph_images import CTgraph_images
        env = gym.make(name)
        env_config = CTgraph_conf(env_config_path)
        env_config = env_config.getParameters()
        imageDataset = CTgraph_images(env_config)
        self.env_config=env_config

        state, _, _, _ = env.init(env_config, imageDataset)
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=state.shape)
        self.action_dim = env.action_space.n
        if env_config['image_dataset']['1D']:
            self.state_dim = int(np.prod(env.observation_space.shape))
        else:
            self.state_dim = env.observation_space.shape

        self.env = self.set_monitor(env, log_dir)

        # get all tasks in graph environment instance
        from itertools import product
        depth = env_config['graph_shape']['depth']
        branch = env_config['graph_shape']['depth']
        tasks = list(product(list(range(branch)), repeat=depth))
        self.tasks = [{'goal': np.array(task)} for task in tasks]
        self.current_task = self.tasks[0]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        if self.env_config['image_dataset']['1D']: state = state.ravel()
        return state, reward, done, info

    def reset(self):
        state, _, _, _ = self.env.reset() # ctgraph returns state, reward, done, info in reset
        if self.env_config['image_dataset']['1D']: state = state.ravel()
        # return only state when reset is called to conform with other env in the repo
        return state

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.env.unwrapped.set_high_reward_path(taskinfo['goal'])
        self.current_task = taskinfo
    
    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=False):
        if requires_task_label:
            tasks_label = np.eye(len(self.tasks)).astype(np.float32)
            tasks = copy.deepcopy(self.tasks)
            for task, label in zip(tasks, tasks_label):
                task['task_label'] = label
            return tasks
        else:
            return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        tasks_idx = np.random.randint(low=0, high=len(self.tasks), size=(num_tasks,))
        if requires_task_label:
            all_tasks = copy.deepcopy(self.tasks)
            tasks_label = np.eye(len(all_tasks)).astype(np.float32)
            tasks = []
            for idx in tasks_idx:
                task = all_tasks[idx]
                task['task_label'] = tasks_label[idx]
                tasks.append(task)
            return tasks
        else:
            tasks = [self.tasks[idx] for idx in tasks_idx]
            return tasks

class PixelAtari(BaseTask):
    def __init__(self, name, seed=0, log_dir=None,
                 frame_skip=4, history_length=4, dataset=False):
        BaseTask.__init__(self)
        env = make_atari(name, frame_skip)
        env.seed(seed)
        if dataset:
            env = DatasetEnv(env)
            self.dataset_env = env
        env = self.set_monitor(env, log_dir)
        env = wrap_deepmind(env, history_length=history_length)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.name = name

class RamAtari(BaseTask):
    def __init__(self, name, no_op, frame_skip, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = self.set_monitor(env, log_dir)
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = SkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = 128

class Pendulum(BaseTask):
    def __init__(self, log_dir=None):
        BaseTask.__init__(self)
        self.name = 'Pendulum-v0'
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(2 * action, -2, 2))

class Box2DContinuous(BaseTask):
    def __init__(self, name, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Roboschool(BaseTask):
    def __init__(self, name, log_dir=None):
        import roboschool
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Bullet(BaseTask):
    def __init__(self, name, log_dir=None):
        import pybullet_envs
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class PixelBullet(BaseTask):
    def __init__(self, name, seed=0, log_dir=None, frame_skip=4, history_length=4):
        import pybullet_envs
        self.name = name
        env = gym.make(name)
        env.seed(seed)
        env = RenderEnv(env)
        env = self.set_monitor(env, log_dir)
        env = SkipEnv(env, skip=frame_skip)
        env = WarpFrame(env)
        env = WrapPyTorch(env)
        if history_length:
            env = StackFrame(env, history_length)
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.env = env

class ProcessTask:
    def __init__(self, task_fn, log_dir=None):
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = ProcessWrapper(worker_pipe, task_fn, log_dir)
        self.worker.start()
        self.pipe.send([ProcessWrapper.SPECS, None])
        self.state_dim, self.action_dim, self.name = self.pipe.recv()

    def step(self, action):
        self.pipe.send([ProcessWrapper.STEP, action])
        return self.pipe.recv()

    def reset(self):
        self.pipe.send([ProcessWrapper.RESET, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([ProcessWrapper.EXIT, None])

    def reset_task(self, task_info):
        self.pipe.send([ProcessWrapper.RESET_TASK, task_info])
        return self.pipe.recv()

    def set_task(self, task_info):
        self.pipe.send([ProcessWrapper.SET_TASK, task_info])

    def get_task(self):
        self.pipe.send([ProcessWrapper.GET_TASK, None])
        return self.pipe.recv()

    def get_all_tasks(self, requires_task_label):
        self.pipe.send([ProcessWrapper.GET_ALL_TASKS, requires_task_label])
        return self.pipe.recv()
    
    def random_tasks(self, num_tasks):
        self.pipe.send([ProcessWrapper.RANDOM_TASKS, num_tasks])
        return self.pipe.recv()

class ProcessWrapper(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    RESET_TASK = 4
    SET_TASK = 5
    GET_TASK = 6
    GET_ALL_TASKS = 7
    RANDOM_TASKS = 8
    def __init__(self, pipe, task_fn, log_dir):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.task_fn = task_fn
        self.log_dir = log_dir

    def run(self):
        np.random.seed()
        seed = np.random.randint(0, sys.maxsize)
        task = self.task_fn(log_dir=self.log_dir)
        task.seed(seed)
        while True:
            op, data = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(task.step(data))
            elif op == self.RESET:
                self.pipe.send(task.reset())
            elif op == self.EXIT:
                self.pipe.close()
                return
            elif op == self.SPECS:
                self.pipe.send([task.state_dim, task.action_dim, task.name])
            elif op == self.RESET_TASK:
                self.pipe.send(task.reset_task(data))
            elif op == self.SET_TASK:
                self.pipe.send(task.set_task(data))
            elif op == self.GET_TASK:
                self.pipe.send(task.get_task())
            elif op == self.GET_ALL_TASKS:
                self.pipe.send(task.get_all_tasks(data))
            elif op == self.RANDOM_TASKS:
                self.pipe.send(task.random_tasks())
            else:
                raise Exception('Unknown command')

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir=None, single_process=False):

        if single_process:
            self.tasks = [task_fn(log_dir=log_dir) for _ in range(num_workers)]
        else:
            self.tasks = [ProcessTask(task_fn, log_dir) for _ in range(num_workers)]
        self.state_dim = self.tasks[0].state_dim
        self.action_dim = self.tasks[0].action_dim
        self.name = self.tasks[0].name
        self.single_process = single_process

    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return np.stack(results)

    def close(self):
        if self.single_process:
            return
        for task in self.tasks: task.close()

    def reset_task(self, task_info):
        results = [task.reset_task(task_info) for task in self.tasks]
        return np.stack(results)

    def set_task(self, task_info):
        for task in self.tasks:
            task.set_task(task_info)

    def get_task(self):
        return self.tasks[0].get_task()

    def get_all_tasks(self, requires_task_label):
        return self.tasks[0].get_all_tasks(requires_task_label)
    
    def random_tasks(self, num_tasks):
        return self.tasks[0].random_tasks(num_tasks)
