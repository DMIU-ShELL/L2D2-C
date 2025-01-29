class Robosuite(BaseTask):
    def __init__(self, name, env_config_path, log_dir=None, seed=1000):
        BaseTask.__init__(self)
        self.name = name
        import robosuite
        from robosuite.wrappers.gym_wrapper import GymWrapper
        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
        self.env_config = env_config
        _env_args = env_config['tasks']

        print(_env_args)
        if 'seeds' in env_config.keys():
            seeds = env_config['seeds']
        else:
            seeds = seed
            del seed

        # Select random task from seed using composuite.sample_tasks()

        if isinstance(seeds, int): seeds = [seeds,] * len(_env_args)
        elif isinstance(seeds, list):
            assert len(seeds) == len(_env_args), 'number of seeds in config file should match the number of tasks.'
        else: raise ValueError('invalid seed specification in config file')

        controller_config = robosuite.load_controller_config(default_controller="OSC_POSE")

        self.envs = dict()
        env_names = list()
        for kargs, seed in zip(_env_args, seeds):
            robot, task = kargs
            env = robosuite.make(
                task,
                robots=[robot],
                controller_configs=controller_config,
                use_camera_obs=False, # Camera observations (default True)
                has_renderer=False, # Disable on-screen rendering
                has_offscreen_renderer=True, # Use offscreen rendering if camera obs is enabled
                render_camera=None, # Which camera to render if using camera obs
                camera_names=["frontview"], # Cameras to include (if enabled)
                camera_heights=256,
                camera_widths=256,
                camera_depths=False,
                use_object_obs=True,
                reward_shaping=False
            )
            env = GymWrapper(env)

            self.envs['{0}_{1}Subtask'.format(robot, task)] = env
            env_names.append('{0}_{1}Subtask'.format(robot, task))

        #self.envs = {'{0}_seed{1}'.format(name, seed) : composuite.sample_tasks(experiment_type='default', num_train=1, shuffling_seed=seed) for name, seed in zip(env_names, seeds)}
        #env_names = ['{0}_seed{1}'.format(name, seed) for name, seed in zip(env_names, seeds)]


        print("\nenv_names:", env_names)
        print("\nself.envs:", self.envs)
        print("\n")
        self.observation_space = self.envs[env_names[0]].observation_space
        self.action_space = self.envs[env_names[0]].action_space
        self.state_dim = self.observation_space.shape

        if 'action_dim' in env_config.keys():
            self.action_dim = env_config['action_dim']
        else:
            self.action_dim = self.envs[env_names[0]].action_space.n

        for name in self.envs.keys():
            self.envs[name] = self.set_monitor(self.envs[name], log_dir)

        self.task_label_dim = env_config['label_dim']
        self.one_hot_labels = True if env_config['one_hot'] else False

        self.tasks = [{'name': name, 'task': name, 'task_label': None} \
                   for name in self.envs.keys()]
        
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32)
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]

        self.current_task = self.tasks[0]
        self.env = self.envs[self.current_task['task']]
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        # Example success criteria for the Lift task
        if self.env.env_name == "Lift":
            # Check if the object is above a certain height
            obj_height = self.env.sim.data.body_xpos[self.env.obj_body_id][2]
            success_threshold = 0.2  # Adjust this threshold as needed
            info['Success'] = obj_height > success_threshold

        if done or truncated:
            state = self.reset()
            done = done or truncated

        return state, reward, done, info
    
    def reset(self):
        state, info = self.env.reset()
        return state
    
    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()
    
    def set_task(self, taskinfo):
        self.current_task = taskinfo
        self.env = self.envs[self.current_task['task']]
    
    def get_task(self):
        return self.current_task
    
    def get_all_tasks(self, requires_task_label=True):
        return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        return NotImplementedError

class RobosuiteFlatObs(Robosuite):
    def __init__(self, name, env_config_path, log_dir=None, seed=1000, eval_mode=False):
        super(RobosuiteFlatObs, self).__init__(name, env_config_path, log_dir, eval_mode)
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_map = {}

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        # Example success criteria for the Lift task
        info['Success'] = 1 if reward == 1 else 0

        if done or truncated:
            state = self.reset()
            done = done or truncated

        return state.ravel(), reward, done, info
    
    def reset(self):
        state, info = self.env.reset()
        return state.ravel()
