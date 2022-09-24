import dis
from pathos.multiprocessing import ProcessingPool
import multiprocessing as mp

x = 5

class Config(object):
    def __init__(self):
        self.task_fn = None
        self.num_workers = 4
        self.log_dir = None


class Wrapper(object):
    def __init__(self, log_dir=None):
        self.state_dim = 3
        self.log_dir = log_dir   
        

class ParallelizedTasks(object):
    def __init__(self, task_fn, num_workers, log_dir=None):
        self.tasks = [task_fn(log_dir) for _ in range(num_workers)]


class Agent(object):
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()


def main():
    mp.set_start_method('spawn', force=True)
    config = Config()
    config.log_dir = '/directory/'
    task_fn = lambda log_dir: Wrapper(log_dir)
    config.task_fn = lambda: ParallelizedTasks(task_fn, config.num_workers, log_dir = config.log_dir)

    dis.dis(config.task_fn)
    print(type(config.task_fn))

    def work(foo):
        foo.work()

        
    
main()