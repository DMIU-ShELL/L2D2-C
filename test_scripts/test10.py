import multiprocessing as mp
import time
import torch.distributed as dist
import torch
from queue import Empty

class Agent(object):
    def __init__(self, rank):
        self.rank = rank
        self.iteration = 0

    def iteration_loop(self, queue_a, queue_c):
        msg = 'Get Mask!'
        while True:
            time.sleep(1)


            # if the task label has changed tell comm to get mask
            if msg is not None:
                queue_c.put(msg)

            # then check if comm has sent a mask in this cycle. otherwise get it in a later cycle
            try:
                mask = queue_a.get_nowait()
                print('Received mask: ', mask, '. Distilling knowledge to network!')
            except Empty:
                print('No mask from network. Continue learning from scratch')
                

            msg = None

            print(mp.current_process().name, 'Iteration: ', self.iteration)
            self.iteration += 1

            # task change
            if self.iteration == 5:
                self.iteration = 0
                print('Task changing. Do Query')
                msg = 'Get Mask!'


    def run_iteration_loop(self, queue_a, queue_c):
        p = mp.Process(target=self.iteration_loop, args=(queue_a, queue_c))
        p.start()

        return p
    
    

class Communication(object):
    def __init__(self):
        self.mask = [0, 0, 1]

    # replace with actual communication code
    def get_mask_from_network(self):
        print(mp.current_process().name)
        return self.mask

    def do_something(self, queue_c, queue_a):
        while True:
            time.sleep(1)
            try:
                msg = queue_c.get_nowait()
                print('Received task: ', msg)

                if msg is not None:
                    queue_a.put(self.get_mask_from_network())
            except Empty:
                print(mp.current_process().name, 'is idling')

            msg = None

    def run_comm_loop(self, queue_c, queue_a):
        p = mp.Process(target=self.do_something, args=(queue_c, queue_a))
        p.start()

        return p


if __name__ == '__main__':
    print(mp.current_process().name, ' is initialising the system')
    time.sleep(1)

    manager = mp.Manager()
    queue_a = manager.Queue()
    queue_c = manager.Queue()


    agent = Agent(0)
    agent_p = agent.run_iteration_loop(queue_a, queue_c)

    comm = Communication()
    comm_p = comm.run_comm_loop(queue_c, queue_a)

    comm_p.join()
    agent_p.join()



    """
    queue = multiprocessing.Queue()

    size = 2
    processes = []

    for rank in size:
        p = multiprocessing.Process(target=worker, args=(queue,))
        p.start()
        processes.append(p)
    
    queue.put(MyFancyClass('Fancy Dan'))
    
    # Wait for the worker to finish
    queue.close()
    queue.join_thread()
    p.join()
    """