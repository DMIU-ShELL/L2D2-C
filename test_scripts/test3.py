import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, size):
    tensor = torch.zeros(1)
    # Send the tensor to process 1
    while True:
        time.sleep(1)
        req = dist.isend(tensor=tensor, dst=1, tag=5)
        req.wait()
        print('Rank ', rank, ' has data ', tensor[0])
        tensor += 1

def run2(rank, size):
    # Receive tensor from process 0
    while True:
        tensor = torch.zeros(1)
        time.sleep(10)
        req = dist.irecv(tensor=tensor, src=0, tag=5)
        req.wait()
        print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '0.0.0.0'
    os.environ['MASTER_PORT'] = '5000'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        if rank == 0:
            p = mp.Process(target=init_process, args=(rank, size, run))
            p.start()
            processes.append(p)
        else:
            p = mp.Process(target=init_process, args=(rank, size, run2))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
