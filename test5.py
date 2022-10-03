"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import multiprocessing as mp
import numpy as np
import datetime

import argparse

def run(rank, size):
    """ Distributed function to be implemented later. """
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1

        req = dist.isend(tensor=tensor, dst=1)
    else:

        req = dist.irecv(tensor=tensor, src=0)

    req.wait()

    print('Rank', rank, ' has data ', tensor[0])
    print(type(tensor[0]))


class Communication(object):
    META_INF_TASK_SZ = 3
    META_INF_MASK_SZ = 4
    META_INF_IDX_PROC_ID = 0
    META_INF_IDX_MSG_TYPE = 1
    META_INF_IDX_MSG_DATA = 2
    META_INF_IDX_MSK_RW = 3

    # message type (META_INF_IDX_MSG_TYPE) values
    MSG_TYPE_SEND_REQ = 0
    MSG_TYPE_RECV_RESP = 1
    MSG_TYPE_RECV_REQ = 2
    MSG_TYPE_SEND_RESP = 3

    # message data (META_INF_IDX_MSG_DATA) values
    MSG_DATA_NULL = 0 # an empty message
    MSG_DATA_MSK = 1
    MSG_DATA_RW = 2

    # number of seconds to sleep/wait
    SLEEP_DURATION = 10
                
    def __init__(self, rank, size):
        print(rank, size)
        self.buff_send_recv_req = [torch.ones(Communication.META_INF_TASK_SZ + 3, ) \
                * torch.inf for _ in range(2)]


    def send_receive_request(self, rank, size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='gloo', rank=rank, world_size=size, timeout=datetime.timedelta(seconds=30))

        task_label = torch.tensor([0., 0., 1.])
        data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf
        data[Communication.META_INF_IDX_PROC_ID] = rank
        data[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_REQ
        if task_label is None:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL
        else:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_MSK
            data[Communication.META_INF_TASK_SZ : ] = task_label # NOTE deepcopy?

        print(data)
        req = dist.all_gather(self.buff_send_recv_req, data, async_op=True)
        print('hello')

        req.wait()
        print('goodbye')


        print(self.buff_send_recv_req)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int)
    parser.add_argument('size', type=int)
    args = parser.parse_args()
    size = args.size
    rank = args.rank

    processes = []
    mp.set_start_method("fork")
    comm = Communication(rank, size)
    p = mp.Process(target=comm.send_receive_request, args=(rank, size))
    p.start()
    p.join()
    #pool = mp.Pool(1)
    #res = pool.apply_async(comm.send_receive_request, (rank,))

    #print(res.get())