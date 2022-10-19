import torch
import torch.distributed as dist
import numpy as np
import multiprocess as mp
import os
import argparse

meta_responses = [{'dst_agent_id': 0, 'mask_reward': 0.9485069444444445, 'dist': 0.0, 'resp_task_label': torch.tensor([0., 1., 0.])}]
meta_responses = [{'dst_agent_id': 0, 'mask_reward': torch.inf, 'dist': torch.inf, 'resp_task_label': torch.tensor([torch.inf]*3)}]

# DETECT MODULE CONSTANTS
# Threshold for embedding/tasklabel distance (similarity)
THRESHOLD = 0.0

# buffer indexes
META_INF_IDX_PROC_ID = 0
META_INF_IDX_MSG_TYPE = 1
META_INF_IDX_MSG_DATA = 2

META_INF_IDX_MSK_RW = 3
META_INF_IDX_TASK_SZ = 3 # only for the send_recv_request buffer

META_INF_IDX_DIST = 4
META_INF_IDX_TASK_SZ_ = 5 # for the meta send recv buffer


META_INF_IDX_MASK_SZ = 6

# message type (META_INF_IDX_MSG_TYPE) values
MSG_TYPE_SEND_REQ = 0
MSG_TYPE_RECV_RESP = 1
MSG_TYPE_RECV_REQ = 2
MSG_TYPE_SEND_RESP = 3

# message data (META_INF_IDX_MSG_DATA) values
MSG_DATA_NULL = 0 # an empty message
MSG_DATA_TSK = 1
MSG_DATA_MSK = 2
MSG_DATA_META = 3

# number of seconds to sleep/wait
SLEEP_DURATION = 1


agent_id = 1


def _null_message(msg):
    # check whether message sent denotes or is none.
    if bool(msg[META_INF_IDX_MSG_DATA] == MSG_DATA_NULL):
        return True
    else:
        return False

def _send_meta_response(req_dict):
    dst_agent_id = req_dict['dst_agent_id']
    mask_reward = req_dict['mask_reward']
    distance = req_dict['dist']
    emb_label = req_dict['resp_task_label']

    #print(dst_agent_id, mask_reward, emb_label, flush=True)
    #print(type(dst_agent_id), type(mask_reward), type(emb_label), flush=True)

    if isinstance(emb_label, np.ndarray):
        emb_label = torch.tensor(emb_label, dtype=torch.float32)

    # Consider changing to local buffer
    buff = torch.ones(META_INF_IDX_TASK_SZ_ + 3, dtype=torch.float32) * torch.inf
    buff[META_INF_IDX_PROC_ID] = agent_id
    buff[META_INF_IDX_MSG_TYPE] = MSG_TYPE_SEND_RESP

    #self.logger.info('send_resp: responding to agent {0} query'.format(dst_agent_id))
    #self.logger.info('send_resp: mask (response) data type: {0}'.format(type(mask)))

    # If mask is none then send back torch.inf
    # otherwise send mask
    if mask_reward is torch.inf:
        #print('Mask reward is None -> True', flush=True)
        buff[META_INF_IDX_MSG_DATA] = MSG_DATA_NULL

    else:
        #print('Mask reward is None -> False', flush=True)
        #print(mask_reward, emb_label, distance, flush=True)
        #print(type(mask_reward), type(emb_label), type(distance), flush=True)
        # if the mask is none but there is a mask reward, then overwrite the buffer with
        # the meta data. Otherwise don't do anything.
        buff[META_INF_IDX_MSG_DATA] = MSG_DATA_META
        buff[META_INF_IDX_MSK_RW] = mask_reward
        buff[META_INF_IDX_DIST] = distance
        buff[META_INF_IDX_TASK_SZ_ :] = emb_label


    print(buff)
    #self.logger.info('Sending metadata to agent {0}: {1}'.format(dst_agent_id, buff, buff.dtype))
    # actual send
    print('STARTING SEND')
    req_send = dist.isend(tensor=buff, dst=dst_agent_id)
    req_send.wait()
    print('COMPLETED SEND', flush=True)
    return

def send_meta_response(requesters):
    '''
    Send response for each requester agent.
    '''
    if requesters:
        #self.logger.info('send_resp:')
        for requester in requesters:
            _send_meta_response(requester)



def receive_meta_response(await_response):
    '''
    Receives the response from all in the network agents.
    '''
    _buff_recv = torch.ones_like(torch.tensor([torch.inf]*5)) * torch.inf
    req_recv = dist.irecv(tensor=_buff_recv, src=1)
    req_recv.wait()
    
    print(_buff_recv[1])
    
    '''ret = list()
    _buff_recv = [torch.ones_like(torch.tensor([torch.inf]*5)) * torch.inf for _ in range(2)]
    print(_buff_recv)
    if any(await_response):
        for idx in range(2):
            print(idx)
            print(_buff_recv[idx])
            if idx == agent_id:
                continue
            #if self.handle_recv_resp[idx] is None:
                #self.logger.info('recv_resp: set up handle to receive response from agent {0}'.format(idx))
            req_recv = dist.irecv(tensor=_buff_recv[idx], src=idx)
            req_recv.wait()
            #print('COMPLETED RECEPTION from agent {0}'.format(idx), flush=True)

        #time.sleep(ParallelComm.SLEEP_DURATION)
        #meta_responses = [{'dst_agent_id': 0, 'mask_reward': 0.9485069444444445, 'dist': 0.0, 'resp_task_label': torch.tensor([0., 1., 0.])}]
        #_buff_recv = torch.tensor([0, 1, 3, 0.9485069444444445, 0.0, torch.tensor([0., 1., 0.])])
        print(_buff_recv)
        # check whether message has been received
        for idx in range(2):
            _buff = _buff_recv[idx]

            if idx == agent_id:
                d = {}
                d['agent_id'] = agent_id
                d['mask_reward'] = torch.inf
                d['dist'] = torch.inf
                d['emb_label'] = _buff[5:].detach().cpu().numpy()
                ret.append(d)

            else:
                if _null_message(_buff):
                    d = {}
                    d['agent_id'] = int(_buff[0])
                    d['mask_reward'] = torch.inf
                    d['dist'] = torch.inf
                    d['emb_label'] = _buff[5:].detach().cpu().numpy()
                    ret.append(d)
                    #self.logger.info('recv_resp: appending {0} response'.format(None))
                
                elif _buff[META_INF_IDX_MSG_DATA] == torch.inf:
                    ret.append(False)
                    #self.logger.info('recv_resp: appending False response. All hope is lost')

                elif _buff[META_INF_IDX_MSG_DATA] == MSG_DATA_META:
                    d = {}
                    d['agent_id'] = int(_buff[0])
                    d['mask_reward'] = float(_buff[3])
                    d['dist'] = float(_buff[4])
                    d['emb_label'] = _buff[5:].detach().cpu().numpy()
                    ret.append(d)'''
    return ret



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int)
    args = parser.parse_args()
    rank = args.rank

    init_address = '127.0.0.1'
    init_port = '25006'

    if init_address in ['127.0.0.1', 'localhost']:
        os.environ['MASTER_ADDR'] = init_address
        os.environ['MASTER_PORT'] = init_port
        comm_init_str = 'env://'
    else:
        comm_init_str = 'tcp://{0}:{1}'.format(init_address, init_port)

    dist.init_process_group(backend='gloo', init_method=comm_init_str, rank=rank, world_size=2)

    if rank == 1:
        send_meta_response(meta_responses)

    elif rank == 0:
        await_response = [True, True]
        result = receive_meta_response(await_response)
