# -*- coding: utf-8 -*-
import os
import copy
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist

class Communication(object):
    META_INF_SZ = 3
    META_INF_IDX_PROC_ID = 0
    META_INF_IDX_MSG_TYPE = 1
    META_INF_IDX_MSG_DATA = 2
    
    # message type (META_INF_IDX_MSG_TYPE) values
    MSG_TYPE_SEND_REQ = 0
    MSG_TYPE_RECV_RESP = 1
    MSG_TYPE_RECV_REQ = 2
    MSG_TYPE_SEND_RESP = 3

    # message data (META_INF_IDX_MSG_DATA) values
    MSG_DATA_NULL = 0 # an empty message
    MSG_DATA_SET = 1

    # number of seconds to sleep/wait
    SLEEP_DURATION = 1

    def __init__(self, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port):
        super(Communication, self).__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.task_label_sz = task_label_sz
        self.mask_sz = mask_sz
        self.logger = logger

        if init_address in ['127.0.0.1', 'localhost']:
            os.environ['MASTER_ADDR'] = init_address
            os.environ['MASTER_PORT'] = init_port
            comm_init_str = 'env://'
        else:
            comm_init_str = 'tcp://{0}:{1}'.format(init_address, init_port)

        self.handle_send_recv_req = None
        self.handle_recv_resp = [None, ] * num_agents
        self.handle_send_resp = [None, ] * num_agents

        self.buff_send_recv_req = [torch.ones(Communication.META_INF_SZ + task_label_sz, ) \
            * torch.inf for _ in range(num_agents)]
        self.buff_recv_resp = [torch.ones(Communication.META_INF_SZ + mask_sz, ) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_resp = [torch.ones(Communication.META_INF_SZ + mask_sz, ) * torch.inf \
            for _ in range(num_agents)]

        logger.info('*****agent {0} / initialising transfer (communication) module'.format(agent_id))
        dist.init_process_group(backend='gloo', init_method=comm_init_str, rank=agent_id, \
            world_size=num_agents, timeout=datetime.timedelta(seconds=30))

    def _null_message(self, msg):
        # check whether message sent denotes or is none.
        if bool(msg[Communication.META_INF_IDX_MSG_DATA] == Communication.MSG_DATA_NULL):
            return True
        else:
            return False

    '''
    method to receive request from other agents (query whether current agent possess knowledge
    about queried task), as well as query (send request to) other agents.
    '''
    def send_receive_request(self, task_label):
        if isinstance(task_label, np.ndarray):
            task_label = torch.tensor(task_label, dtype=torch.float32)
        self.logger.info('send_recv_req, req data: {0}'.format(task_label))
        # from message to send from agent (current node), can be NULL message or a valid
        # request based on given task label
        data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf
        data[Communication.META_INF_IDX_PROC_ID] = self.agent_id
        data[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_REQ
        if task_label is None:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL
        else:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_SET
            data[Communication.META_INF_SZ : ] = task_label # NOTE deepcopy?

        # actual send/receive
        self.handle_send_recv_req = dist.all_gather(tensor_list=self.buff_send_recv_req, \
            tensor=data, async_op=True)
        # briefly wait to see if other agents will send their request
        time.sleep(Communication.SLEEP_DURATION)
        
        # check buffer for incoming requests
        idxs = list(range(len(self.buff_send_recv_req)))
        idxs.remove(self.agent_id)
        ret = []
        for idx in idxs :
            buff = self.buff_send_recv_req[idx]
            if self._null_message(buff):
                ret.append(None)
            else:
                self.logger.info('send_recv_req: request received from agent {0}'.format(idx))
                d = {}
                d['sender_agent_id'] = int(buff[Communication.META_INF_IDX_PROC_ID])
                d['msg_type'] = int(buff[Communication.META_INF_IDX_MSG_TYPE])
                d['msg_data'] = int(buff[Communication.META_INF_IDX_MSG_DATA])
                d['task_label'] = buff[Communication.META_INF_SZ : ]
                ret.append(d)
        return ret

    def send_response(self, requesters):
        self.logger.info('send_resp:')
        for requester in requesters:
            self._send_response(requester)

    def _send_response(self, req_dict):
        requester_agent_id = req_dict['sender_agent_id']
        mask = req_dict['mask']
        buff = self.buff_send_resp[requester_agent_id]
        buff[Communication.META_INF_IDX_PROC_ID] = self.agent_id
        buff[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_RESP

        self.logger.info('send_resp: responding to agent {0} query'.format(requester_agent_id))
        self.logger.info('send_resp: mask (response) data type: {0}'.format(type(mask)))

        if mask is None:
            buff[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL
            buff[Communication.META_INF_SZ : ] = torch.inf
        else:
            buff[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_SET
            buff[Communication.META_INF_SZ : ] = mask # NOTE deepcopy? 

        # actual send
        self.handle_send_resp = dist.isend(tensor=buff, dst=requester_agent_id)
        return

    def receive_response(self):
        self.logger.info('recv_resp:')
        for idx in range(self.num_agents):
            if idx == self.agent_id:
                continue
            if self.handle_recv_resp[idx] is None:
                self.logger.info('recv_resp: set up handle to receive response from agent {0}'.format(idx))
                self.handle_recv_resp[idx] = dist.irecv(tensor=self.buff_recv_resp[idx], src=idx)

        time.sleep(Communication.SLEEP_DURATION)

        # check whether message has been received
        ret = []
        for idx in range(self.num_agents):
            if idx == self.agent_id:
                ret.append(None)
                continue

            msg = self.buff_recv_resp[idx]
            if self._null_message(msg):
                ret.append(None)
                self.logger.info('recv_resp: appending {0} response'.format(None))
            elif msg[Communication.META_INF_IDX_MSG_DATA] == torch.inf:
                ret.append(False)
                self.logger.info('recv_resp: appending False response')
            else:
                mask = copy.deepcopy(msg[Communication.META_INF_SZ : ])
                ret.append(mask)
                self.logger.info('recv_resp: appending {0} response'.format(mask))

            # reset buffer and handle
            self.buff_recv_resp[idx][:] = torch.inf
            self.handle_recv_resp[idx] = None 
        return ret

    def barrier(self):
        dist.barrier()

