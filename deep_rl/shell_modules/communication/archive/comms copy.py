# -*- coding: utf-8 -*-
'''
_________                                           .__                  __   .__                 
\_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
/    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
\     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
 \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
        \/               \/       \/             \/          \/      \/                        \/ 

'''
from logging import BufferingFormatter
import os
import copy
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist
import multiprocess as mp
from queue import Empty


######## COMMUNICATION CLASSES
'''
Original communication class. Contains the original implementation of the communication module
fast but at the expense of bandwidth.
'''
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
        self.buff_recv_mask = [torch.ones(Communication.META_INF_SZ + mask_sz, ) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_mask = [torch.ones(Communication.META_INF_SZ + mask_sz, ) * torch.inf \
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
        buff = self.buff_send_mask[requester_agent_id]
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
        self.handle_send_resp[requester_agent_id] = dist.isend(tensor=buff, dst=requester_agent_id)
        return

    def receive_response(self):
        self.logger.info('recv_resp:')
        for idx in range(self.num_agents):
            if idx == self.agent_id:
                continue
            if self.handle_recv_resp[idx] is None:
                self.logger.info('recv_resp: set up handle to receive response from agent {0}'.format(idx))
                self.handle_recv_resp[idx] = dist.irecv(tensor=self.buff_recv_mask[idx], src=idx)

        time.sleep(Communication.SLEEP_DURATION)

        # check whether message has been received
        ret = []
        for idx in range(self.num_agents):
            if idx == self.agent_id:
                ret.append(None)
                continue

            msg = self.buff_recv_mask[idx]
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
            self.buff_recv_mask[idx][:] = torch.inf
            self.handle_recv_resp[idx] = None 
        return ret

    def barrier(self):
        dist.barrier()

'''
Revise communication class. Contains the new implementation of the communication module
Improves on the bandwidth efficiency by some margin by reducing the amount of masks
that are communicated over the network, but will likely take longer to complete

Is currently the communication method used in the parallelisation wrapper

TODO: Debug this version of the communication module. Ensure it is working as expected.
probably best to use it in a synchronised setting first before moving to full async mode.
To do this use waits for the handlers etc. The code should work fingers crossed.
'''
from colorama import Fore, Back, Style
class ParallelComm(object):
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

    # Task label size can be replaced with the embedding size.
    def __init__(self, agent_id, num_agents, emb_label_sz, mask_sz, logger, init_address, init_port, mode):
        super(ParallelComm, self).__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.emb_label_sz = emb_label_sz
        self.mask_sz = mask_sz
        self.logger = logger
        self.mode = mode

        print('MASK SIZE IS: ', self.mask_sz)
        
        # Setup init string for process group
        if init_address in ['127.0.0.1', 'localhost']:
            os.environ['MASTER_ADDR'] = init_address
            os.environ['MASTER_PORT'] = init_port
            self.comm_init_str = 'env://'
        else:
            self.comm_init_str = 'tcp://{0}:{1}'.format(init_address, init_port)

        # Setup async communication handlers
        self.handle_send_recv_req = None
        self.handle_recv_resp = [None, ] * num_agents
        self.handle_send_resp = [None, ] * num_agents

        # Setup communication buffers
        self.buff_send_recv_req = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ + emb_label_sz, dtype=torch.float32) \
            * torch.inf for _ in range(num_agents)]

        self.buff_recv_task = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_task = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]

        self.buff_recv_mask = [torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_mask = [torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]


        # Delete these buffers later (sync_gather_meta)
        self.buff_send_recv_meta = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) \
            * torch.inf for _ in range(num_agents)]
        self.buff_send_recv_msk_req = torch.ones(ParallelComm.META_INF_IDX_TASK_SZ + emb_label_sz, dtype=torch.float32) * torch.inf

    def init_dist(self):
        '''
        Initialise the process group for torch. Return boolean of is processes group initialised.
        '''
        self.logger.info('*****agent {0} / initialising transfer (communication) module'.format(self.agent_id))
        dist.init_process_group(backend='gloo', init_method=self.comm_init_str, rank=self.agent_id, \
            world_size=self.num_agents)

        return dist.is_initialized()

    def _null_message(self, msg):
        # check whether message sent denotes or is none.
        if bool(msg[Communication.META_INF_IDX_MSG_DATA] == Communication.MSG_DATA_NULL):
            return True
        else:
            return False

    # Original send and recv request of a tasklabel
    def send_receive_request(self, emb_label):
        '''
        Setup up the communication buffer with the necessary flags and data.
        Send buffer to all agents in the network.
        Then listen for any requests from other agents and unpack the buffer.
        Return the data to the main process for use in the next process.

        Task label can be interchangeable with the task embedding. Just needs to be a numpy 
        array to be compatible.

        # Data buffer for communication looks like this
        #   task label: [0, 0, 1]
        #   [agentid, communication type, is data or null flag, 0, 0, 1]
        '''
        #### TO DO: Switch task labels with embeddings?
        # or with new idea task labels are made for each embedding cluster?

        if isinstance(emb_label, np.ndarray):
            emb_label = torch.tensor(emb_label, dtype=torch.float32)

            
        self.logger.info('send_recv_req, req data: {0}'.format(emb_label))
        # from message to send from agent (current node), can be NULL message or a valid
        # request based on given task label
        data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf

        # Send the task label or embedding to the other agents.
        data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_REQ

        if emb_label is None:
            data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
        else:
            data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_TSK
            data[ParallelComm.META_INF_IDX_TASK_SZ : ] = emb_label # NOTE deepcopy?


        self.logger.info('actual send, recv')
        # actual send/receive
        self.handle_send_recv_req = dist.all_gather(tensor_list=self.buff_send_recv_req, \
            tensor=data, async_op=True)


        #time.sleep(ParallelComm.SLEEP_DURATION)
        self.handle_send_recv_req.wait()
        #print(self.buff_send_recv_req, flush=True)


        # briefly wait to see if other agents will send their request
        #time.sleep(ParallelComm.SLEEP_DURATION)
        
        # check buffer for incoming requests
        idxs = list(range(len(self.buff_send_recv_req)))
        idxs.remove(self.agent_id)
        ret = []
        for idx in idxs :
            buff = self.buff_send_recv_req[idx]

            if self._null_message(buff):
                d = {}
                d['sender_agent_id'] = int(buff[ParallelComm.META_INF_IDX_PROC_ID])
                d['msg_data'] = None
                ret.append(d)
            else:
                self.logger.info('send_recv_req: request received from agent {0}'.format(idx))
                d = {}
                d['sender_agent_id'] = int(buff[ParallelComm.META_INF_IDX_PROC_ID])
                d['msg_type'] = int(buff[ParallelComm.META_INF_IDX_MSG_TYPE])
                d['msg_data'] = int(buff[ParallelComm.META_INF_IDX_MSG_DATA])
                d['task_label'] = buff[ParallelComm.META_INF_IDX_TASK_SZ : ]
                ret.append(d)
        return ret

    # Multi-threaded handling of metadata send recv.
    def send_meta_response(self, meta_responses):
        '''
        Send response for each requester agent.
        '''
        if meta_responses:
            self.logger.info('send_meta_resp:')
            for meta_response in meta_responses:
                self._send_meta_response(meta_response)
    def _send_meta_response(self, req_dict):
        '''
        Sends either the mask or meta data to another agent.
        '''
        self.logger.info('send_meta_resp {0}'.format(req_dict))
        dst_agent_id = req_dict['dst_agent_id']

        # Get the mask, mask reward and embedding/tasklabel
        mask_reward = req_dict['mask_reward']
        distance = req_dict['dist']
        emb_label = req_dict['resp_task_label']


        if isinstance(emb_label, np.ndarray):
            emb_label = torch.tensor(emb_label, dtype=torch.float32)

        buff_list = [self.agent_id, ParallelComm.MSG_TYPE_SEND_RESP]
        # Consider changing to local buffer
        #buff = torch.ones_like(self.buff_send_task[dst_agent_id]) * torch.inf
        #buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        #buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

        # If mask is none then send back torch.inf
        if mask_reward is None:
            buff_list.append(ParallelComm.MSG_DATA_NULL)
            buff_list.append(torch.inf)
            buff_list.append(torch.inf)
            buff_list.append(torch.inf)
            buff_list.append(torch.inf)
            buff_list.append(torch.inf)
            
            #print('Mask reward is None', flush=True)
            #buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
            #print('Sending metadata buffer', buff, flush=True)
        # otherwise send mask
        else:
            buff_list.append(ParallelComm.MSG_DATA_META)
            buff_list.append(mask_reward)
            buff_list.append(distance)
            buff_list.append(emb_label)
            print(buff_list)
            #print('Mask reward is not None', flush=True)
            #buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_META
            #print(buff, flush=True)
            #buff[ParallelComm.META_INF_IDX_MSK_RW] = mask_reward
            #print(buff, flush=True)
            #print(distance, type(distance))
            #buff[ParallelComm.META_INF_IDX_DIST] = distance
            #print(buff, flush=True)
            #buff[ParallelComm.META_INF_IDX_TASK_SZ_ :] = emb_label
            #print('Sending metadata buffer', buff, flush=True)

        buff = torch.tensor(buff_list)
        print('Buffer to send', buff)

        #self.logger.info('Sending metadata to agent {0}: {1}'.format(dst_agent_id, buff, buff.dtype))
        # actual send
        req_send = dist.isend(tensor=buff, dst=dst_agent_id)
        req_send.wait()
        #self.handle_send_resp[dst_agent_id].wait()
        return
    def receive_meta_response(self, await_response):
        '''
        Receives the response from all in the network agents.
        '''
        ret = list()
        if any(await_response):
            for idx in range(self.num_agents):
                if idx == self.agent_id:
                    continue
                if self.handle_recv_resp[idx] is None:
                    self.logger.info('recv_meta_resp: set up handle to receive response from agent {0}'.format(idx))
                    req_recv = dist.irecv(tensor=self.buff_recv_task[idx], src=idx)
                    req_recv.wait()
                #self.handle_recv_resp[idx].wait()

            #time.sleep(ParallelComm.SLEEP_DURATION)
            print('Received buffer', self.buff_recv_task)
            # check whether message has been received
            for idx in range(self.num_agents):
                _buff = self.buff_recv_task[idx]

                if idx == self.agent_id:
                    d = {}
                    d['agent_id'] = self.agent_id
                    d['mask_reward'] = torch.inf
                    d['dist'] = torch.inf
                    d['emb_label'] = _buff[5:].detach().cpu().numpy()
                    ret.append(d)

                else:
                    if self._null_message(_buff):
                        d = {}
                        d['agent_id'] = int(_buff[0])
                        d['mask_reward'] = torch.inf
                        d['dist'] = torch.inf
                        d['emb_label'] = _buff[5:].detach().cpu().numpy()
                        ret.append(d)
                        self.logger.info('recv_resp: appending {0} response'.format(None))
                    
                    elif _buff[Communication.META_INF_IDX_MSG_DATA] == torch.inf:
                        ret.append(False)
                        self.logger.info('recv_resp: appending False response. All hope is lost')

                    elif _buff[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_META:
                        d = {}
                        d['agent_id'] = int(_buff[0])
                        d['mask_reward'] = float(_buff[3])
                        d['dist'] = float(_buff[4])
                        d['emb_label'] = _buff[5:].detach().cpu().numpy()
                        ret.append(d)
                        # Change this at some point to take the original data and not string
                        self.logger.info('recv_resp: appending metadata response from agent {0}'. format(d['agent_id']))

                # reset buffer and handle
                self.buff_recv_mask[idx][:] = torch.inf
                self.handle_recv_resp[idx] = None
        return ret

    def send_recv_meta(self, meta_responses, await_response):
        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        print('send_recv_meta()')

        _ = pool1.apply_async(self.send_meta_response, (meta_responses,))
        time.sleep(0.2)
        result = pool2.apply_async(self.receive_meta_response, (await_response,))

        return result.get()


    # Multi-threaded handling of mask request send recv
    def send_mask_request(self, send_msk_requests):

        if send_msk_requests:
            for agent_id, emb_label in send_msk_requests.items():
                if agent_id == self.agent_id:
                    continue

                # Convert embedding label to tensor
                if isinstance(emb_label, np.ndarray):
                    emb_label = torch.tensor(emb_label, dtype=torch.float32)

                # Initialise a buffer for one agent embedding/tasklabel
                data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf
                # Populate the tensor with necessary data
                data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
                data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_REQ
                
                if emb_label is None:
                    # If emb_label is none it means we reject the agent
                    data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
                        
                else:
                    # Otherwise we want the agent's mask
                    data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_TSK
                    data[ParallelComm.META_INF_IDX_TASK_SZ : ] = emb_label # NOTE deepcopy?

                # Send out the mask request or rejection to each agent that sent metadata
                req = dist.isend(tensor=data, dst=agent_id)
                req.wait()
    def receive_mask_requests(self, expecting):
        # Check for mask requests from other agents if expecting any requests
        ret = []
        if expecting:
            # If expecting a request for a mask, check for each expected agent id
            for idx in expecting:
                print('IDX in EXPECTING: ', idx, flush=True)
                data = torch.ones_like(self.buff_send_recv_req[idx]) * torch.inf
                req = dist.irecv(tensor=data, src=idx)
                req.wait()
                print('RECEIVING: ', data, idx, '\n', flush=True)
                #time.sleep(ParallelComm.SLEEP_DURATION)

                # If response was null message then this agent has been rejected.
                # If so, remove the idx from expecting and check the next id.
                # if no more idxs then return the list of dictionaries. (can be empty)
                if self._null_message(data):
                    expecting.remove(idx)

                # If not rejected then we need to send the mask to the requester
                else:
                    d = {}
                    d['requester_agent_id'] = int(data[ParallelComm.META_INF_IDX_PROC_ID])
                    d['msg_type'] = int(data[ParallelComm.META_INF_IDX_MSG_TYPE])
                    d['msg_data'] = int(data[ParallelComm.META_INF_IDX_MSG_DATA])
                    d['task_label'] = data[ParallelComm.META_INF_IDX_TASK_SZ : ]
                    ret.append(d)
                    expecting.remove(idx)

        # Return a list of dictionaries for each agent that wants a mask
        return expecting, ret
    
    def send_recv_mask_req(self, send_msk_requests, expecting):
        #print('SEND_MSK_REQ: ', send_msk_requests)

        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        _ = pool1.apply_async(self.send_mask_request, (send_msk_requests,))
        result = pool2.apply_async(self.receive_mask_requests, (expecting,))

        return result.get()

        '''
        thread1 = threading.Thread(target=self.send_mask_request, args=(send_msk_requests,))
        thread2 = threading.Thread(target=self.receive_mask_requests, args=(expecting,))

        if send_msk_requests:
            for agent_id, emb_label in send_msk_requests.items():
                new_group = dist.new_group([self.agent_id, agent_id], backend='gloo')
                print(agent_id, emb_label)
                
                # Convert embedding label to tensor
                if isinstance(emb_label, np.ndarray):
                    emb_label = torch.tensor(emb_label, dtype=torch.float32)

                # Initialise a buffer for one agent embedding/tasklabel
                data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf

                # Populate the tensor with necessary data
                data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
                data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_REQ
                
                if emb_label is None:
                    # If emb_label is none it means we reject the agent
                    data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
                        
                else:
                    # Otherwise we want the agent's mask
                    data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_TSK
                    data[ParallelComm.META_INF_IDX_TASK_SZ : ] = emb_label # NOTE deepcopy?

                print('SENDING: ', data, agent_id)
                # Send out the mask request or rejection to each agent that sent metadata
                req = dist.broadcast(tensor=data, src=self.agent_id, group=new_group, async_op=True)
                #req.wait()


        ret = []
        if expecting:
            for idx in expecting:
                new_group = dist.new_group([self.agent_id, idx], backend='gloo')
                print('IDX in EXPECTING: ', idx)
                data = torch.ones_like(self.buff_send_recv_req[idx]) * torch.inf
                req = dist.broadcast(tensor=data, src=idx, group=new_group, async_op=True)
                req.wait()
                print('RECEIVING: ', data, idx)

                # If response was null message then this agent has been rejected.
                # If so, remove the idx from expecting and check the next id.
                # if no more idxs then return the list of dictionaries. (can be empty)
                if self._null_message(data):
                    expecting.remove(idx)

                # If not rejected then we need to send the mask to the requester
                else:
                    d = {}
                    d['requester_agent_id'] = int(data[ParallelComm.META_INF_IDX_PROC_ID])
                    d['msg_type'] = int(data[ParallelComm.META_INF_IDX_MSG_TYPE])
                    d['msg_data'] = int(data[ParallelComm.META_INF_IDX_MSG_DATA])
                    d['task_label'] = data[ParallelComm.META_INF_IDX_TASK_SZ : ]
                    ret.append(d)
                    expecting.remove(idx)

        # Return a list of dictionaries for each agent that wants a mask
        return expecting, ret
        '''


    # Multi-threaded handling of mask send recv
    def send_mask(self, masks_list):
        print('Agent entered send_mask() actual')
        for item in masks_list:
            print(item)
            mask = item['mask']
            dst_agent_id = item['dst_agent_id']


            buff = torch.ones_like(self.buff_send_mask[dst_agent_id]) * torch.inf

            buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
            buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

            if mask is None:
                buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL

            else:
                buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
                buff[ParallelComm.META_INF_IDX_MASK_SZ : ] = mask # NOTE deepcopy?

            print('send_mask(): sending buffer', buff, flush=True)

            #print('Buffer to be sent with mask: ', buff, flush=True)
            # Send the mask to the destination agent id
            print('Starting send data')
            req = dist.isend(tensor=buff, dst=dst_agent_id)
            req.wait()
            print('send_mask() completed. Returning')
            #print('Sending mask to agents part 2. All complete!')
        return
    def receive_mask(self, best_agent_id):
        received_mask = None
        print('Agent entered receive_mask()')
        print(Fore.GREEN + 'Send Mask Best Agent ID: ', best_agent_id, flush=True)
        if best_agent_id:
            # We want to get the mask from the best agent
            buff = torch.ones_like(self.buff_recv_mask[self.agent_id]) * torch.inf
            print(buff, len(buff))
            # Receive the buffer containing the mask. Wait for 10 seconds to make sure mask is received
            print('Mask recv start')
            req = dist.irecv(tensor=buff, src=best_agent_id)
            req.wait()
            print('Mask recv end')
            #time.sleep(ParallelComm.SLEEP_DURATION)
            print('Mask after recv:', buff)
            # If the buffer was a null response (which it shouldn't be)
            # *meep*
            if self._null_message(buff):
                # Shouldn't reach this hopefully :^)
                return None

            # otherwise return the mask
            elif buff[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
                if buff[ParallelComm.META_INF_IDX_PROC_ID] == best_agent_id:
                    received_mask = copy.deepcopy(buff[ParallelComm.META_INF_IDX_MASK_SZ :])
                    #return buff[ParallelComm.META_INF_IDX_MASK_SZ : ]


            #received_mask = self.receive_mask_response(best_agent_id)
            print(Fore.GREEN + 'Received mask length: ', len(received_mask))

            # Reset the best agent id for the next request
            best_agent_id = None

        return received_mask, best_agent_id

    def send_recv_mask(self, masks_list, best_agent_id):
        #print('Send Recv Mask Function', len(mask), dst_agent_id, best_agent_id)
        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        if masks_list:
            print('send_mask():', masks_list)
            _ = pool1.apply_async(self.send_mask, (masks_list,))

        if best_agent_id:
            print('recv_mask():', best_agent_id)
            result = pool2.apply_async(self.receive_mask, (best_agent_id,))
            mask, best_agent_id = result.get()
            # Return the mask and best_agent_id
            return mask, best_agent_id

        # If the recv was not run then return NoneType for mask and whatever was passed for
        # best_agent_id (probably []).
        return None, best_agent_id

    '''
    # Defunct methods (Will likely delete in the future)
    def sync_gather_meta(self, agents):
        print('Agents: ', agents)
        new_group = dist.new_group(ranks=agents, backend='gloo')

        data = torch.ones_like(self.buff_send_recv_meta[0]) * torch.inf

        data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

        # If mask is none then send back torch.inf
        # otherwise send mask
        if mask_reward is None:
            data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL

        else:
            # if the mask is none but there is a mask reward, then overwrite the buffer with
            # the meta data. Otherwise don't do anything.
            data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_META
            data[ParallelComm.META_INF_IDX_MSK_RW] = mask_reward
            data[ParallelComm.META_INF_IDX_DIST] = distance
            data[ParallelComm.META_INF_IDX_TASK_SZ_ :] = emb_label

        req = dist.all_gather(tensor_list=self.buff_send_recv_meta, tensor=data, async_op=True)
        req.wait()

        # check buffer for incoming requests
        idxs = list(range(len(self.buff_send_recv_meta)))
        idxs.remove(self.agent_id)
        ret = []
        for idx in idxs :
            buff = self.buff_send_recv_meta[idx]

            if self._null_message(buff):
                ret.append(None)
            else:
                self.logger.info('send_recv_req: request received from agent {0}'.format(idx))
                d = {}
                d['agent_id'] = copy.deepcopy(msg[ParallelComm.META_INF_IDX_PROC_ID])
                d['mask_reward'] = copy.deepcopy(msg[ParallelComm.META_INF_IDX_MSK_RW])
                d['dist'] = copy.deepcopy(msg[ParallelComm.META_INF_IDX_DIST])
                d['emb_label'] = copy.deepcopy(msg[ParallelComm.META_INF_IDX_TASK_SZ_ :])
                ret.append(d)
        return ret

    def broadcast_mask(self, src, agents):
        new_group = dist.new_group(ranks=agents, backend='gloo')
        buff = torch.ones_like(self.buff_send_mask[0]) * torch.inf

        buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

        if mask is None:
            buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
            buff[ParallelComm.META_INF_SZ : ] = torch.inf

        else:
            buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
            buff[ParallelComm.META_INF_SZ : ] = mask # NOTE deepcopy?

        # Send the mask to the destination agent id
        req = dist.broadcast(tensor=buff, src=src, group=new_group, async_op=True)
        req.wait()

        return buff

    def broadcast_label(self, label):
        #req = dist.broadcast(tensor=label, src=0, )

        #return label
        pass

    def gather_masks(self, mask):
        #gather_list = [torch.ones(2 + mask_sz, ) * torch.inf for _ in range(num_agents)]

        #print(gather_list)
        #tensor = gather_list[0]
        #print(tensor)

        #req = dist.gather(tensor=tensor, gather_list=gather_list, dst=0, group=None, async_op=True)
        #req.wait()

        #if self.agent_id == 0:
        #    ret = []
        #    for idx in len(gather_list):
        #        gather_list[idx]
        #    return gather_list
        pass

    def barrier(self):
        dist.barrier()

    def fetch_all(self):
        pass
    '''
   
   
    ### Core functions for communication module. Main loop and parallelisation.
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop):

        # Initialise the process group for torch distributed
        proc_check = self.init_dist()

        #queue_mask.put(proc_check)


        msg = None
        # Store the best agent id for quick reference
        best_agent_id = None
        # Store list of agent IDs that this agent has sent metadata to
        #expecting = list()

        # initial state of input variables to loop
        comm_iter = 0
        while True:
            expecting = []
            # Get the latest states of these variables
            track_tasks, mask_rewards_dict, await_response = queue_loop.get()
            print()
            print()
            print(Fore.GREEN + 'COMMUNICATION ITERATION: ', comm_iter)
            print(Fore.GREEN + '', track_tasks, mask_rewards_dict, await_response)
            
            # Try getting the label request, otherwise it will be NoneType
            try:
                # Initially msg will be populated with the first task label then set to None after the 
                # first completed communication loop. After that the msg will be None until a new query
                # is requested from the agent.
                msg = queue_label.get_nowait()
                print(Fore.GREEN + 'Comm Module msg this iter: ', msg)
            except Empty:
                print(Fore.GREEN + 'FAILED TO GET LABEL FROM AGENT')
                continue
            
            
            #if self.mode == 'ondemand':
            #######################   COMMUNICATION STEP ONE    #######################
            ####################### REQUESTS BETWEEN ALL AGENTS #######################
            # send out broadcast request to all other agents for task label
            #print(Fore.GREEN + 'Doing request')
            other_agents_request = self.send_receive_request(msg)
            print()
            print(Fore.GREEN + 'Other agent requests: ', other_agents_request)
            print()



            #######################   COMMUNICATION STEP TWO    #######################
            ####################### SEND AND RECV METADATA #######################
            # Respond to received queries with metadata.
            # Meta data contains the reward for the similar task, distance and the similar 
            # tasklabel/embedding.
            meta_responses = []
            if other_agents_request:
                for req in other_agents_request:
                    if req['msg_data'] is None: continue
                    # Update the track tasks with the latest information
                    track_tasks[req['sender_agent_id']] = req['task_label']

                    # Compute the embedding distance. Maybe there is a better way to achieve this
                    req_label_as_np = req['task_label'].detach().cpu().numpy()
                    print(Fore.GREEN + 'Requested label from agent {0}: {1}'.format(req['sender_agent_id'], req_label_as_np))
                    print(Fore.GREEN + 'Current knowledge base for this agent: ', mask_rewards_dict)

                    # For each embedding/tasklabel reward pair, calculate the distance to the
                    # requested embedding/tasklabel.
                    # If the distance is below the THRESHOLD then send it back
                    # otherwise send nothing back.

                    # Iterate through the knowledge base and compute the distances
                    dist_list = []
                    _d= {}
                    for tlabel, treward in mask_rewards_dict.items():
                        if treward != 0.0:
                            print(np.asarray(tlabel), treward)
                            dist = np.sum(abs(np.subtract(req_label_as_np, np.asarray(tlabel))))
                            _dist = {}
                            _dist['label'] = tlabel
                            _dist['dist'] = dist
                            dist_list.append(_dist)

                    print(dist_list)

                    if dist_list:
                        # sort the dictionary by distance from lowest to highest
                        dist_list = sorted(dist_list, key=lambda d: (d['dist']))
                        print(dist_list)

                        # iterate through the knowledge
                        for kdict in dist_list:
                            tlabel = kdict['label'] # tuple
                            tdist = kdict['dist']   # float
                            treward = mask_rewards_dict[tlabel] # float

                            print(tlabel, tdist, treward)

                            if tdist <= ParallelComm.THRESHOLD:
                                _d['dst_agent_id'] = req['sender_agent_id']
                                _d['mask_reward'] = treward
                                _d['dist'] = tdist
                                _d['resp_task_label'] = np.asarray(tlabel)

                                meta_responses.append(_d)
                                expecting.append(_d['dst_agent_id'])

                    if not _d:
                        _d['dst_agent_id'] = req['sender_agent_id']
                        _d['mask_reward'] = None
                        _d['dist'] = None
                        _d['resp_task_label'] = None

                        meta_responses.append(_d)
                        expecting.append(_d['dst_agent_id'])

            print()
            print('Meta responses to send to other agents:', meta_responses, len(meta_responses))
            print(Fore.GREEN + 'Expecting mask request from these agents:', expecting)
            print(Fore.GREEN + 'Awaiting Responses? ', await_response)
            print()
            
            results = self.send_recv_meta(meta_responses, await_response)
            print('Metadata from all agents: ', results)
            print()

            """
            send_msk_request = dict()
            # if not results something bad has happened
            if results:
                # Sort received meta data by smallest distance (primary) and highest reward (secondary),
                # using full bidirectional multikey sorting (fancy words for such a simple concept)
                results = sorted(results, key=lambda d: (d['dist'], -d['mask_reward']))
                print(Fore.GREEN + 'Metadata sorted: ', results)
                print()

                # Select the best agent
                selected = False
                for idx in range(len(await_response)):
                    # Do some checks to remove to useless results
                    #if idx == self.agent_id: continue
                    if await_response[idx] is False: continue
                    if results[idx] is False: continue
                    elif results[idx] is None: await_response[idx] = False

                    # Otherwise unpack the metadata
                    else:
                        recv_agent_id = results[idx]['agent_id']
                        if recv_agent_id == self.agent_id:
                            await_response[idx] = False
                            continue
                        recv_msk_rw = results[idx]['mask_reward']
                        recv_label = results[idx]['emb_label']
                        recv_dist = results[idx]['dist']

                        # If the received distance is below the THRESHOLD and the agent
                        # hasn't selected a best agent yet then select this response
                        if recv_dist <= ParallelComm.THRESHOLD and recv_dist != torch.inf and selected == False:
                            # Add the agent id and embedding/tasklabel from the agent
                            # to a dictionary to send requests/rejections out to.
                            '''if recv_agent_id == self.agent_id:
                                _temp_send_msk_request = dict()
                                _temp_send_msk_request[self.agent_id] = recv_label
                                _temp_send_msk_request.update(send_msk_request)
                                send_msk_request = _temp_send_msk_request

                                del _temp_send_msk_request

                            else:
                                send_msk_request[recv_agent_id] = recv_label'''
                            send_msk_request[recv_agent_id] = recv_label

                            # Make a note of the best agent id in memory of this agent
                            # We will use this later to check the response from the best agent
                            best_agent_id = recv_agent_id

                            # Make the selected flag true so we don't pick anymore to send
                            selected = True

                        # If best selected or doesn't meet criteria, then send rejection
                        # i.e., None
                        else:
                            '''if recv_agent_id == self.agent_id:
                                _temp_send_msk_request = dict()
                                _temp_send_msk_request[self.agent_id] = None
                                _temp_send_msk_request.update(send_msk_request)
                                send_msk_request = _temp_send_msk_request

                                del _temp_send_msk_request

                            else:
                                send_msk_request[recv_agent_id] = None''' 
                            send_msk_request[recv_agent_id] = None

                        # We have checked the response so set it to False until the next task change
                        # and request loop begins
                        await_response[idx] = False

                    # Need to fix this logging message
                    self.logger.info('Meta data received from agent {0}'.format(idx))
            print(Fore.GREEN + 'Mask requests to send: ', send_msk_request)
            print(await_response)

            
            #######################     COMMUNICATION STEP FOUR      #######################
            ####################### SEND MASK REQUESTS OR REJECTIONS #######################
            msk_requests = []
            msk_requests = self.send_recv_mask_req(send_msk_request, expecting)
            print(Fore.GREEN + 'After receiving mask req: ', msk_requests)



            
            ####################### COMMUNICATION STEP FIVE #######################
            # Now the agent needs to send a mask to each agent in the msk_requests list
            # if it is not empty.
            '''if msk_requests:
                # Iterate through the requests
                for req in msk_requests:
                    # Send a mask to the requesting agent
                    agent_id = req['requester_agent_id']

                    # Send label:id to agent
                    _temp_label = req['task_label']
                    queue_label_send.put((_temp_label, agent_id))
                    print('Sending mask to agents part 1')

            try:
                mask, agent_id = queue_mask_recv.get_nowait()
                self.send_mask_response((agent_id, mask))
                print('Sending mask to agents part 2. All complete!')

            except Empty:
                pass

            # Expecting a response from the best agent id
            received_mask = None

            print(Fore.GREEN + 'Best Agent: ', best_agent_id)
            if best_agent_id:
                # We want to get the mask from the best agent
                received_mask = self.receive_mask_response(best_agent_id)
                print(Fore.GREEN + 'Received mask length: ', len(received_mask))

                # Reset the best agent id for the next request
                best_agent_id = None'''


            received_mask = None

            masks_list = []
            if msk_requests:
                print('Agent has entered send_mask()')
                # Iterate through the requests
                # Send the label to be converted, to the agent
                for req in msk_requests:
                    d = {}
                    print('Mask Requests: ', req, flush=True)
                    # Send a mask to the requesting agent
                    dst_agent_id = req['requester_agent_id']
                    # Send label:id to agent
                    _temp_label = req['task_label']
                    queue_label_send.put_nowait((_temp_label, dst_agent_id))

                    print('Send label to be converted:', _temp_label, dst_agent_id, flush=True)
                    print('Sending mask to agents part 1', flush=True)

                    # wait to receive a mask from the agent module. do not continue until you receive
                    # this mask. agent will see the request eventually and send back the converted mask.
                    mask, dst_agent_id = queue_mask_recv.get()
                    print('MASK RECEIVED FROM AGENT MODULE:', len(mask), dst_agent_id)

                    d['mask'] = mask
                    d['dst_agent_id'] = dst_agent_id
                    masks_list.append(d)
                    
            print('Checking before send_recv_mask():', masks_list, best_agent_id)
            received_mask, best_agent_id = self.send_recv_mask(masks_list, best_agent_id)


            print(Fore.GREEN + 'Returning to agent', received_mask, best_agent_id, flush=True)
            # Send the mask to the agent (mask or NoneType) as well as the other variables 
            # used in the rest of the agent iteration. The communication loop will get
            # these variables back in the next cycle after being updated by the agent.
            queue_mask.put_nowait((received_mask, track_tasks, await_response))
            print(Fore.GREEN + 'Return completed', flush=True)
            
            #elif self.mode == 'fetchall':
            #    raise ValueError('{0} communication mode has not been implemented!'.format(mode))
            #else:
            #    raise ValueError('{0} communication mode has not been implemented!'.format(mode))
            """
            comm_iter += 1

    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop):
        p = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop))
        p.start()
        return p


'''
from ..deep_rl.utils.logger import Logger, get_logger
from ..deep_rl.utils.misc import get_default_log_dir

if __name__ == '__main__':
    agent_id = 0
    num_agents = 2
    emb_label_sz = 3
    mask_sz = 11000

    exp_id = 'upz'
    exp_id = '{0}-seed-{1}'.format(exp_id, 9157)
    path_name = '{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, agent_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    logger = get_logger()

    init_address = '127.0.0.1'
    init_port = 5000
    mode = 'ondemand'
    comm1 = ParallelComm(agent_id, num_agents, emb_label_sz, mask_sz, logger, init_address, init_port, mode)
    
    agent_id = 1
    comm2 = ParallelComm(agent_id, num_agents, emb_label_sz, mask_sz, logger, init_address, init_port, mode)
'''
