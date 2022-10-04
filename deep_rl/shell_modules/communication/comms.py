# -*- coding: utf-8 -*-
import os
import copy
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist
import multiprocess as mp
#import ray


class Communication(object):
    META_INF_IDX_PROC_ID = 0
    META_INF_IDX_MSG_TYPE = 1
    META_INF_IDX_MSG_DATA = 2

    META_INF_IDX_MSK_RW = 3
    META_INF_IDX_TASK_SZ = 3 # only for the send_recv_request buffer

    META_INF_IDX_DIST = 4
    META_INF_IDX_TASK_SZ_ = 5
    

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
    SLEEP_DURATION = 10

    # Task label size can be replaced with the embedding size.
    def __init__(self, agent_id, num_agents, emb_label_sz, mask_sz, logger, init_address, init_port):
        super(Communication, self).__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.emb_label_sz = emb_label_sz
        self.mask_sz = mask_sz
        self.logger = logger
        

        if init_address in ['127.0.0.1', 'localhost']:
            os.environ['MASTER_ADDR'] = init_address
            os.environ['MASTER_PORT'] = init_port
            self.comm_init_str = 'env://'
        else:
            self.comm_init_str = 'tcp://{0}:{1}'.format(init_address, init_port)

        self.handle_send_recv_req = None
        self.handle_recv_resp = [None, ] * num_agents
        self.handle_send_resp = [None, ] * num_agents

        self.buff_send_recv_req = [torch.ones(Communication.META_INF_TASK_SZ + emb_label_sz, ) \
            * torch.inf for _ in range(num_agents)]

        self.buff_recv_resp = [torch.ones(Communication.META_INF_IDX_MASK_SZ + mask_sz, ) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_resp = [torch.ones(Communication.META_INF_IDX_MASK_SZ + mask_sz, ) * torch.inf \
            for _ in range(num_agents)]

        self.buff_send_recv_req_msk = [torch.ones(Communication.META_INF_IDX_TASK_SZ)]

    def init_dist(self):
        '''
        Initialise the process group for torch.
        '''
        self.logger.info('*****agent {0} / initialising transfer (communication) module'.format(self.agent_id))
        dist.init_process_group(backend='gloo', init_method=self.comm_init_str, rank=self.agent_id, \
            world_size=self.num_agents, timeout=datetime.timedelta(seconds=30))

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
        data[Communication.META_INF_IDX_PROC_ID] = self.agent_id
        data[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_REQ

        if emb_label is None:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL
        else:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_TSK
            data[Communication.META_INF_IDX_TASK_SZ : ] = emb_label # NOTE deepcopy?


        # actual send/receive
        self.handle_send_recv_req = dist.all_gather(tensor_list=self.buff_send_recv_req, \
            tensor=data, async_op=True)

        #self.handle_send_recv_req.wait()


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
                d['task_label'] = buff[Communication.META_INF_IDX_TASK_SZ : ]
                ret.append(d)
        return ret

    def send_meta_response(self, requesters):
        '''
        Send response for each requester agent.
        '''
        self.logger.info('send_resp:')
        for requester in requesters:
            self._send_meta_response(requester)

    def _send_meta_response(self, req_dict):
        '''
        Sends either the mask or meta data to another agent.
        '''
        requester_agent_id = req_dict['sender_agent_id']

        # Get the mask, mask reward and embedding/tasklabel
        mask_reward = req_dict['mask_reward']
        emb_label = req_dict['task_label']
        distance = req_dict['distance']


        buff = self.buff_send_resp[requester_agent_id]
        buff[Communication.META_INF_IDX_PROC_ID] = self.agent_id
        buff[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_RESP

        self.logger.info('send_resp: responding to agent {0} query'.format(requester_agent_id))
        self.logger.info('send_resp: mask (response) data type: {0}'.format(type(mask)))

        # If mask is none then send back torch.inf
        # otherwise send mask
        if mask_reward is None:
            buff[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL

        else:
            # if the mask is none but there is a mask reward, then overwrite the buffer with
            # the meta data. Otherwise don't do anything.
            buff[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_META
            buff[Communication.META_INF_IDX_MSK_RW] = mask_reward
            buff[Communication.META_INF_IDX_DIST] = distance
            buff[Communication.META_INF_IDX_TASK_SZ_ :] = emb_label

        # actual send
        self.handle_send_resp[requester_agent_id] = dist.isend(tensor=buff, dst=requester_agent_id)
        return

    def receive_meta_response(self):
        '''
        Receives the response from all in the network agents.
        '''
        self.logger.info('recv_resp:')
        for idx in range(self.num_agents):
            if idx == self.agent_id:
                continue
            if self.handle_recv_resp[idx] is None:
                self.logger.info('recv_resp: set up handle to receive response from agent {0}'.format(idx))
                self.handle_recv_resp[idx] = dist.irecv(tensor=self.buff_recv_resp[idx], src=idx)

        time.sleep(Communication.SLEEP_DURATION)

        # check whether message has been received
        results = list()
        ret = dict()
        for idx in range(self.num_agents):
            if idx == self.agent_id:
                results.append(None)
                continue

            msg = self.buff_recv_resp[idx]
            if self._null_message(msg):
                results.append(None)
                self.logger.info('recv_resp: appending {0} response'.format(None))

            elif msg[Communication.META_INF_IDX_MSG_DATA] == MSG_DATA_META:
                ret['agent_id'] = copy.deepcopy(msg[Communication.META_INF_IDX_PROC_ID])
                ret['mask_reward'] = copy.deepcopy(msg[Communication.META_INF_IDX_MSK_RW])
                ret['dist'] = copy.deepcopy(msg[Communication.META_INF_IDX_DIST])
                ret['emb_label'] = copy.deepcopy(msg[Communication.META_INF_IDX_TASK_SZ_ :])
                results.append(ret)

                # Change this at some point to take the original data and not string
                self.logger.info('recv_resp: appending {0} response'. format('META DATA'))

            else:
                pass



            # code for receive response for the mask    
            #elif msg[Communication.META_INF_IDX_MSG_DATA] == MSG_DATA_MSK:
            #    ret['agent_id'] = copy.deepcopy(msg[Communication.META_INF_IDX_PROC_ID])
            #    ret['mask'] = copy.deepcopy(msg[Communication.META_INF_IDX_MSK_SZ : ])
            #    results.append(ret)

                # Change this at some point to take the original data and not string
            #    self.logger.info('recv_resp: appending {0} response'.format('MASK'))

            # reset buffer and handle
            self.buff_recv_resp[idx][:] = torch.inf
            self.handle_recv_resp[idx] = None 
        return results

    def barrier(self):
        dist.barrier()

    def send_receive_mask_request(self, topthree):
        # TODO: Merge this function with the other send_receive_mask function.
        '''
        Sends a request to the top three agents for masks using the their embeddings.
        Checks for similar requests.
        '''
        topthreeids = list(topthree.keys())
        topthreeids.append(self.agent_id)

        # Get the labels for the top three agents
        topthreelabels = list(topthree.values())

        agents = [torch.ones(self.num_agents) * torch.inf]

        # make new sub process group for this agent and the top one ~ three agents
        group = dist.new_group(topthreeids)

        gather_tensor = []

        for idx in enumerate(topthreeids):
            if idx == self.agent_id
            agent_id = topthreeids[idx]
            emb_label = topthreelabels[idx]

            if isinstance(emb_label, np.ndarray):
                emb_label = torch.tensor(emb_label, dtype=torch.float32)

            data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf

            data = self.buff_send_request[agent_id]
            data[Communication.META_INF_IDX_PROC_ID] = self.agent_id
            data[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_REQ
            
            if emb_label is None:
                # Should never reach this point in the code. If it does then reality has broken.
                data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL
                 
            else:
                data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_TSK
                data[Communication.META_INF_TASK_SZ : ] = emb_label # NOTE deepcopy?


            if 
            self.handle_send_recv_req = dist.isend(tensor=data, gather_list=gather_tensor dst=self.agent_id, group=group, async_op=True)


        time.sleep(Communication.SLEEP_DURATION)

        check buffer for incoming requests
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
                d['task_label'] = buff[Communication.META_INF_IDX_TASK_SZ : ]
                ret.append(d)
        return ret


class CommProcess(mp.Process):
    SR_REQUEST = 0
    S_REPSPONSE = 1
    R_RESPONSE = 2
    BROADCAST = 3
    BARRIER = 4
    INIT = 5

    def __init__(self, pipe, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.comm = Communication(agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port)

    def run(self):
        while True:
            op, data = self.pipe.recv()
            if op == self.SR_REQUEST:
                self.pipe.send(self.comm.send_receive_request(data))

            elif op == self.S_REPSPONSE:
                self.comm.send_response(data)

            elif op == self.R_RESPONSE:
                self.pipe.send(self.comm.receieve_response())

            elif op == self.BROADCAST:
                self.comm.broadcast_existence()

            elif op == self.BARRIER:
                self.comm.barrier()

            elif op == self.INIT:
                self.comm.init_proc_grp()

            else:
                raise Exception('Unknown command')
                
class ParallelizedComm:
    def __init__(self, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port):
        self.pipe, worker_pipe = mp.Pipe([True])
        self.worker = CommProcess(worker_pipe, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port)
        self.worker.start()

    def send_receive_request(self, data):
        self.pipe.send([CommProcess.SR_REQUEST, data])
        return self.pipe.recv()

    def send_response(self, data):
        self.pipe.send([CommProcess.S_REPSPONSE, data])

    def receieve_response(self):
        self.pipe.send([CommProcess.R_RESPONSE, None])
        return self.pipe.recv()

    def broadcast_existence(self):
        self.pipe.send([CommProcess.BROADCAST, None])

    def barrier(self):
        self.pipe.send([CommProcess.BARRIER, None])

    def init_proc_grp(self):
        self.pipe.send([CommProcess.INIT, None])