# -*- coding: utf-8 -*-
'''
_________                                           .__                  __   .__                 
\_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
/    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
\     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
 \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
        \/               \/       \/             \/          \/      \/                        \/ 

'''
import os
import copy
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist
import multiprocess as mp
from queue import Empty



# Add back the old comm as well


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
    

    META_INF_IDX_MASK_SZ = 3
    
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

        self.buff_recv_meta = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_meta = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]

        self.buff_recv_mask = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf 
        self.buff_send_mask = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf
            
    def init_dist(self):
        '''
        Initialise the process group for torch. Return boolean of is processes group initialised.
        '''
        self.logger.info('*****agent {0} / initialising transfer (communication) module'.format(self.agent_id))
        dist.init_process_group(backend='gloo', init_method=self.comm_init_str, rank=self.agent_id, \
            world_size=self.num_agents, timeout=datetime.timedelta(seconds=30))

        return dist.is_initialized()

    def _null_message(self, msg):
        # check whether message sent denotes or is none.
        if bool(msg[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_NULL):
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
        print(self.buff_send_recv_req, flush=True)


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
    def send_meta_response(self, requesters):
        '''
        Send response for each requester agent.
        '''
        if requesters:
            #self.logger.info('send_resp:')
            for requester in requesters:
                self._send_meta_response(requester)
    def _send_meta_response(self, req_dict):
        '''
        Sends either the mask or meta data to another agent.
        '''
        self.logger.info('send_resp {0}'.format(req_dict))
        
        dst_agent_id = req_dict['dst_agent_id']
        mask_reward =req_dict['mask_reward']
        distance = req_dict['dist']
        distance = np.float64(distance)
        emb_label = req_dict['resp_task_label']

        #print(dst_agent_id, mask_reward, distance, emb_label)
        #print(type(dst_agent_id), type(mask_reward), type(distance), type(emb_label))

        #print(dst_agent_id, mask_reward, emb_label, flush=True)
        #print(type(dst_agent_id), type(mask_reward), type(emb_label), flush=True)

        #if isinstance(emb_label, np.ndarray):
        #    emb_label = torch.tensor(emb_label, dtype=torch.float32)

        # Consider changing to local buffer
        buff = torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + self.emb_label_sz, dtype=torch.float32) * torch.inf
        buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

        #self.logger.info('send_resp: responding to agent {0} query'.format(dst_agent_id))
        #self.logger.info('send_resp: mask (response) data type: {0}'.format(type(mask)))

        # If mask is none then send back torch.inf
        # otherwise send mask
        if mask_reward is torch.inf:
            #print('Mask reward is None -> True', flush=True)
            buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL

        else:
            #print('Mask reward is None -> False', flush=True)
            #print(mask_reward, emb_label, distance, flush=True)
            #print(type(mask_reward), type(emb_label), type(distance), flush=True)
            # if the mask is none but there is a mask reward, then overwrite the buffer with
            # the meta data. Otherwise don't do anything.
            buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_META
            buff[ParallelComm.META_INF_IDX_MSK_RW] = mask_reward
            buff[ParallelComm.META_INF_IDX_DIST] = distance
            buff[ParallelComm.META_INF_IDX_TASK_SZ_ :] = emb_label

        #self.logger.info('Sending metadata to agent {0}: {1}'.format(dst_agent_id, buff, buff.dtype))
        # actual send
        #print(buff)
        #for idx, val in enumerate(buff):
            #print(type(val), val)
        #print(type(buff))
        #print(type(buff[0]), type(buff[1]), type(buff[2]), type(buff[3]), type(buff[4]), type(buff[5]))
        #print('STARTING SEND')
        req_send = dist.isend(tensor=buff, dst=dst_agent_id)
        req_send.wait()
        #print('COMPLETED SEND', flush=True)
    def receive_meta_response(self, await_response):
        '''
        Receives the response from all in the network agents.
        '''
        #print(await_response)
        ret = []
        _buff_list = []
        if any(await_response):
            #self.logger.info('recv_resp:')
            for idx in range(self.num_agents):
                #print(idx)
                _buff_recv = torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + self.emb_label_sz, dtype=torch.float32) * torch.inf
                if idx == self.agent_id:
                    _buff_list.append(_buff_recv)

                else:
                    req_recv = dist.irecv(tensor=_buff_recv, src=idx)
                    req_recv.wait()
                    _buff_list.append(_buff_recv)
                    #print('COMPLETED RECEPTION from agent {0}'.format(idx), flush=True)

            #time.sleep(ParallelComm.SLEEP_DURATION)

            #print(_buff_list)
            # check whether message has been received
            for idx in range(self.num_agents):
                _buff = _buff_list[idx]

                if idx == self.agent_id:
                    #print('recv buff self agent id')
                    d = {}
                    d['agent_id'] = self.agent_id
                    d['mask_reward'] = torch.inf
                    d['dist'] = torch.inf
                    d['emb_label'] = _buff[5:].detach().cpu().numpy()
                    ret.append(d)

                else:
                    if self._null_message(_buff):
                        #print('recv buff is null')
                        d = {}
                        d['agent_id'] = int(_buff[0])
                        d['mask_reward'] = torch.inf
                        d['dist'] = torch.inf
                        d['emb_label'] = _buff[5:].detach().cpu().numpy()
                        ret.append(d)
                        #self.logger.info('recv_resp: appending {0} response'.format(None))
                    
                    elif _buff[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
                        #print('recv buff failed')
                        ret.append(False)
                        #self.logger.info('recv_resp: appending False response. All hope is lost')

                    elif _buff[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_META:
                        #print('recv buff success')
                        d = {}
                        d['agent_id'] = int(_buff[0])
                        d['mask_reward'] = float(_buff[3])
                        d['dist'] = float(_buff[4])
                        d['emb_label'] = _buff[5:].detach().cpu().numpy()
                        ret.append(d)
        return ret

    def send_recv_meta(self, requesters, await_response):
        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        #self.logger.info('send_recv_meta():')

        _ = pool1.apply_async(self.send_meta_response, (requesters,))
        #self.send_meta_response(requesters)
        #time.sleep(0.2)
        result = pool2.apply_async(self.receive_meta_response, (await_response,))
        #result = self.receive_meta_response(await_response)
        #return result

        pool1.close()
        pool2.close()
        return result.get()


    # Multi-threaded handling of mask request send recv
    def send_mask_request(self, send_msk_requests):
        # TODO: Merge this function with the other send_receive_mask function.
        '''
        Sends a request to the top three agents for masks using the their embeddings.
        Checks for similar requests.
        '''
        print('SEND_MSK_REQ: ', send_msk_requests, flush=True)

        if send_msk_requests:
            for agent_id, emb_label in send_msk_requests.items():
                print(agent_id, emb_label, flush=True)
                
                
                #if agent_id == self.agent_id:
                #    print('continue')
                #    continue

                #print('sending', flush=True)
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
                print('SENDING: ', data, agent_id, '\n', flush=True)
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
        return ret
    
    def send_recv_mask_req(self, send_msk_requests, expecting):
        #print('SEND_MSK_REQ: ', send_msk_requests)

        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        #self.send_mask_request(send_msk_requests)
        _ = pool1.apply_async(self.send_mask_request, (send_msk_requests,))
        result = pool2.apply_async(self.receive_mask_requests, (expecting,))

        pool1.close()
        pool2.close()
        #return None, None
        return result.get()


    # Multi-threaded handling of mask send recv
    def send_mask(self, masks_list):
        if masks_list:
            for mask_dict in masks_list:
                self._send_mask(mask_dict)

    def _send_mask(self, mask_dict):
            print('Agent entered send_mask()', flush=True)
            print(mask_dict, flush=True)
            mask = mask_dict['mask']
            dst_agent_id = int(mask_dict['dst_agent_id'])
            print(type(mask), mask.dtype, flush=True)

            buff = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz, dtype=torch.float32) * torch.inf

            buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
            buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

            if mask is None:
                buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
                buff[ParallelComm.META_INF_IDX_MASK_SZ : ] = torch.inf

            else:
                buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
                buff[ParallelComm.META_INF_IDX_MASK_SZ : ] = mask # NOTE deepcopy?

            print('send_mask(): sending buffer', buff, flush=True)
            req_send = dist.isend(tensor=buff, dst=dst_agent_id)
            #req_send.wait()
            print('send_mask() completed. Returning', flush=True)
    def receive_mask(self, best_agent_id):
        received_mask = None
        
        print('Agent entered receive_mask()', flush=True)
        if best_agent_id:
            # We want to get the mask from the best agent
            _buff = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz, dtype=torch.float32) * torch.inf
            print(_buff, len(_buff))
            # Receive the buffer containing the mask. Wait for 10 seconds to make sure mask is received
            print('Mask recv start', flush=True)

            
            req_recv = dist.irecv(tensor=_buff, src=best_agent_id)
            #time.sleep(ParallelComm.SLEEP_DURATION)
            req_recv.wait()
            print('Mask recv end', flush=True)
            #time.sleep(ParallelComm.SLEEP_DURATION)

            # otherwise return the mask
            if _buff[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
                pass

            elif _buff[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
                print('MASK DATA RECEIVED', flush=True)
                if _buff[ParallelComm.META_INF_IDX_PROC_ID] == best_agent_id:
                    print('SENDER IS BEST AGENT', flush=True)
                    received_mask = _buff[ParallelComm.META_INF_IDX_MASK_SZ : ]

            # Reset the best agent id for the next request
            best_agent_id = None

        return received_mask, best_agent_id

    def send_recv_mask(self, masks_list, best_agent_id):
        #print('Send Recv Mask Function', len(mask), dst_agent_id, best_agent_id)
        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        #self.send_mask(masks_list)
        #received_mask, best_agent_id = self.receive_mask(best_agent_id)

        #return received_mask, best_agent_id
        #print('send_mask():', masks_list)
        _ = pool1.apply_async(self.send_mask, (masks_list,))
        result = pool2.apply_async(self.receive_mask, (best_agent_id,))

        pool1.close()
        pool2.close()

        # If the recv was not run then return NoneType for mask and whatever was passed for
        # best_agent_id (probably []).
        return result.get()

    '''
    def send_recv_mask_all_gather(self, masks_list, best_agent_id):
        
        if masks_list:
            for idx in range(len(masks_list) + 1):
                msk_buff_list = [torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz, dtype=torch.float32) * torch.inf for _ in range(2)]
                
                try:
                    # If masks_list then send out a mask
                    mask = masks_list[idx]['mask']
                    dst_agent_id = int(masks_list[idx]['dst_agent_id'])
                    new_group = dist.new_group([self.agent_id, dst_agent_id])


                    # Setup buffer to send

                    send_buff = msk_buff_list[0]
                    send_buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
                    send_buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

                    if mask is None:
                        send_buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL

                    else:
                        send_buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
                        send_buff[ParallelComm.META_INF_IDX_MASK_SZ : ] = mask # NOTE deepcopy?


                    mask_handler = dist.all_gather(tensor_list=msk_buff_list, tensor=send_buff, group=new_group, async_op=True)
                    mask_handler.wait()

                except IndexError:
                    # Otherwise receive a mask
                    new_group = dist.new_group([self.agent_id, best_agent_id])
                    mask_handler = dist.all_gather(tensor_list=msk_buff_list, tensor=null_buff, group=new_group, async_op=True)

    '''

    ### Core functions
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop):

        # Initialise the process group for torch distributed
        proc_check = self.init_dist()

        queue_mask.put(proc_check)


        msg = None
        # Store the best agent id for quick reference
        best_agent_id = None
        best_agent_rw = {}

        # initial state of input variables to loop
        comm_iter = 0
        while True:
            time.sleep(1)
            expecting = list() # Come back to once all components are checked. Might cause issues

            # Get the latest states of these variables
            track_tasks, mask_rewards_dict, await_response = queue_loop.get()
            print()
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
                print(Fore.GREEN + 'FAILED')
                continue
            
            
            #if self.mode == 'ondemand':
            #######################   COMMUNICATION STEP ONE    #######################
            ####################### REQUESTS BETWEEN ALL AGENTS #######################
            # send out broadcast request to all other agents for task label
            #print(Fore.GREEN + 'Doing request')
            start_time = time.time()
            dist.barrier()
            other_agents_request = self.send_receive_request(msg)
            print('******** TIME TAKEN FOR SEND_RECV_REQ():', time.time()-start_time)
            print()
            print(Fore.GREEN + 'Other agent requests: ', other_agents_request)



            requests = []
            for req in other_agents_request:
                if req is None: continue
                req['mask'] = agent.label_to_mask(req['task_label'])
                requests.append(req)
            if len(requests) > 0:
                self.send_response(requests)

            # if the agent earlier sent a request, check whether response has been sent.
            if any(await_response):
                logger.info('awaiting response: {0}'.format(await_response))
                masks = []
                received_masks = comm.receive_response()
                for i in range(len(await_response)):
                    if await_response[i] is False: continue

                    if received_masks[i] is False: continue
                    elif received_masks[i] is None: await_response[i] = False
                    else:
                        masks.append(received_masks[i])
                        await_response[i] = False
                logger.info('number of task knowledge received: {0}'.format(len(masks)))
                # TODO still need to fix how mask is used.
                agent.distil_task_knowledge(masks)

            
                    

            received_mask = None

            queue_mask.put_nowait((received_mask, track_tasks, await_response, best_agent_rw))
            
            
            comm_iter += 1
            dist.barrier()

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