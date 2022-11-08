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
    META_INF_IDX_TASK_SZ = 3
    META_INF_IDX_TASK_SZ_ = 4
    META_INF_IDX_MASK_SZ = 7
    
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

        self.buff_recv_mask = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf 
        self.buff_send_mask = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf
            
    def init_dist(self):
        '''
        Initialise the process group for torch. Return boolean of is processes group initialised.
        '''
        self.logger.info('*****agent {0} / initialising transfer (communication) module'.format(self.agent_id))
        dist.init_process_group(backend='gloo', init_method=self.comm_init_str, rank=self.agent_id, \
            world_size=self.num_agents) #timeout=datetime.timedelta(seconds=30)

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

    def send_response(self, requesters):
        if requesters:
            self.logger.info('send_resp:')
            for requester in requesters:
                self._send_response(requester)
    def _send_response(self, req_dict):
        requester_agent_id = req_dict['dst_agent_id']
        mask = req_dict['mask']
        reward = req_dict['reward']
        task_label = req_dict['task_label']

        print(requester_agent_id, type(requester_agent_id))
        print(mask, type(mask), mask.dtype)
        print(reward, type(reward))
        print(task_label, type(task_label), task_label.dtype)


        buff = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz, dtype=torch.float32) * torch.inf
        buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

        self.logger.info('send_resp: responding to agent {0} query'.format(requester_agent_id))
        self.logger.info('send_resp: mask (response) data type: {0}'.format(mask))

        if mask is None:
            buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
            buff[ParallelComm.META_INF_IDX_MSK_RW] = torch.inf
            buff[ParallelComm.META_INF_IDX_TASK_SZ_ : ParallelComm.META_INF_IDX_MASK_SZ] = task_label
            buff[ParallelComm.META_INF_IDX_MASK_SZ : ] = torch.inf
        else:
            buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
            buff[ParallelComm.META_INF_IDX_MSK_RW] = reward
            buff[ParallelComm.META_INF_IDX_TASK_SZ_ : ParallelComm.META_INF_IDX_MASK_SZ] = task_label
            buff[ParallelComm.META_INF_IDX_MASK_SZ : ] = mask # NOTE deepcopy? 

        # actual send
        self.logger.info('send_resp: buffer to send {0}'.format(buff))
        send_resp = dist.isend(tensor=buff, dst=requester_agent_id)
        send_resp.wait()
        return

    def receive_response(self, await_response):
        if any(await_response):
            self.logger.info('recv_resp:')
            _buff_list = [torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz, dtype=torch.float32) * torch.inf for _ in range(self.num_agents)]

            for idx in range(self.num_agents):
                if idx == self.agent_id: continue
                self.logger.info('recv_resp: set up handle to receive response from agent {0}'.format(idx))
                recv_resp = dist.irecv(tensor=_buff_list[idx], src=idx)
                recv_resp.wait()
            #time.sleep(ParallelComm.SLEEP_DURATION)

            print('RECEIVED A BUFFER:', _buff_list)

            # check whether message has been received
            ret = []
            for idx in range(self.num_agents):
                if idx == self.agent_id:
                    ret.append(None)
                    continue

                msg = _buff_list[idx]
                if self._null_message(msg):
                    ret.append(None)
                    self.logger.info('recv_resp: appending {0} response'.format(None))
                elif msg[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
                    ret.append(False)
                    self.logger.info('recv_resp: appending False response')
                else:
                    d = {}
                    d['mask'] = copy.deepcopy(msg[ParallelComm.META_INF_IDX_MASK_SZ : ])
                    d['reward'] = msg[ParallelComm.META_INF_IDX_MSK_RW]
                    d['task_label'] = msg[ParallelComm.META_INF_IDX_TASK_SZ_ : ParallelComm.META_INF_IDX_MASK_SZ]
                    ret.append(d)

                # reset buffer and handle
                #_buff_list[idx][:] = torch.inf
                #self.handle_recv_resp[idx] = None 
            return ret

    def send_recv_resp(self, requesters, await_response):
        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        #self.send_response(requesters)
        #return self.receive_response(await_response)

        _ = pool1.apply_async(self.send_response, (requesters,))
        results = pool2.apply_async(self.receive_response, (await_response,))

        pool1.close()
        pool2.close()

        return results.get()

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
        comm_iter = 1
        while True:
            # Get the latest states of these variables
            track_tasks, mask_rewards_dict, await_response = queue_loop.get()

            print()
            print(Fore.GREEN + 'COMMUNICATION ITERATION: ', comm_iter)
            print(Fore.GREEN + '', track_tasks, mask_rewards_dict, await_response)
            

            try:
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



            start_time = time.time()
            requesters = []
            dst_agent_id_temp = None
            print('KNOWLEDGE BASE:', mask_rewards_dict)
            for req in other_agents_request:
                if req is None: continue

                temp_reward = 0.0
                if tuple(req['task_label'].detach().cpu().numpy()) in mask_rewards_dict:
                    temp_reward = mask_rewards_dict[tuple(req['task_label'].detach().cpu().numpy())]
                queue_label_send.put((req['task_label'], req['sender_agent_id'], temp_reward))
                dst_agent_id_temp = req['sender_agent_id']

            try:
                mask, dst_agent_id, mask_reward, task_label = queue_mask_recv.get_nowait()
                d = {}
                d['mask'] = mask
                d['dst_agent_id'] = dst_agent_id
                d['reward'] = mask_reward
                d['task_label'] = task_label
                requesters.append(d)
            except:
                d = {}
                d['mask'] = None
                d['dst_agent_id'] = dst_agent_id_temp
                requesters.append(None)
                print('Agent didnt send back converted mask :(')
                pass
            print('******** TIME TAKEN FOR MASK PROCESSING THIS ITERATION:', time.time()-start_time)




            best_mask = None
            print('Requesters in this iteration:', requesters)
            start_time = time.time()
            masks = []
            received_masks = self.send_recv_resp(requesters, await_response)

            if received_masks:
                for i in range(len(await_response)):
                    if await_response[i] is False: continue
                    if received_masks[i] is False: continue
                    elif received_masks[i] is None: await_response[i] = False
                    else:
                        masks.append(received_masks[i])
                        await_response[i] = False


                self.logger.info('number of task knowledge received: {0}'.format(len(masks)))


                masks = [i for i in masks if i is not None]
                masks = [i for i in masks if i is not False]
                masks = sorted(masks, key=lambda d: d['reward'])


            # Will only run if masks is populated.
            for mask in masks:
                if mask['mask'] is not None:
                    # If knowledge already in knowledge base for this task label
                    if tuple(mask['task_label'].detach().cpu().numpy()) in mask_rewards_dict:
                        # Then check if the reward is higher then the local reward
                        if np.around(mask['reward'], 6) > mask_rewards_dict[tuple(msg)]:
                            best_mask = mask['mask']
                            best_agent_rw = np.around(mask['reward'], 6)

                    # Otherwise get the mask anyway so that we can have some knowledge
                    else:
                        best_mask = mask['mask']
                        best_agent_rw = np.around(mask['reward'], 6)


            print('******** TIME TAKEN FOR SEND RECV MASK:', time.time()-start_time)

            
                        

            print('Mask in this iteration:', best_mask)
            queue_mask.put_nowait((best_mask, track_tasks, await_response, best_agent_rw))
            
            
            comm_iter += 1
            dist.barrier()

    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop):
        p = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop))
        p.start()
        return p