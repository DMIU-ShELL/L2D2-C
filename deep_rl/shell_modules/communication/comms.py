# -*- coding: utf-8 -*-

#   _________                                           .__                  __   .__                 
#   \_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
#   /    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
#   \     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
#    \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
#           \/               \/       \/             \/          \/      \/                        \/ 
#
#                                                   (╯°□°)╯︵ ┻━┻
import os
import copy
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist
import multiprocessing as mp
from queue import Empty
import socket
import pickle


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
from colorama import Fore
class ParallelComm(object):
    # DETECT MODULE CONSTANTS
    # Threshold for embedding/tasklabel distance (similarity)
    # This should be taken from the detect module probably
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
    MSG_TYPE_SEND_QUERY = 0
    MSG_TYPE_SEND_META = 1
    MSG_TYPE_SEND_REQ = 2
    MSG_TYPE_SEND_MASK = 3

    # message data (META_INF_IDX_MSG_DATA) values
    MSG_DATA_NULL = 0 # an empty message
    MSG_DATA_TSKtoQUERY = 1
    MSG_DATA_TSKtoMSK = 2
    MSG_DATA_MSK = 3
    MSG_DATA_META = 4

    # number of seconds to sleep/wait
    SLEEP_DURATION = 1

    # Task label size can be replaced with the embedding size.
    def __init__(self, agent_id, num_agents, emb_label_sz, mask_sz, logger, init_address, init_port, mode, mask_interval):
        super(ParallelComm, self).__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.emb_label_sz = emb_label_sz
        self.mask_sz = mask_sz
        self.logger = logger
        self.mode = mode
        self.mask_interval = mask_interval
        self.init_port = init_port
        self.init_address = init_address
        self.other_ports = [29500 + i for i in range(num_agents)]
        self.other_address = ['127.0.0.1'] * num_agents
        self.other_dst = dict(zip(self.other_address, self.other_ports))

        print('MASK SIZE IS: ', self.mask_sz)
        
        # Setup init string for process group
        #if init_address in ['127.0.0.1', 'localhost']:
        #    os.environ['MASTER_ADDR'] = init_address
        #    os.environ['MASTER_PORT'] = init_port
        #    self.comm_init_str = 'env://'
        #else:
        #    self.comm_init_str = 'tcp://{0}:{1}'.format(init_address, init_port)

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

    def _null_message(self, msg):
        # check whether message sent denotes or is none.
        if bool(msg[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_NULL):
            return True

        else:
            return False

    def client(self, data, address, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((address, port))
                data = pickle.dumps(data)
                s.sendall(data)
                data = None
            return data
        except:
            return data

    # Query send and recv functions
    def send_query(self, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        self.logger.info('send_recv_req, req data: {0}'.format(embedding))

        data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf
        data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
        data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_REQ

        if embedding is None:
            data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL

        else:
            data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_TSKtoQUERY
            data[ParallelComm.META_INF_IDX_TASK_SZ : ] = embedding # NOTE deepcopy?

        # Try to send a query to all known destinations. Skip the ones that don't work
        for address, port in self.other_dst:
            self.client(data, address, port)
    def recv_query(self, buffer):
        ret = []
        if self._null_message(buffer):
            d = {}
            d['sender_agent_id'] = int(buffer[ParallelComm.META_INF_IDX_PROC_ID])
            d['msg_data'] = None
            ret.append(d)

        else:
            d = {}
            d['sender_agent_id'] = int(buffer[ParallelComm.META_INF_IDX_PROC_ID])
            d['msg_type'] = int(buffer[ParallelComm.META_INF_IDX_MSG_TYPE])
            d['msg_data'] = int(buffer[ParallelComm.META_INF_IDX_MSG_DATA])
            d['embedding'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ : ]
            ret.append(d)
        return ret
    
    # Metadata pre-processing, send and recv functions
    def proc_meta(self, other_agent_req, mask_rewards_dict):
        meta_response = {}

        # if populated prepare metadata responses
        if other_agent_req is not None:
            # If the req is none, which it usually will be, just skip.
            if other_agent_req['msg_data'] is None:
                pass

            else:
                req_label_as_np = other_agent_req['embedding'].detach().cpu().numpy()

                # Iterate through the knowledge base and compute the distances
                # Distance calculation should be replaced by a function from the detect module
                for tlabel, treward in mask_rewards_dict.items():
                    if treward != np.around(0.0, decimals=6):
                        distance = np.sum(abs(np.subtract(req_label_as_np, np.asarray(tlabel))))

                        if distance <= ParallelComm.THRESHOLD:
                            meta_response['dst_agent_id'] = other_agent_req['sender_agent_id']
                            meta_response['mask_reward'] = treward
                            meta_response['dist'] = distance
                            meta_response['resp_embedding'] = torch.tensor(tlabel)

        return meta_response
    def send_meta(self, meta_response):
        if meta_response:
            dst_agent_id = meta_response['dst_agent_id']
            mask_reward =meta_response['mask_reward']
            distance = meta_response['dist']
            distance = np.float64(distance)
            embedding = meta_response['resp_embedding']

            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)

            # Consider changing to local buffer
            data = torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + self.emb_label_sz, dtype=torch.float32) * torch.inf
            data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
            data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_META

            # If mask is none then send back torch.inf otherwise send mask
            if mask_reward is torch.inf:
                data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL

            else:
                # if the mask is none but there is a mask reward, then overwrite the buffer with
                # the meta data. Otherwise don't do anything.
                data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_META
                data[ParallelComm.META_INF_IDX_MSK_RW] = mask_reward
                data[ParallelComm.META_INF_IDX_DIST] = distance
                data[ParallelComm.META_INF_IDX_TASK_SZ_ :] = embedding

            _address = self.other_address[dst_agent_id]
            _port = self.other_dst[_address]
            self.client(data, _address, _port)
    def recv_meta(self, buffer):
        ret = {}
        if self._null_message(buffer):
            ret = None
        
        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            # Code should never reach this point
            ret = None

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_META:
            ret['agent_id'] = int(buffer[0])
            ret['mask_reward'] = float(buffer[3])
            ret['dist'] = float(buffer[4])
            ret['embedding'] = buffer[5:].detach().cpu().numpy()

        return ret, int(buffer[0])


    # Mask request pre-processing, send and recv functions
    def proc_mask_req(self, metadata):
        send_msk_requests = dict()
            
        # if not results something bad has happened
        if len(metadata) > 0:
            # Sort received meta data by smallest distance (primary) and highest reward (secondary),
            # using full bidirectional multikey sorting (fancy words for such a simple concept)
            results = sorted(results, key=lambda d: (d['dist'], -d['mask_reward']))
            print(Fore.GREEN + 'Metadata responses sorted: ')
            for item in results:
                print(item)

            # Iterate through the await_response list. Upon task change this is an array:
            # [True,] * num_agents
            selected = False    # flag to check if the best agent of the lot has been selected
            for idx in range(len(await_response)):
                # Do some checks to remove to useless results
                if results[idx]['agent_id'] == self.agent_id:
                    await_response[idx] = False
                    continue

                if await_response[idx] is False:
                    continue

                if results[idx] is False:
                    continue

                elif results[idx]['mask_reward'] == torch.inf:
                    await_response[idx] = False

                # Otherwise unpack the metadata
                else:
                    recv_agent_id = results[idx]['agent_id']
                    recv_msk_rw = results[idx]['mask_reward']
                    recv_dist = results[idx]['dist']
                    recv_label = results[idx]['emb_label']

                    # If the recv_dist is lower or equal to the threshold and a best agent
                    # hasn't been selected yet then continue
                    if recv_dist <= ParallelComm.THRESHOLD and selected == False:
                        # Check if the reward is greater than the current reward for the task
                        # or if the knowledge even exists.
                        if tuple(msg) in mask_rewards_dict.keys():
                            if shell_iterations % self.mask_interval == 0:
                                if round(recv_msk_rw, 6) > mask_rewards_dict[tuple(msg)]:
                                    # Add the agent id and embedding/tasklabel from the agent
                                    # to a dictionary to send requests/rejections to.
                                    send_msk_requests[recv_agent_id] = recv_label
                                    # Make a note of the best agent id in memory of this agent
                                    # We will use this later to get the mask from the best agent
                                    best_agent_id = recv_agent_id
                                    best_agent_rw[tuple(msg)] = np.around(recv_msk_rw, 6)
                                    # Make the selected flag true so we don't pick another agent
                                    selected = True

                                else:
                                    # if this agent's reward is higher or the same as the other best agent
                                    # then reject the best agent
                                    recv_agent_id = results[idx]['agent_id']
                                    send_msk_requests[recv_agent_id] = None


                            else:
                                # if this agent's reward is higher or the same as the other best agent
                                # then reject the best agent
                                recv_agent_id = results[idx]['agent_id']
                                send_msk_requests[recv_agent_id] = None
                                

                        # If we don't have any knowledge present for the task then get the mask 
                        # anyway from the best agent.
                        else:
                            # Add the agent id and embedding/tasklabel from the agent
                            # to a dictionary to send requests/rejections to.
                            send_msk_requests[recv_agent_id] = recv_label
                            # Make a note of the best agent id in memory of this agent
                            # We will use this later to get the mask from the best agent
                            best_agent_id = np.around(recv_agent_id, 6)
                            best_agent_rw[tuple(msg)] = recv_msk_rw
                            # Make the selected flag true so we don't pick another agent
                            selected = True

                    else:
                        # if this agent's reward is higher or the same as the other best agent
                        # then reject the best agent
                        recv_agent_id = results[idx]['agent_id']
                        send_msk_requests[recv_agent_id] = None

                    # We have checked the response so set it to False until the next task change
                    # and request loop begins
                    await_response[idx] = False
    def send_mask_req(self, send_msk_requests):
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
    def recv_mask_req(self, buffer):
        '''
        Unpacks the received mask request into a dictionary
        '''
        ret = {}
        if self._null_message(buffer):
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            pass

        else:
            ret['requester_agent_id'] = int(buffer[ParallelComm.META_INF_IDX_PROC_ID])
            ret['msg_type'] = int(buffer[ParallelComm.META_INF_IDX_MSG_TYPE])
            ret['msg_data'] = int(buffer[ParallelComm.META_INF_IDX_MSG_DATA])
            ret['task_label'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ : ]

        return ret


    # Mask send and recv
    def proc_mask(self, mask_req, queue_label_send, queue_mask_recv):
        resp = {}
        if mask_req:
            # Iterate through the requests
            # Send the label to be converted, to the agent
            
            _temp_labels = {}
            if type(mask_req) is dict:
                print('Mask request: ', mask_req, flush=True)

                # Send a mask to the requesting agent
                dst_agent_id = mask_req['requester_agent_id']
                # Send label:id to agent
                _temp_labels[dst_agent_id] = mask_req['task_label']

            queue_label_send.put((_temp_labels))

            print('Send label to be converted:', _temp_labels, flush=True)

            # wait to receive a mask from the agent module. do not continue until you receive
            # this mask. agent will see the request eventually and send back the converted mask.
            
            masks_list = queue_mask_recv.get()
    def send_mask(self, mask_resp):
        if mask_resp:
            dst_agent_id = int(mask_resp['dst_agent_id'])
            emb_label = mask_resp['label']
            mask = mask_resp['mask']

            data = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz + self.emb_label_sz, dtype=torch.float32) * torch.inf

            data[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
            data[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_MASK

            if mask is None:
                data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
                data[ParallelComm.META_INF_IDX_MASK_SZ : ] = torch.inf

            else:
                data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
                data[ParallelComm.META_INF_IDX_MASK_SZ : ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz] = mask # NOTE deepcopy?
                data[ParallelComm.META_INF_IDX_MASK_SZ + self.mask_sz : ] = emb_label

            _address = self.other_address[dst_agent_id]
            _port = self.other_dst[_address]
            self.client(data, _address, _port)
    def recv_mask(self, buffer, best_agent_id):
        received_mask = None
        received_label = None

        if buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
            if buffer[ParallelComm.META_INF_IDX_PROC_ID] == best_agent_id:
                received_mask = buffer[ParallelComm.META_INF_IDX_MASK_SZ : ParallelComm.META_INF_IDX_MASK_SZ+self.mask_sz]
                received_label = buffer[ParallelComm.META_INF_IDX_MASK_SZ+self.mask_sz : ]

        else:
            pass

            

        return received_mask, received_label


    # Main loop + listening server
    def server(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop):
        shell_iterations = None
        best_agent_id = None
        mask_rewards_dict = {}
        best_agent_rw = {}
        metadata = [None] * self.num_agents
        
        while True:
            recv_mask = None
            recv_embedding = None
            best_agent_id_ = None

            # Do some checks on the agent/communication interaction queues and perform actions based on those
            mask_rewards_dict, shell_iterations = queue_loop.get_nowait()

            try:
                msg = queue_label.get_nowait()
            except Empty:
                msg = None

            '''try:
                masks_list = queue_mask_recv.get_nowait()
            except Empty:
                masks_list = []

            if len(masks_list) > 0:
                pass
                # Send some masks?'''
            
            # Send out a query when shell iterations matches mask interval if the agent is working on a task
            if msg is not None:
                if shell_iterations % self.mask_interval == 0:
                    self.send_query(msg)

            # Listen for messages
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.init_address, self.init_port))
                s.listen()
                conn, addr = s.accept()
                with conn:
                    self.logger.info(f"Connected by {addr}")
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            print('Something went wrong. Nothing was received')
                            break
                        
                        # Unpack the received data
                        data = pickle.loads(data)
                        self.logger.info(f"Received {data!r}")

                        # Events based on what data it receives
                        # Maybe we can multithread these when the time comes...
                        if data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_QUERY:
                            other_agent_req = self.recv_query(data)
                            meta_response = self.proc_meta(other_agent_req, mask_rewards_dict)
                            self.send_meta(meta_response)

                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_META:
                            # Come back and figure this part out later...
                            other_agent_meta, idx = self.recv_meta(data)
                            metadata[idx] = other_agent_meta
                            best_agent_id, mask_req = self.proc_mask_req(metadata)
                            best_agent_id_ = best_agent_id

                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_REQ:
                            mask_req = self.recv_mask_req(data)
                            mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
                            self.send_mask(mask_resp)
                        
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_MASK:
                            recv_mask, recv_embedding = self.recv_mask(data, best_agent_id)
                            best_agent_id = None

            queue_mask.put_nowait((recv_mask, best_agent_rw, best_agent_id_, recv_embedding))
            

            

    ### Core functions
    '''def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, queue_check):
        msg = None
        # Store the best agent id for quick reference
        best_agent_id = None
        best_agent_rw = {}

        # initial state of input variables to loop
        comm_iter = 0
        while True:
            START = time.time()
            expecting = list() # Come back to once all components are checked. Might cause issues

            # Get the latest states of these variables
            track_tasks, mask_rewards_dict, await_response, shell_iterations = queue_loop.get()
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
            dist.monitored_barrier(wait_all_ranks=True)
            other_agents_request = self.send_receive_request(msg)
            END1 = time.time()-start_time
            print('******** TIME TAKEN FOR SEND_RECV_REQ():', END1)
            print()
            print(Fore.GREEN + 'Other agent requests: ', other_agents_request)

            
            #######################   COMMUNICATION STEP TWO    #######################
            ####################### SEND AND RECV META REPONSES #######################
            # Respond to received queries with metadata.
            # Meta data contains the reward for the similar task, distance and the similar 
            # tasklabel/embedding.

            ### SEND META RESPONSES
            # Go through each request from the network of agents


            ### SEND RECV META RESPONSES
            # Receive metadata response from other agents for a embedding/tasklabel request from 
            # this agent.
            #print(Fore.GREEN + 'Awaiting Responses? ', await_response)

            results = []
            start_time = time.time()
            dist.monitored_barrier(wait_all_ranks=True)
            results = self.send_recv_meta(meta_responses, await_response)
            END2 = time.time()-start_time
            print('******** TIME TAKEN FOR SEND_RECV_META():', END2)
            print()



            

            print(Fore.GREEN + 'Mask requests to send to other agents: ', send_msk_requests)

            best_agent_id_ = best_agent_id  # temp variable for logging purposes
            
            #######################     COMMUNICATION STEP FOUR      #######################
            ####################### SEND MASK REQUESTS OR REJECTIONS #######################
            ### SEND MASK REQUEST OR REJECTION
            msk_requests = []
            print('Before send_recv_req():', send_msk_requests, expecting)
            start_time = time.time()
            dist.monitored_barrier(wait_all_ranks=True)
            msk_requests = self.send_recv_mask_req(send_msk_requests, expecting)
            END3 = time.time()-start_time
            print('******** TIME TAKEN FOR SEND_RECV_MASK_REQ():', END3)
            print()

            print(Fore.GREEN + 'After send_recv_req():', msk_requests)


            ####################### COMMUNICATION STEP FIVE #######################
            # Now the agent needs to send a mask to each agent in the msk_requests list
            # if it is not empty



            print()
            print('Before mask exchange:', msk_requests, best_agent_id)

            

                #for item in conversions:
                #    d = {}

                #    for dst, mask in conversions.items():
                #        d = {}
                #        d['mask'] = mask
                #        d['dst_agent_id'] = dst
                        
                #        masks_list.append(d)
                


            print()
            print()
            print('Masks to send:', masks_list)
            received_mask = None
            received_label = None
            start_time = time.time()
            dist.monitored_barrier(wait_all_ranks=True)
            received_mask, best_agent_id, received_label = self.send_recv_mask(masks_list, best_agent_id)
            END4 = time.time()-start_time
            print('***** TIME TAKEN FRO SEND_RECV_MASK():', END4)
            print(Fore.GREEN + 'Mask received for distillation', received_mask, best_agent_id, received_label, flush=True)
            queue_mask.put_nowait((received_mask, track_tasks, await_response, best_agent_rw, best_agent_id_, received_label))

            comm_iter += 1

            END5 = time.time()-START
            print('***** COMM ITERATION TIME ELAPSED:', END5)
            timings.append([comm_iter, END1, END2, END3, END4, END5])
            np.savetxt(self.logger.log_dir + '/timings_{0}.csv'.format(self.agent_id), timings, delimiter=',')
'''
    
    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop):
        p = mp.Process(target=self.server, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop))
        p.start()
        return p


class ParallelCommEval(object):
    def __init__(self):
        pass

