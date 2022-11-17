# -*- coding: utf-8 -*-

#   _________                                           .__                  __   .__                 
#   \_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
#   /    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
#   \     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
#    \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
#           \/               \/       \/             \/          \/      \/                        \/ 
#
#                                                   (╯°□°)╯︵ ┻━┻
import copy
import datetime
import multiprocessing as mp
import multiprocessing.dummy as mpd
import os
import pickle
import socket
import ssl
import time
from queue import Empty

import numpy as np
import torch

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
    META_INF_IDX_ADDRESS = 0
    META_INF_IDX_PORT = 1
    META_INF_IDX_MSG_TYPE = 2
    META_INF_IDX_MSG_DATA = 3

    META_INF_IDX_MSK_RW = 4
    META_INF_IDX_TASK_SZ = 4 # only for the send_recv_request buffer

    META_INF_IDX_DIST = 5
    META_INF_IDX_TASK_SZ_ = 6 # for the meta send recv buffer
    

    META_INF_IDX_MASK_SZ = 4
    
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

    TIMEOUT = 5

    # Task label size can be replaced with the embedding size.
    def __init__(self, agent_id, num_agents, emb_label_sz, mask_sz, logger, init_address, init_port, mode, mask_interval, addresses, ports):
        super(ParallelComm, self).__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.emb_label_sz = emb_label_sz
        self.mask_sz = mask_sz
        self.logger = logger
        self.mode = mode
        self.mask_interval = mask_interval
        self.init_port = int(init_port)
        self.init_address = init_address

        # Address-port lookup table

        self.other_ports = ports
        self.other_address = addresses

        print('ports:', self.other_ports)
        print('addresses:', self.other_address)

        print('mask size:', self.mask_sz)
        
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
        '''self.buff_send_recv_req = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ + emb_label_sz, dtype=torch.float32) \
            * torch.inf for _ in range(num_agents)]

        self.buff_recv_meta = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]
        self.buff_send_meta = [torch.ones(ParallelComm.META_INF_IDX_TASK_SZ_ + emb_label_sz, dtype=torch.float32) * torch.inf \
            for _ in range(num_agents)]

        self.buff_recv_mask = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf 
        self.buff_send_mask = torch.ones(ParallelComm.META_INF_IDX_MASK_SZ + mask_sz, dtype=torch.float32) * torch.inf'''

    def _null_message(self, msg):
        # check whether message sent denotes or is none.
        if bool(msg[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_NULL):
            return True

        else:
            return False

    def client(self, data, address, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                #self.logger.info(f'Sending query to {address}:{port}')
                s.connect((address, port))
                s.sendall(data)
                data = None
            return data
        except:
            return data

    # Query send and recv functions
    def send_query(self, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        if embedding is None:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_TSKtoQUERY, embedding]

        # Try to send a query to all known destinations. Skip the ones that don't work
        data = pickle.dumps(data)
        if len(self.other_address) == len(self.other_ports):
            for i in range(len(self.other_address)):
                if self.other_address[i] == self.init_address and self.other_ports[i] == self.init_port: continue
                self.client(data, self.other_address[i], self.other_ports[i])
    def recv_query(self, buffer):
        ret = {}
        if self._null_message(buffer):
            ret = None

        else:
            ret['sender_address'] = str(buffer[ParallelComm.META_INF_IDX_ADDRESS])
            ret['sender_port'] = int(buffer[ParallelComm.META_INF_IDX_PORT])
            ret['msg_type'] = int(buffer[ParallelComm.META_INF_IDX_MSG_TYPE])
            ret['msg_data'] = int(buffer[ParallelComm.META_INF_IDX_MSG_DATA])
            ret['embedding'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ]

        return ret
    
    # Metadata pre-processing, send and recv functions
    def proc_meta(self, other_agent_req, knowledge_base):
        meta_response = {}
        print(other_agent_req)
        # if populated prepare metadata responses
        if other_agent_req is not None:
            # If the req is none, which it usually will be, just skip.
            #if other_agent_req['msg_data'] is None: pass

            #else:
            req_label_as_np = other_agent_req['embedding'].detach().cpu().numpy()
            print(req_label_as_np, type(req_label_as_np), flush=True)

            # Iterate through the knowledge base and compute the distances
            print('Knowledge base in proc_meta:', knowledge_base)
            for tlabel, treward in knowledge_base.items():
                print(tlabel, treward, flush=True)
                if treward != np.around(0.0, decimals=6):
                    distance = np.sum(abs(np.subtract(req_label_as_np, np.asarray(tlabel))))
                    print(distance, flush=True)
                    
                    if distance <= ParallelComm.THRESHOLD:
                        meta_response['dst_address'] = other_agent_req['sender_address']
                        meta_response['dst_port'] = other_agent_req['sender_port']
                        meta_response['mask_reward'] = treward
                        meta_response['dist'] = distance
                        meta_response['resp_embedding'] = torch.tensor(tlabel)

        return meta_response
    def send_meta(self, meta_response):
        if meta_response:
            dst_address = meta_response['dst_address']
            dst_port = meta_response['dst_port']
            mask_reward = meta_response['mask_reward']
            distance = meta_response['dist']
            distance = np.float64(distance)
            embedding = meta_response['resp_embedding']

            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)

            # Consider changing to local buffer
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_META]

            if mask_reward is torch.inf:
                data.append(ParallelComm.MSG_DATA_NULL)

            else:
                data.append(ParallelComm.MSG_DATA_META)
                data.append(mask_reward)
                data.append(distance)
                data.append(embedding)

            data = pickle.dumps(data)
            if dst_address in self.other_address and dst_port in self.other_ports:
                if self.other_address.index(dst_address) == self.other_ports.index(dst_port):
                    self.client(data, dst_address, dst_port)
    def recv_meta(self, buffer):
        ret = {'address': None, 'port': None, 'mask_reward': 0.0, 'dist': torch.inf, 'embedding': None}
        if self._null_message(buffer):
            pass
        
        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            # Code should never reach this point
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_META:
            ret['address'] = str(buffer[ParallelComm.META_INF_IDX_ADDRESS])
            ret['port'] = int(buffer[ParallelComm.META_INF_IDX_PORT])
            ret['mask_reward'] = float(buffer[ParallelComm.META_INF_IDX_MSK_RW])
            ret['dist'] = float(buffer[ParallelComm.META_INF_IDX_DIST])
            ret['embedding'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ_].detach().cpu().numpy()

        return ret, ret['address'], ret['port']


    # Mask request pre-processing, send and recv functions
    def proc_mask_req(self, metadata, knowledge_base):
        send_msk_requests = {}
        best_agent_id = None
        best_agent_rw = {}

        print(metadata)
        print(knowledge_base)
            
        # if not results something bad has happened
        if len(metadata) > 0:
            # Sort received meta data by smallest distance (primary) and highest reward (secondary),
            # using full bidirectional multikey sorting (fancy words for such a simple concept)
            metadata = {k: metadata[k] for k in sorted(metadata, key=lambda d: (metadata[d]['dist'], -metadata[d]['mask_reward']))}
            print(Fore.GREEN + 'Metadata responses sorted: ')
            for item in metadata:
                print(item)

            
            best_agent_id = None
            best_agent_rw = {}

            for _, data_dict in metadata.items():
                # Do some checks to remove to useless results
                if str(data_dict['address'] + ':' + str(data_dict['port'])) == str(self.init_address + ':' + str(self.init_port)): continue
                if data_dict is None: continue
                elif data_dict['mask_reward'] == torch.inf: pass

                # Otherwise unpack the metadata
                else:
                    recv_address = data_dict['address']
                    recv_port = data_dict['port']
                    recv_msk_rw = data_dict['mask_reward']
                    recv_dist = data_dict['dist']
                    recv_label = data_dict['embedding']

                    # If the recv_dist is lower or equal to the threshold and a best agent
                    # hasn't been selected yet then continue
                    if recv_msk_rw != 0.0:
                        if recv_dist <= ParallelComm.THRESHOLD:
                            # Check if the reward is greater than the current reward for the task
                            # or if the knowledge even exists.
                            if tuple(recv_label) in knowledge_base.keys():
                                #if shell_iterations % self.mask_interval == 0:
                                if round(recv_msk_rw, 6) > knowledge_base[tuple(recv_label)]:
                                    # Add the agent id and embedding/tasklabel from the agent
                                    # to a dictionary to send requests/rejections to.
                                    send_msk_requests[recv_address + ':' + str(recv_port)] = recv_label
                                    # Make a note of the best agent id in memory of this agent
                                    # We will use this later to get the mask from the best agent
                                    best_agent_id = recv_address + ':' + str(recv_port)
                                    best_agent_rw[tuple(recv_label)] = np.around(recv_msk_rw, 6)
                                    break

                            # If we don't have any knowledge present for the task then get the mask 
                            # anyway from the best agent.
                            else:
                                send_msk_requests[recv_address + ':' + str(recv_port)] = recv_label
                                best_agent_id = recv_address + ':' + str(recv_port)
                                best_agent_rw[tuple(recv_label)] = np.around(recv_msk_rw, 6)
                                break

        return send_msk_requests, best_agent_id, best_agent_rw
    def send_mask_req(self, send_msk_requests):
        print('SEND_MSK_REQ: ', send_msk_requests, flush=True)

        if send_msk_requests:
            for destination, embedding in send_msk_requests.items():
                destination = destination.split(':')
                dst_address = destination[0]
                dst_port = destination[1]


                # Convert embedding label to tensor
                if isinstance(embedding, np.ndarray):
                    embedding = torch.tensor(embedding, dtype=torch.float32)

                data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_REQ]
                
                if embedding is None:
                    # If emb_label is none it means we reject the agent
                    data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
                        
                else:
                    # Otherwise we want the agent's mask
                    data[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_TSKtoMSK
                    data[ParallelComm.META_INF_IDX_TASK_SZ] = embedding # NOTE deepcopy?

                # Send out the mask request or rejection to each agent that sent metadata
                data = pickle.dumps(data)
                if dst_address in self.other_address and dst_port in self.other_ports:
                    if self.other_address.index(dst_address) == self.other_ports.index(dst_port):
                        self.client(data, dst_address, dst_port)
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
            ret['sender_address'] = str(buffer[ParallelComm.META_INF_IDX_ADDRESS])
            ret['sender_port'] = int(buffer[ParallelComm.META_INF_IDX_PORT])
            ret['msg_type'] = int(buffer[ParallelComm.META_INF_IDX_MSG_TYPE])
            ret['msg_data'] = int(buffer[ParallelComm.META_INF_IDX_MSG_DATA])
            ret['embedding'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ]

        return ret


    # Mask send and recv
    def proc_mask(self, mask_req, queue_label_send, queue_mask_recv):
        resp = {}
        if mask_req:
            # Iterate through the requests
            # Send the label to be converted, to the agent
            
            conversion_req = {}
            if type(mask_req) is dict:
                print('Mask request: ', mask_req, flush=True)
                
                # Send label:id to agent
                conversion_req['address'] = mask_req['sender_address']
                conversion_req['port'] = mask_req['sender_port']
                conversion_req['embedding'] = mask_req['embedding']

            queue_label_send.put((conversion_req))

            print('Send label to be converted:', conversion_req, flush=True)

            # wait to receive a mask from the agent module. do not continue until you receive
            # this mask. agent will see the request eventually and send back the converted mask.
            
            masks_list = queue_mask_recv.get()
    def send_mask(self, mask_resp):
        if mask_resp:
            dst_address = str(mask_resp['dst_address'])
            dst_port = int(mask_resp['dst_port'])
            embedding = mask_resp['label']
            mask = mask_resp['mask']

            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_MASK]

            if mask is None:
                data.append(ParallelComm.MSG_DATA_NULL)

            else:
                data.append(ParallelComm.MSG_DATA_MSK)
                data.append(mask)
                data.append(embedding)
            
            data = pickle.dumps(data)
            if dst_address in self.other_address and dst_port in self.other_ports:
                if self.other_address.index(dst_address) == self.other_ports.index(dst_port):
                    self.client(data, dst_address, dst_port)
    def recv_mask(self, buffer, best_agent_id):
        received_mask = None
        received_label = None

        if buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
            if buffer[ParallelComm.META_INF_IDX_ADDRESS] + ':' + str(buffer[ParallelComm.META_INF_IDX_PORT]) == best_agent_id:
                received_mask = buffer[4]
                received_label = buffer[5]
        else:
            pass

            

        return received_mask, received_label


    def query(self, data, knowledge_base):
        other_agent_req = self.recv_query(data)
        self.logger.info(Fore.GREEN + 'other agent request:', other_agent_req)
        meta_response = self.proc_meta(other_agent_req, knowledge_base)
        self.send_meta(meta_response)
    def meta(self, data, metadata, knowledge_base, time_to_select):
        other_agent_meta, address, port = self.recv_meta(data)
        if address is not None and port is not None:
            metadata[address + ':' + str(port)] = other_agent_meta

        if time_to_select:
            print('Time to select best agent!!!! :DDDDDD')
            mask_req, best_agent_id, best_agent_rw = self.proc_mask_req(metadata, knowledge_base)
            self.send_mask_req(mask_req)

            return True, best_agent_id, best_agent_rw

        return False, None, {}
    def req(self, data, queue_label_send, queue_mask_recv):
        mask_req = self.recv_mask_req(data)
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
        self.send_mask(mask_resp)

    def server(self, knowledge_base, queue_mask, queue_mask_recv, queue_label_send):
        metadata = {}
        flag = False
        start_time = 0
        while True:
            recv_mask = None
            best_agent_rw = None
            best_agent_id = None
            recv_embedding = None
                

            # Listen for messages
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.init_address, self.init_port))
                s.listen()
                conn, addr = s.accept()
                with conn:
                    self.logger.info(Fore.GREEN + f"Connected by {addr}")
                    while True:

                        data = conn.recv(4096)
                        if data[0] != addr[0] or data[1] != addr[1]: break  # If the address doesn't match the address in the buffer then something is very wrong. Close the connection
                        if not data: break  # If not data then close the connection

                        # Unpack the received data
                        data = pickle.loads(data)
                        self.logger.info(Fore.GREEN + f"RECEIVED DATA: {data!r}")
                        print(knowledge_base)

                        # Events based on what data it receives
                        # Maybe we can multithread these when the time comes...
                        if data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_QUERY:
                            t_query = mpd.Pool(processes=1)
                            _ = t_query.apply_async(self.query, (data, knowledge_base))
                            t_query.close()

                        #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_META:
                        #    if flag == False:
                        #        flag = True
                        #        start_time = time.time()

                        #    select = False
                        #    time_elapsed = time.time() - start_time
                            # Wait for 10 seconds or 0.5s*num_agents. Whichever comes first
                            # Maybe fork a new thread here to start the sorting process after time and use the metadata list.
                            # thread will start, and wait for x amount of time. i.e., sleep(x) and then carry out the sort-select process and return this to the agent.
                            # Will have to figure out how to get the information back.
                            # We are going to run out of cores/threads very soon :^)
                        #    if time_elapsed >= 500*self.num_agents or time_elapsed >= 5000:
                        #        select = True

                        #    t_meta = mpd.Pool(processes=1)
                        #    result = t_meta.apply_async(self.meta, (data, metadata, knowledge_base, select))
                        #    flag, best_agent_id, best_agent_rw = result.get()
                        #    t_meta.close()

                        #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_REQ:
                        #    t_req = mpd.Pool(processes=1)
                        #    _ = t_req.apply_async(self.req, (data, queue_label_send, queue_mask_recv))
                        #    t_req.close()

                        #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_MASK:
                        #    recv_mask, recv_embedding = self.recv_mask(data, best_agent_id)
                        #    best_agent_id = None

            if recv_mask is not None and best_agent_rw is not None and best_agent_id is not None and recv_embedding is not None:
                queue_mask.put((recv_mask, best_agent_rw, best_agent_id, recv_embedding))

    # Main loop + listening server
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base):
        shell_iterations = None
        #best_agent_id = None
        #best_agent_rw = {}
        #metadata = [None] * self.num_agents

        # Initialise the listening server
        p_server = mp.Process(target=self.server, args=(knowledge_base, queue_mask, queue_mask_recv, queue_label_send))
        p_server.start()

        # Initialise the client loop
        while True:
            print()
            # Do some checks on the agent/communication interaction queues and perform actions based on those
            shell_iterations = queue_loop.get()     # This makes the communication module synchronised to the agent. If we remove this, the communication module will be superfast. Maybe this can make the system explode.
            self.logger.info(Fore.GREEN + f'Knowledge base in this iteration: {knowledge_base}')

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
                if shell_iterations % 1 == 0:
                    self.send_query(msg)

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
    
    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base):
        p_client = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base))
        p_client.start()


class ParallelCommEval(object):
    def __init__(self):
        pass