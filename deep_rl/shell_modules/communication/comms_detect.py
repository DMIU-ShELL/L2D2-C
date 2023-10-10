# -*- coding: utf-8 -*-
#   _________                                           .__                  __   .__                 
#   \_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
#   /    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
#   \     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
#    \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
#           \/               \/       \/             \/          \/      \/                        \/ 
#
#                                                 (╯°□°)╯︵ ┻━┻
from colorama import Fore
from copy import deepcopy
import datetime
import multiprocessing as mp
import multiprocessing.dummy as mpd
import os
import sys
import pickle
import socket
import ssl
import struct
import time
from errno import ECONNRESET
from queue import Empty
from itertools import cycle
import urllib.request
from random import shuffle, uniform
import torch.nn.functional as F

import numpy as np
import torch
import traceback
import ipinfo




class ParallelCommDetect(object):
    ### COMMUNCIATION MODULE HYPERPARAMETERS
    # DETECT MODULE CONSTANTS
    # Threshold for embedding/tasklabel distance (similarity)
    # This should be taken from the detect module eventually
    THRESHOLD = 0.0

    # SSL/TLS PATHS
    # Paths to the SSL/TLS certificates and key
    CERTPATH = 'certificates/certificate.pem'
    KEYPATH = 'certificates/key.pem'

    
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
    MSG_TYPE_SEND_JOIN = 4
    MSG_TYPE_SEND_LEAVE = 5
    MSG_TYPE_SEND_TABLE = 6

    # message data (META_INF_IDX_MSG_DATA) values
    MSG_DATA_NULL = 0 # an empty message
    MSG_DATA_QUERY = 1
    MSG_DATA_MSK_REQ = 2
    MSG_DATA_MSK = 3
    MSG_DATA_META = 4

    # Task label size can be replaced with the embedding size.
    def __init__(self, embd_dim, mask_dim, logger, init_port, reference, seen_tasks, manager, localhost, mode, dropout, threshold):
        super(ParallelCommDetect, self).__init__()
        self.embd_dim = embd_dim            # Dimensions of the the embedding
        self.mask_dim = mask_dim            # Dimensions of the mask for use in buffers. May no longer be needed
        self.logger = logger                # Logger object for logging CLI outputs.
        self.mode = mode                    # Communication operation mode. Currently only ondemand knowledge is implemented
        self.threshold = threshold

        # Address and port for this agent
        if localhost: self.init_address = '127.0.0.1'
        else: self.init_address = self.init_address = urllib.request.urlopen('https://v4.ident.me').read().decode('utf8') # Use this to get the public ip of the host server.
        self.init_port = int(init_port)

        # Shared memory variables. Make these into attributes of the communication module to make it easier to use across the many sub processes of the module.
        self.query_list = manager.list([item for item in reference if item != (self.init_address, self.init_port)]) # manager.list(reference)
        self.reference_list = manager.list(deepcopy(self.query_list))   # Weird thing was happening here so used deepcopy to recreate the manager ListProxy with the addresses.
        self.knowledge_base = seen_tasks
        self.world_size = manager.Value('i', len(self.reference_list))
        self.masks = manager.list()


        

        # COMMUNICATION DROPOUT
        # Used to simulate percentage communication dropout in the network. Currently only limits the amount of queries and not a total communication blackout.
        self.dropout = dropout  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout


        
        # LOGGING INCOMING CONNECTIONS FOR VISUALISATION TOOL
        #access_token = '8ad435f2bc1b48'
        #self.handler = ipinfo.getHandler(access_token)

        #details = self.handler.getDetails(self.init_address)
        #self.connections = [['ip', 'port', 'country', 'city', 'region', 'timezone', 'postal', 'lat', 'long', 'timestamp']]
        #self.connections.append([details.ip, self.init_port, details.country, details.city, details.region, details.timezone, details.postal, details.latitude, details.longitude, time.time()])

        # Debugging CLI outputs
        self.debug_output()

    def debug_output(self):
        print('Query table:')
        for addr in self.query_list: print(addr[0], addr[1])

        print('\nReference table:')
        for addr in self.reference_list: print(addr[0], addr[1])

        print(f'\nlistening server params ->\naddress: {self.init_address}\nport: {self.init_port}\n')
        print(f'mask size: {self.mask_dim}')
        print(f'embedding size: {self.embd_dim}\n')

    def _null_message(self, msg):
        """
        Checks if a message contains null i.e, no data.

        Args:
            msg: A list received from another agent.

        Returns:
            A boolean indicating whether A list contains null data.
        """

        # check whether message sent denotes or is none.
        if bool(msg[ParallelCommDetect.META_INF_IDX_MSG_DATA] == ParallelCommDetect.MSG_DATA_NULL):
            return True

        else:
            return False

    '''###############################################################################
    ### 
    def proc_meta(self, other_agent_req):
        """
        Processes a query for an embedding and produces a response to send back to the requesting agent.
        
        Args:
            other_agent_req: A dictionary containing the information for the query request.
        
        Returns:
            meta_response: A dictionary containing the response information.
        """

        if other_agent_req is not None:
            other_agent_req['response'] = False
            embedding = other_agent_req['embedding'].detach().cpu().numpy()
            sender_reward = other_agent_req['sender_reward']

            # Iterate through the knowledge base and compute the distances
            for tlabel, treward in self.knowledge_base.items():
                if treward > 0.0:
                    if 0.9 * treward > sender_reward:
                        tdist = float(torch.linalg.vector_norm(embedding - torch.squeeze(torch.tensor(tlabel))))
                        self.logger.info(f'{tdist} meta distance')
                        if tdist <= self.threshold:
                            other_agent_req['response'] = True
                            other_agent_req['reward'] = treward
                            other_agent_req['dist'] = tdist
                            other_agent_req['resp_embedding'] = torch.squeeze(torch.tensor(tlabel))

        return other_agent_req
    def send_meta(self, meta_resp):
        if meta_resp and meta_resp['response']:
            data = [
                self.init_address,
                self.init_port,
                ParallelCommDetect.MSG_TYPE_SEND_META,
                ParallelCommDetect.MSG_DATA_META,
                meta_resp.get('reward', None),
                meta_resp.get('dist', None),
                meta_resp.get('resp_embedding', None)
            ]

            self.client(data, str(meta_resp['sender_address']), int(meta_resp['sender_port']))
    def send_to_agent(self, queue_mask):
        self.logger.info(Fore.CYAN + 'Data is a mask')
        # Unpack the received data
        received_masks = self.recv_masks(data)
        received_mask, received_label, received_reward, ip, port = self.recv_mask(data)

        self.logger.info(f'{received_mask, received_label, received_reward, ip, port}')
        # Send the reeceived information back to the agent process if condition met
        if received_mask is not None and received_label is not None and received_reward is not None:
            self.logger.info('Sending mask data to agent')
            queue_mask.put(self.masks)'''

    ###############################################################################
    ### Query send and recv methods.
    def send_query(self, dict_to_query, queue_mask):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            dict_to_query: A dictionary with structure {'task_emb': <tensor>, 'reward': <float>, 'ground_truth': <tensor>}
        """

        embedding = dict_to_query['task_emb']
        reward = dict_to_query['reward']
        label = dict_to_query['ground_truth']   # For validation purposes. We can feed this into pick_meta() to perform validation and log false positives.

        # Prepare the data for sending
        if embedding is None:
            data = [self.init_address, self.init_port, ParallelCommDetect.MSG_TYPE_SEND_QUERY, ParallelCommDetect.MSG_DATA_NULL]
        else:
            data = [self.init_address, self.init_port, ParallelCommDetect.MSG_TYPE_SEND_QUERY, ParallelCommDetect.MSG_DATA_QUERY, embedding, reward]

        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in list(self.query_list):
            self.client(data, addr[0], addr[1], is_query=True)

        time.sleep(0.2)
        self.logger.info('Sending mask data to agent')
        queue_mask.put(list(self.masks))

    def update_params(self, data):
        _query_list = data[3]
        _query_list.reverse()

        # Insert addresses from _query_list into query_list if they are not already present
        for addr in _query_list:
            if addr not in self.query_list:
                self.query_list.insert(0, addr)

        self.world_size.value = len(self.query_list) + 1

    def received_query(self, data, queue_label_send, queue_mask_recv):
        """
        Event handler for receiving a query from another agent. Unpacks the buffer received from another agent, processes the request and sends a mask response if conditions met.
        
        Args:
            data: A list received from another agent.
        """

        # Query to mask response pipeline
        def recv_query(buffer):
            """
            Unpacks the data buffer received from another agent for a query.
            
            Args:
                buffer: A list received from another agent.
                
            Returns:
                ret: A dictionary containing the unpacked data.
            """
            sender_address = str(buffer[ParallelCommDetect.META_INF_IDX_ADDRESS])
            sender_port = int(buffer[ParallelCommDetect.META_INF_IDX_PORT])
            embedding = buffer[ParallelCommDetect.META_INF_IDX_TASK_SZ]
            sender_reward = buffer[-1]

            # Create a dictionary with the unpacked data
            ret = {
                'sender_address': sender_address,
                'sender_port': sender_port,
                'sender_embedding': embedding,
                'sender_reward': sender_reward
            }

            # Handle when receiving a query from an unknown agent
            if (sender_address, sender_port) not in self.query_list:
                self.client([self.init_address, self.init_port, ParallelCommDetect.MSG_TYPE_SEND_TABLE, list(self.query_list)], sender_address, sender_port)
                self.query_list.append((sender_address, sender_port))

            # Refresh the world_size value
            self.world_size.value = len(self.query_list) + 1

            return ret
        
        def proc_meta(query):
            """
            Find the most similar task record and get the internal task id if any satisfying entries found. Create response dictionary and return otherwise return NoneType.
            
            Args:
                query: A dictionary consisting of the response information to send to a specific agent.

            Returns:
                The mask_req dictionary with the converted mask now included.
            """
            target_embedding = query['sender_embedding']
            target_reward = query['sender_reward']

            self.logger.info(f'Sender emb: {target_embedding}, Sender rw: {target_reward}')

            # Extract embeddings and rewards from the data_dict
            embeddings = []
            rewards = []

            # Get embedding-rewards for any entry where the reward condition 0.9 * reward > sender_reward is met.
            for value in self.knowledge_base.values():
                if 'task_emb' in value:
                    known_embedding = value['task_emb']
                    known_reward = value['reward']

                    # Apply the condition: 0.9 * known_reward > target_reward
                    if 0.9 * known_reward > target_reward:
                        embeddings.append(known_embedding)
                        rewards.append(known_reward)

            self.logger.info(f'embeddings: {embeddings}')
            self.logger.info(f'rewards: {rewards}')
                        

            # If any are found from previous condition then we want to compute embedding distances and get the entry with the most similarity under the threshold.
            if len(embeddings) > 0:
                try:
                    # Convert to a list of PyTorch tensors
                    embeddings = torch.stack(embeddings)
                    #rewards = torch.stack(rewards)

                    # Calculate cosine similarities using PyTorch
                    similarities = F.cosine_similarity(embeddings, target_embedding.unsqueeze(0))
                    self.logger.info(Fore.LIGHTGREEN_EX + f'SIMILARITIES: {similarities}, {type(similarities)}')
                    
                    # Find the index of the closest entry
                    closest_index = torch.argmax(similarities).item()

                    # Get the corresponding key
                    closest_key = list(self.knowledge_base.keys())[closest_index]

                    # Get the closest similarity
                    closest_similarity = torch.min(similarities).item()

                    # Define a threshold for cosine similarity (adjust as needed)
                    cosine_similarity_threshold = 0.4  # Example threshold

                    print(closest_key)

                    # Check if the closest similarity is above the threshold
                    if closest_similarity >= cosine_similarity_threshold:
                        # Get information from the registry for the closest key (internal task id)
                        query['response_reward'] = self.knowledge_base[closest_key]['reward']
                        query['response_similarity'] = closest_similarity  # Convert to a Python float
                        query['response_task_id'] = closest_key
                        query['response_embedding'] = self.knowledge_base[closest_key]['task_emb']
                        query['response_label'] = self.knowledge_base[closest_key]['ground_truth']

                        '''at this point query is contains the following:
                        query = {
                        Query:
                        sender_address  (String)
                        sender_port (Integer)
                        sender_embedding    (Tensor)
                        sender_reward   (Float)
                        
                        Response:
                        response_reward (Float)
                        response_dist   (Float)
                        response_task_id    (Integer)
                        response_embedding  (Tensor)
                        response_label  (Tensor)
                        
                        response_mask   (Tensor) (Added in the convert process in the agent)
                        }'''
                    
                    else:
                        self.logger.info(Fore.GREEN + "No entries satisfying the condition found.")
                        return None
                    
                except Exception as e:
                    traceback.print_exc()
        
        def send_meta(response):
            """
            Sends a mask response to a specific agent.
            
            Args:
                mask_resp: A dictionary consisting of the information to send to a specific agent.    
            """
            data = [
                self.init_address,
                self.init_port,
                ParallelCommDetect.MSG_TYPE_SEND_MASK,
                ParallelCommDetect.MSG_DATA_MSK,
                response.get('response_reward', None),
                response.get('response_similarity', None),
                response.get('response_embedding', None),
                response.get('response_task_id', None),
                response.get('response_label', None)
            ]

            self.logger.info(f'Sending metadata: {data}')
            self.client(data, str(response['sender_address']), int(response['sender_port']))
   

        # Unpack the query from the other agent
        query = recv_query(data)
        self.logger.info(Fore.CYAN + f'Received query: {query}')

        # Get the mask with the most task similarity, if any such mask exists in the network.
        response = proc_meta(query)
        self.logger.info(Fore.CYAN + f'Processed mask request: {response}')

        # Send mask to querying agent if response is not NoneType
        if response is not None:
            send_meta(response)
    
    def received_meta(self, data):
        
        def recv_meta(data):
            # TODO: Unpack metadata into dictionary an return
            pass

        def proc_requests(response):
            # TODO: Take metadata and determine top 5 agents to request mask from. Return list of mask request dicitonaries.
            pass

        def send_requests(response):
            # TODO: Send mask requests to top 5 agents by iterating over list of mask request dictionaries.
            pass

        meta_data = recv_meta(data)
        self.logger.info(Fore.CYAN + f'Received metadata: {meta_data}')

        mask_requests = proc_requests(meta_data)
        self.logger.info(Fore.CYAN + f'Processed mask requests: {mask_requests}')

        if mask_requests is not None:
            send_requests(mask_requests)

    def received_request(self, data, queue_label_send, queue_mask_recv):

        def recv_request(data):
            pass

        def proc_mask(response):
            queue_label_send.put(response)
            self.logger.info('Mask request sent')
            return queue_mask_recv.get()

        def send_mask(response):
            data = [
                self.init_address,
                self.init_port,
                ParallelCommDetect.MSG_TYPE_SEND_MASK,
                ParallelCommDetect.MSG_DATA_MSK,
                response.get('response_mask', None),
                response.get('response_embedding', None),
                response.get('response_reward', None),
                response.get('response_label', None)
            ]

            self.logger.info(f'Sending mask response: {data}')
            self.client(data, str(response['sender_address']), int(response['sender_port']))


        request = recv_request(data)
        self.logger.info(Fore.CYAN + f'Received mask request: {request}')

        mask_response = proc_mask(request)
        self.logger.info(Fore.CYAN + f'Processed mask response: {mask_response}')

        if mask_response is not None:
            send_mask(mask_response)
    
    def received_mask(self, buffer):
        """
        Unpacks received mask response and appends to list of masks.
        
        Args:
            buffer: A list containing mask information.
        """

        ret = {}

        msg_data = buffer[ParallelCommDetect.META_INF_IDX_MSG_DATA]

        if msg_data == ParallelCommDetect.MSG_DATA_NULL:
            pass
        elif msg_data == ParallelCommDetect.MSG_DATA_MSK:
            #received_mask, received_label, received_reward = buffer[4:7]
            #ip, port = buffer[0], buffer[1]

            ret = {
                'mask'       : buffer[4],
                'embedding'  : buffer[5],
                'reward'     : buffer[6],
                'label'      : buffer[7],
                'ip'         : buffer[0],
                'port'       : buffer[1]
            }

            self.masks.append(ret)

            self.logger.info(Fore.MAGENTA + f"Mask: {ret['mask']}\nEmbedding: {ret['embedding']}]\nReward: {ret['reward']}\nLabel: {ret['label']}\nIP: {ret['ip']}\nPort: {ret['port']}")


    '''def request(self, data, queue_label_send, queue_mask_recv):
        """
        Process a mask request and send a mask response to a querying agent.
        
        Args:
            data: A list containing mask request information.
            queue_label_send: A shared memory queue to send an embedding to be converted by the agent module.
            queue_mask_recv: A shared memory queue to receive a converted mask from the agent module.
        """
        print(Fore.WHITE + 'Mask request:')
        print(data)
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Embedding: {data[4]}')

        mask_req = {'response': True, 'embedding': data[4], 'address': data[0], 'port': data[1]}
        
        # Get the label to mask conversion
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
        self.logger.info(f'Processed mask response: {mask_resp}')

        if mask_resp['response']:
            mask_resp['reward'] = self.knowledge_base.get(tuple(mask_resp['embedding'].tolist()), 0.0)

            # Send the mask response back to the querying agent
            self.send_mask(mask_resp)'''
    

    
    ###############################################################################
    ### Communication module core methods.

    def client(self, data, address, port, is_query=False):
        """
        Client implementation. Begins a TCP connection secured using SSL/TLS to a trusted server ip-port. Attempts to send the serialized bytes.
        
        Args:
            data: A list to be sent to another agent.
            address: The ip of the destination.
            port: The port of the destination.
            is_query: Data to send is a query.
        """
        _data = pickle.dumps(data, protocol=5)
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)

                # If this is a query then drop with a random chance. (For testing communication dropout)
                '''if is_query and int(np.random.choice(2, 1, p=[self.dropout, 1 - self.dropout])) == 1:  # Condition to simulate % communication loss
                    pass
            
                else:'''
                print('ATTEMPTING CONNECTION TO DESTINATION')
                sock.connect((address, port))

                #context = ssl.create_default_context()
                #context.load_cert_chain(ParallelComm.CERTPATH, ParallelComm.KEYPATH)
                #sock = context.wrap_socket(sock, server_side=False)

                _data = struct.pack('>I', len(_data)) + _data
                sock.sendall(_data)
                self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table. This is used to purge potentially dead addresses.
            #try:
            #    self.query_list.remove((address, port))
            #    self.world_size.value = len(self.query_list) + 1
            #except:
            #    pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')

    def event_handler(self, data, queue_mask_recv, queue_label_send):
        ### EVENT HANDLING
        # Agent is sending a query table
        if data[ParallelCommDetect.META_INF_IDX_MSG_TYPE] == ParallelCommDetect.MSG_TYPE_SEND_TABLE:
            self.logger.info(Fore.CYAN + 'Data is a query table')
            self.update_params(data)
            for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')

        # An agent is sending a query
        elif data[ParallelCommDetect.META_INF_IDX_MSG_TYPE] == ParallelCommDetect.MSG_TYPE_SEND_QUERY:
            self.logger.info(Fore.CYAN + 'Data is a query')
            self.received_query(data, queue_label_send, queue_mask_recv)

        # An agent is sending a mask
        elif data[ParallelCommDetect.META_INF_IDX_MSG_TYPE] == ParallelCommDetect.MSG_TYPE_SEND_MASK:
            self.logger.info(Fore.CYAN + 'Data is mask response')
            self.received_mask(data)

        print('\n')

    def server(self, queue_mask_recv, queue_label_send):
        """
        Implementation of the listening server. Binds a socket to a specified port and listens for incoming communication requests. If the connection is accepted using SSL/TLS handshake
        then the connection is secured and data is transferred. Once the data is received, an event is triggered based on the contents of the deserialised data.

        Args:
            knowledge_base: A shared memory dictionary containing the embedding-reward pairs for all observed tasks embeddings.
            queue_mask: A shared memory queue to send received masks to the agent module.
            queue_mask_recv: A shared memory queue to receive masks from the agent module.
            queue_label_send: A shared memory queue to send embeddings to be converted by the agent module.
            world_size: A shared memory variable indicating the size of the network in terms of how many nodes there are.
        """

        def _recvall(conn, n):
            data = bytearray()
            while len(data) < n:
                packet = conn.recv(n - len(data))
                if not packet: return None
                data.extend(packet)
            return data

        def recv_msg(conn):
            msg_length = _recvall(conn, 4)
            if not msg_length: return None
            msg = struct.unpack('>I', msg_length)[0]
            return _recvall(conn, msg)

        #context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        #context.load_cert_chain(certfile='certificates/certificate.pem', keyfile='certificates/key.pem')

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #s = ssl.wrap_socket(s, server_side=True, keyfile=ParallelCommV1.KEYPATH, certfile=ParallelCommV1.CERTPATH, ssl_version=ssl.PROTOCOL_TLSv1_2)    # Uncomment to enable SSL/TLS security. Currently breaks when transferring masks.

        # Bind the socket to the chosen address-port and start listening for connections
        if self.init_address == '127.0.0.1': sock.bind(('127.0.0.1', self.init_port))
        else: sock.bind(('', self.init_port))

        # Set backlog to the world size
        sock.listen(self.world_size.value)
        print('SERVER STARTED')

        while True:
            # Accept the connection
            conn, addr = sock.accept()
            #conn = context.wrap_socket(conn, server_side=True)

            with conn:
                self.logger.info(Fore.CYAN + f'Connected by {addr}')
                while True:
                    try:
                        # Receive the data onto a buffer
                        data = recv_msg(conn)
                        if not data: break
                        data = pickle.loads(data)
                        self.logger.info(Fore.CYAN + f'Received {data!r}')

                        # Potentially deprecated events. Leaving these here incase we need to use some of the code in the future.
                        '''
                        # Agent attempting to join the network
                        #if data[ParallelCommV1.META_INF_IDX_MSG_TYPE] == ParallelCommV1.MSG_TYPE_SEND_JOIN:
                        #    t_validation = mpd.Pool(processes=1)
                        #    t_validation.apply_async(self.recv_join_net, (data, ))
                        #    self.logger.info(Fore.CYAN + 'Data is a join request')
                        #    t_validation.close()
                        #    del t_validation

                        #    for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')
                        #    print(f'{Fore.GREEN}{self.world_size}')

                        # Another agent is leaving the network
                        #elif data[ParallelCommV1.META_INF_IDX_MSG_TYPE] == ParallelCommV1.MSG_TYPE_SEND_LEAVE:
                        #    t_leave = mpd.Pool(processes=1)
                        #    _address, _port = t_leave.apply_async(self.recv_exit_net, (data)).get()

                        #    # Remove the ip-port from the query table for the agent that is leaving
                        #    try: self.query_list.remove(next((x for x in self.query_list if x[0] == addr[0] and x[1] == addr[1])))  # Finds the next Address object with inet4==address and port==port and removes it from the query table.
                        #    except: pass

                        #    self.logger.info(Fore.CYAN + 'Data is a leave request')
                        #    t_leave.close()
                        #    del t_leave
                        '''

                        # Handle connection
                        handler = mpd.Pool(processes=1)
                        handler.apply_async(self.event_handler, (data, queue_mask_recv, queue_label_send))

                    # Handles a connection reset by peer error that I've noticed when running the code. For now it just catches 
                    # the exception and moves on to the next connection.
                    except socket.error as e:
                        if e.errno != ECONNRESET: raise
                        print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                        pass

    # Main loop + listening server initialisation
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv):
        """
        Main communication loop. Sets up the server process, sends out a join request to a known network and begins sending queries to agents in the network.
        Distributes queues for interactions between the communication and agent modules.
        
        Args:
            queue_label: A shared memory queue to send embeddings from the agent module to the communication module.
            queue_mask: A shared memory queue to send masks from the communication module to the agent module.
            queue_label_send: A shared memory queue to send embeddings from the communication module to the agent module for conversion.
            queue_loop: A shared memory queue to send iteration state variables from the communication module to the agent module. Currently the only variable that is sent over the queue is the agent module's iteration value.
            knowledge_base: A shared memory dictionary containing embedding-reward pairs for all observed tasks.
            world_size: A shared memory integer with value of the number of known agents in the network.
        """

        # Initialise the listening server
        p_server = mp.Process(target=self.server, args=(queue_mask_recv, queue_label_send))
        p_server.start()

        # Attempt to join an existing network.
        # TODO: Will have to figure how to heal a severed connection with the new method.
        #self.logger.info(Fore.GREEN + 'Attempting to discover peers from reference...')
        #p_discover = mp.Process(target=self.send_join_net)
        #p_discover.start()

        time.sleep(1)

        # Initialise the client loop
        while True:
            # Attempt to connect to reference agent and get latest table. If the query table is reduced to original state then try to reconnect previous agents
            # using the reference table.
            # Unless there is no reference.
            #try:
                print()
                self.logger.info(Fore.GREEN + f'Knowledge base in this iteration:')
                #for key, val in self.knowledge_base.items(): self.logger.info(f'{key[0:5]} : {val}')
                self.logger.info(Fore.GREEN + f'World size in comm: {self.world_size.value}')
                self.logger.info(Fore.GREEN + f'Query table in this iteration:')
                for addr in self.query_list: print(addr[0], addr[1])
                #self.logger.info(Fore.GREEN + f'Reference table this iteration:')
                #for addr in self.reference_list: print(addr[0], addr[1])


                # Block operation until an embedding is received to query for
                dict_to_query = queue_label.get()
                print(Fore.GREEN + f'RECEIVED DICT TO QUERY: {dict_to_query}')


                # Get the world size based on the number of addresses in the query list
                self.world_size.value = len(self.query_list) + 1


                # Send out a query when shell iterations matches mask interval if the agent is working on a task
                if self.world_size.value > 1:
                    if dict_to_query['reward'] < 0.9:
                        self.send_query(dict_to_query, queue_mask)



            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            #except (SystemExit, KeyboardInterrupt) as e:                           # Uncomment to enable the keyboard interrupt and system exit handling
            #    p_server.close()
            #    #p_discover.close()
            #    #self.send_exit_net()
            #    sys.exit()
    
    # Method for starting parallel comm process
    def parallel(self, manager):
        """
        Parallelization method for starting the communication loop inside a separate process.

        Args:
            manager: A multiprocessing Manager instance for creating shared resources.

        Returns:
            Four multiprocessing Queues for communication between processes.
        """

        queue_label = manager.Queue()
        queue_mask = manager.Queue()
        queue_label_send = manager.Queue()
        queue_mask_recv = manager.Queue()

        # Start the communication loop in a separate process
        p_client = mp.Process(
            target=self.communication,
            args=(queue_label, queue_mask, queue_label_send, queue_mask_recv)
        )
        p_client.start()

        return queue_label, queue_mask, queue_label_send, queue_mask_recv
    

class ParallelCommDetectEval(object):
    ### COMMUNCIATION MODULE HYPERPARAMETERS
    # DETECT MODULE CONSTANTS
    # Threshold for embedding/tasklabel distance (similarity)
    # This should be taken from the detect module eventually
    THRESHOLD = 0.0

    # SSL/TLS PATHS
    # Paths to the SSL/TLS certificates and key
    CERTPATH = 'certificates/certificate.pem'
    KEYPATH = 'certificates/key.pem'

    
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
    MSG_TYPE_SEND_JOIN = 4
    MSG_TYPE_SEND_LEAVE = 5
    MSG_TYPE_SEND_TABLE = 6

    # message data (META_INF_IDX_MSG_DATA) values
    MSG_DATA_NULL = 0 # an empty message
    MSG_DATA_QUERY = 1
    MSG_DATA_MSK_REQ = 2
    MSG_DATA_MSK = 3
    MSG_DATA_META = 4

    # Task label size can be replaced with the embedding size.
    def __init__(self, embd_dim, mask_dim, logger, init_port, reference, seen_tasks, manager, localhost, mode, dropout, threshold):
        super(ParallelCommDetectEval, self).__init__()
        self.embd_dim = embd_dim            # Dimensions of the the embedding
        self.mask_dim = mask_dim            # Dimensions of the mask for use in buffers. May no longer be needed
        self.logger = logger                # Logger object for logging CLI outputs.
        self.mode = mode                    # Communication operation mode. Currently only ondemand knowledge is implemented
        self.threshold = threshold

        # Address and port for this agent
        if localhost: self.init_address = '127.0.0.1'
        else: self.init_address = self.init_address = urllib.request.urlopen('https://v4.ident.me').read().decode('utf8') # Use this to get the public ip of the host server.
        self.init_port = int(init_port)

        # Shared memory variables. Make these into attributes of the communication module to make it easier to use across the many sub processes of the module.
        self.query_list = manager.list([item for item in reference if item != (self.init_address, self.init_port)]) # manager.list(reference)
        self.reference_list = manager.list(deepcopy(self.query_list))   # Weird thing was happening here so used deepcopy to recreate the manager ListProxy with the addresses.
        self.knowledge_base = seen_tasks
        self.world_size = manager.Value('i', len(self.reference_list))
        self.masks = manager.list()


        

        # COMMUNICATION DROPOUT
        # Used to simulate percentage communication dropout in the network. Currently only limits the amount of queries and not a total communication blackout.
        self.dropout = dropout  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout


        
        # LOGGING INCOMING CONNECTIONS FOR VISUALISATION TOOL
        #access_token = '8ad435f2bc1b48'
        #self.handler = ipinfo.getHandler(access_token)

        #details = self.handler.getDetails(self.init_address)
        #self.connections = [['ip', 'port', 'country', 'city', 'region', 'timezone', 'postal', 'lat', 'long', 'timestamp']]
        #self.connections.append([details.ip, self.init_port, details.country, details.city, details.region, details.timezone, details.postal, details.latitude, details.longitude, time.time()])

        # Debugging CLI outputs
        self.debug_output()

    def debug_output(self):
        print('Query table:')
        for addr in self.query_list: print(addr[0], addr[1])

        print('\nReference table:')
        for addr in self.reference_list: print(addr[0], addr[1])

        print(f'\nlistening server params ->\naddress: {self.init_address}\nport: {self.init_port}\n')
        print(f'mask size: {self.mask_dim}')
        print(f'embedding size: {self.embd_dim}\n')

    def _null_message(self, msg):
        """
        Checks if a message contains null i.e, no data.

        Args:
            msg: A list received from another agent.

        Returns:
            A boolean indicating whether A list contains null data.
        """

        # check whether message sent denotes or is none.
        if bool(msg[ParallelCommDetectEval.META_INF_IDX_MSG_DATA] == ParallelCommDetectEval.MSG_DATA_NULL):
            return True

        else:
            return False

    '''###############################################################################
    ### 
    def proc_meta(self, other_agent_req):
        """
        Processes a query for an embedding and produces a response to send back to the requesting agent.
        
        Args:
            other_agent_req: A dictionary containing the information for the query request.
        
        Returns:
            meta_response: A dictionary containing the response information.
        """

        if other_agent_req is not None:
            other_agent_req['response'] = False
            embedding = other_agent_req['embedding'].detach().cpu().numpy()
            sender_reward = other_agent_req['sender_reward']

            # Iterate through the knowledge base and compute the distances
            for tlabel, treward in self.knowledge_base.items():
                if treward > 0.0:
                    if 0.9 * treward > sender_reward:
                        tdist = float(torch.linalg.vector_norm(embedding - torch.squeeze(torch.tensor(tlabel))))
                        self.logger.info(f'{tdist} meta distance')
                        if tdist <= self.threshold:
                            other_agent_req['response'] = True
                            other_agent_req['reward'] = treward
                            other_agent_req['dist'] = tdist
                            other_agent_req['resp_embedding'] = torch.squeeze(torch.tensor(tlabel))

        return other_agent_req
    def send_meta(self, meta_resp):
        if meta_resp and meta_resp['response']:
            data = [
                self.init_address,
                self.init_port,
                ParallelCommDetect.MSG_TYPE_SEND_META,
                ParallelCommDetect.MSG_DATA_META,
                meta_resp.get('reward', None),
                meta_resp.get('dist', None),
                meta_resp.get('resp_embedding', None)
            ]

            self.client(data, str(meta_resp['sender_address']), int(meta_resp['sender_port']))
    def send_to_agent(self, queue_mask):
        self.logger.info(Fore.CYAN + 'Data is a mask')
        # Unpack the received data
        received_masks = self.recv_masks(data)
        received_mask, received_label, received_reward, ip, port = self.recv_mask(data)

        self.logger.info(f'{received_mask, received_label, received_reward, ip, port}')
        # Send the reeceived information back to the agent process if condition met
        if received_mask is not None and received_label is not None and received_reward is not None:
            self.logger.info('Sending mask data to agent')
            queue_mask.put(self.masks)'''

    ###############################################################################
    ### Query send and recv methods.
    def send_query(self, dict_to_query, queue_mask):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            dict_to_query: A dictionary with structure {'task_emb': <tensor>, 'reward': <float>, 'ground_truth': <tensor>}
        """

        embedding = dict_to_query['task_emb']
        reward = dict_to_query['reward']
        label = dict_to_query['ground_truth']   # For validation purposes. We can feed this into pick_meta() to perform validation and log false positives.

        # Prepare the data for sending
        if embedding is None:
            data = [self.init_address, self.init_port, ParallelCommDetectEval.MSG_TYPE_SEND_QUERY, ParallelCommDetectEval.MSG_DATA_NULL]
        else:
            data = [self.init_address, self.init_port, ParallelCommDetectEval.MSG_TYPE_SEND_QUERY, ParallelCommDetectEval.MSG_DATA_QUERY, embedding, reward]

        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in list(self.query_list):
            self.client(data, addr[0], addr[1], is_query=True)

        time.sleep(0.2)
        self.logger.info('Sending mask data to agent')
        queue_mask.put(list(self.masks))

    def update_params(self, data):
        _query_list = data[3]
        _query_list.reverse()

        # Insert addresses from _query_list into query_list if they are not already present
        for addr in _query_list:
            if addr not in self.query_list:
                self.query_list.insert(0, addr)

        self.world_size.value = len(self.query_list) + 1

    def received_query(self, data, queue_label_send, queue_mask_recv):
        """
        Event handler for receiving a query from another agent. Unpacks the buffer received from another agent, processes the request and sends a mask response if conditions met.
        
        Args:
            data: A list received from another agent.
        """

        # Query to mask response pipeline
        def recv_query(buffer):
            """
            Unpacks the data buffer received from another agent for a query.
            
            Args:
                buffer: A list received from another agent.
                
            Returns:
                ret: A dictionary containing the unpacked data.
            """
            sender_address = str(buffer[ParallelCommDetectEval.META_INF_IDX_ADDRESS])
            sender_port = int(buffer[ParallelCommDetectEval.META_INF_IDX_PORT])
            embedding = buffer[ParallelCommDetectEval.META_INF_IDX_TASK_SZ]
            sender_reward = buffer[-1]

            # Create a dictionary with the unpacked data
            ret = {
                'sender_address': sender_address,
                'sender_port': sender_port,
                'sender_embedding': embedding,
                'sender_reward': sender_reward
            }

            # Handle when receiving a query from an unknown agent
            if (sender_address, sender_port) not in self.query_list:
                self.client([self.init_address, self.init_port, ParallelCommDetectEval.MSG_TYPE_SEND_TABLE, list(self.query_list)], sender_address, sender_port)
                self.query_list.append((sender_address, sender_port))

            # Refresh the world_size value
            self.world_size.value = len(self.query_list) + 1

            return ret
        
        def proc_mask(query):
            """
            Find the most similar task record and get the internal task id if any satisfying entries found. Create response dictionary and return otherwise return NoneType.
            
            Args:
                query: A dictionary consisting of the response information to send to a specific agent.

            Returns:
                The mask_req dictionary with the converted mask now included.
            """
            target_embedding = query['sender_embedding']
            target_reward = query['sender_reward']

            self.logger.info(f'Sender emb: {target_embedding}, Sender rw: {target_reward}')

            # Extract embeddings and rewards from the data_dict
            embeddings = []
            rewards = []

            # Get embedding-rewards for any entry where the reward condition 0.9 * reward > sender_reward is met.
            for value in self.knowledge_base.values():
                if 'task_emb' in value:
                    known_embedding = value['task_emb']
                    known_reward = value['reward']

                    # Apply the condition: 0.9 * known_reward > target_reward
                    if 0.9 * known_reward > target_reward:
                        embeddings.append(known_embedding)
                        rewards.append(known_reward)

            self.logger.info(f'embeddings: {embeddings}')
            self.logger.info(f'rewards: {rewards}')
                        

            # If any are found from previous condition then we want to compute embedding distances and get the entry with the most similarity under the threshold.
            if len(embeddings) > 0:
                try:
                    # Convert to a list of PyTorch tensors
                    embeddings = torch.stack(embeddings)
                    #rewards = torch.stack(rewards)

                    # Calculate cosine similarities using PyTorch
                    similarities = F.cosine_similarity(embeddings, target_embedding.unsqueeze(0))
                    self.logger.info(Fore.LIGHTGREEN_EX + f'SIMILARITIES: {similarities}, {type(similarities)}')
                    
                    # Find the index of the closest entry
                    closest_index = torch.argmax(similarities).item()

                    # Get the corresponding key
                    closest_key = list(self.knowledge_base.keys())[closest_index]

                    # Get the closest similarity
                    closest_similarity = torch.min(similarities).item()

                    # Define a threshold for cosine similarity (adjust as needed)
                    cosine_similarity_threshold = 0.4  # Example threshold

                    print(closest_key)

                    # Check if the closest similarity is above the threshold
                    if closest_similarity >= cosine_similarity_threshold:
                        # Get information from the registry for the closest key (internal task id)
                        query['response_reward'] = self.knowledge_base[closest_key]['reward']
                        query['response_similarity'] = closest_similarity  # Convert to a Python float
                        query['response_task_id'] = closest_key
                        query['response_embedding'] = self.knowledge_base[closest_key]['task_emb']
                        query['response_label'] = self.knowledge_base[closest_key]['ground_truth']

                        '''at this point query is contains the following:
                        query = {
                        Query:
                        sender_address  (String)
                        sender_port (Integer)
                        sender_embedding    (Tensor)
                        sender_reward   (Float)
                        
                        Response:
                        response_reward (Float)
                        response_dist   (Float)
                        response_task_id    (Integer)
                        response_embedding  (Tensor)
                        response_label  (Tensor)
                        
                        response_mask   (Tensor) (Added in the convert process in the agent)
                        }'''

                        self.logger.info(Fore.GREEN + 'Found valid entry. Requesting agent to get mask from network.')
                        queue_label_send.put(query)
                        self.logger.info('Mask request sent')
                        return queue_mask_recv.get()
                    
                    else:
                        self.logger.info(Fore.GREEN + "No entries satisfying the condition found.")
                        return None
                    
                except Exception as e:
                    traceback.print_exc()
        
        def send_mask(response):
            """
            Sends a mask response to a specific agent.
            
            Args:
                mask_resp: A dictionary consisting of the information to send to a specific agent.    
            """
            data = [
                self.init_address,
                self.init_port,
                ParallelCommDetectEval.MSG_TYPE_SEND_MASK,
                ParallelCommDetectEval.MSG_DATA_MSK,
                response.get('response_mask', None),
                response.get('response_embedding', None),
                response.get('response_reward', None),
                response.get('response_label', None)
            ]

            self.logger.info(f'Sending mask response: {data}')
            self.client(data, str(response['sender_address']), int(response['sender_port']))
   

        # Unpack the query from the other agent
        query = recv_query(data)
        self.logger.info(Fore.CYAN + f'Received query: {query}')

        # Get the mask with the most task similarity, if any such mask exists in the network.
        response = proc_mask(query)
        self.logger.info(Fore.CYAN + f'Processed mask request: {response}')

        # Send mask to querying agent if response is not NoneType
        if response is not None:
            send_mask(response)
    
    def received_mask(self, buffer):
        """
        Unpacks received mask response and appends to list of masks.
        
        Args:
            buffer: A list containing mask information.
        """

        ret = {}

        msg_data = buffer[ParallelCommDetectEval.META_INF_IDX_MSG_DATA]

        if msg_data == ParallelCommDetectEval.MSG_DATA_NULL:
            pass
        elif msg_data == ParallelCommDetectEval.MSG_DATA_MSK:
            #received_mask, received_label, received_reward = buffer[4:7]
            #ip, port = buffer[0], buffer[1]

            ret = {
                'mask'       : buffer[4],
                'embedding'  : buffer[5],
                'reward'     : buffer[6],
                'label'      : buffer[7],
                'ip'         : buffer[0],
                'port'       : buffer[1]
            }

            self.masks.append(ret)

            self.logger.info(Fore.MAGENTA + f"Mask: {ret['mask']}\nEmbedding: {ret['embedding']}]\nReward: {ret['reward']}\nLabel: {ret['label']}\nIP: {ret['ip']}\nPort: {ret['port']}")


    '''def request(self, data, queue_label_send, queue_mask_recv):
        """
        Process a mask request and send a mask response to a querying agent.
        
        Args:
            data: A list containing mask request information.
            queue_label_send: A shared memory queue to send an embedding to be converted by the agent module.
            queue_mask_recv: A shared memory queue to receive a converted mask from the agent module.
        """
        print(Fore.WHITE + 'Mask request:')
        print(data)
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Embedding: {data[4]}')

        mask_req = {'response': True, 'embedding': data[4], 'address': data[0], 'port': data[1]}
        
        # Get the label to mask conversion
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
        self.logger.info(f'Processed mask response: {mask_resp}')

        if mask_resp['response']:
            mask_resp['reward'] = self.knowledge_base.get(tuple(mask_resp['embedding'].tolist()), 0.0)

            # Send the mask response back to the querying agent
            self.send_mask(mask_resp)'''
    

    
    ###############################################################################
    ### Communication module core methods.

    def client(self, data, address, port, is_query=False):
        """
        Client implementation. Begins a TCP connection secured using SSL/TLS to a trusted server ip-port. Attempts to send the serialized bytes.
        
        Args:
            data: A list to be sent to another agent.
            address: The ip of the destination.
            port: The port of the destination.
            is_query: Data to send is a query.
        """
        _data = pickle.dumps(data, protocol=5)
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)

                # If this is a query then drop with a random chance. (For testing communication dropout)
                '''if is_query and int(np.random.choice(2, 1, p=[self.dropout, 1 - self.dropout])) == 1:  # Condition to simulate % communication loss
                    pass
            
                else:'''
                print('ATTEMPTING CONNECTION TO DESTINATION')
                sock.connect((address, port))

                #context = ssl.create_default_context()
                #context.load_cert_chain(ParallelComm.CERTPATH, ParallelComm.KEYPATH)
                #sock = context.wrap_socket(sock, server_side=False)

                _data = struct.pack('>I', len(_data)) + _data
                sock.sendall(_data)
                self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table. This is used to purge potentially dead addresses.
            #try:
            #    self.query_list.remove((address, port))
            #    self.world_size.value = len(self.query_list) + 1
            #except:
            #    pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')

    def event_handler(self, data, queue_mask_recv, queue_label_send):
        ### EVENT HANDLING
        # Agent is sending a query table
        if data[ParallelCommDetectEval.META_INF_IDX_MSG_TYPE] == ParallelCommDetectEval.MSG_TYPE_SEND_TABLE:
            self.logger.info(Fore.CYAN + 'Data is a query table')
            self.update_params(data)
            for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')

        # An agent is sending a query
        elif data[ParallelCommDetectEval.META_INF_IDX_MSG_TYPE] == ParallelCommDetectEval.MSG_TYPE_SEND_QUERY:
            self.logger.info(Fore.CYAN + 'Data is a query')
            self.received_query(data, queue_label_send, queue_mask_recv)

        # An agent is sending a mask
        elif data[ParallelCommDetectEval.META_INF_IDX_MSG_TYPE] == ParallelCommDetectEval.MSG_TYPE_SEND_MASK:
            self.logger.info(Fore.CYAN + 'Data is mask response')
            self.received_mask(data)

        print('\n')

    def server(self, queue_mask_recv, queue_label_send):
        """
        Implementation of the listening server. Binds a socket to a specified port and listens for incoming communication requests. If the connection is accepted using SSL/TLS handshake
        then the connection is secured and data is transferred. Once the data is received, an event is triggered based on the contents of the deserialised data.

        Args:
            knowledge_base: A shared memory dictionary containing the embedding-reward pairs for all observed tasks embeddings.
            queue_mask: A shared memory queue to send received masks to the agent module.
            queue_mask_recv: A shared memory queue to receive masks from the agent module.
            queue_label_send: A shared memory queue to send embeddings to be converted by the agent module.
            world_size: A shared memory variable indicating the size of the network in terms of how many nodes there are.
        """

        def _recvall(conn, n):
            data = bytearray()
            while len(data) < n:
                packet = conn.recv(n - len(data))
                if not packet: return None
                data.extend(packet)
            return data

        def recv_msg(conn):
            msg_length = _recvall(conn, 4)
            if not msg_length: return None
            msg = struct.unpack('>I', msg_length)[0]
            return _recvall(conn, msg)

        #context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        #context.load_cert_chain(certfile='certificates/certificate.pem', keyfile='certificates/key.pem')

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #s = ssl.wrap_socket(s, server_side=True, keyfile=ParallelCommV1.KEYPATH, certfile=ParallelCommV1.CERTPATH, ssl_version=ssl.PROTOCOL_TLSv1_2)    # Uncomment to enable SSL/TLS security. Currently breaks when transferring masks.

        # Bind the socket to the chosen address-port and start listening for connections
        if self.init_address == '127.0.0.1': sock.bind(('127.0.0.1', self.init_port))
        else: sock.bind(('', self.init_port))

        # Set backlog to the world size
        sock.listen(self.world_size.value)
        print('SERVER STARTED')

        while True:
            # Accept the connection
            conn, addr = sock.accept()
            #conn = context.wrap_socket(conn, server_side=True)

            with conn:
                self.logger.info(Fore.CYAN + f'Connected by {addr}')
                while True:
                    try:
                        # Receive the data onto a buffer
                        data = recv_msg(conn)
                        if not data: break
                        data = pickle.loads(data)
                        self.logger.info(Fore.CYAN + f'Received {data!r}')

                        # Potentially deprecated events. Leaving these here incase we need to use some of the code in the future.
                        '''
                        # Agent attempting to join the network
                        #if data[ParallelCommV1.META_INF_IDX_MSG_TYPE] == ParallelCommV1.MSG_TYPE_SEND_JOIN:
                        #    t_validation = mpd.Pool(processes=1)
                        #    t_validation.apply_async(self.recv_join_net, (data, ))
                        #    self.logger.info(Fore.CYAN + 'Data is a join request')
                        #    t_validation.close()
                        #    del t_validation

                        #    for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')
                        #    print(f'{Fore.GREEN}{self.world_size}')

                        # Another agent is leaving the network
                        #elif data[ParallelCommV1.META_INF_IDX_MSG_TYPE] == ParallelCommV1.MSG_TYPE_SEND_LEAVE:
                        #    t_leave = mpd.Pool(processes=1)
                        #    _address, _port = t_leave.apply_async(self.recv_exit_net, (data)).get()

                        #    # Remove the ip-port from the query table for the agent that is leaving
                        #    try: self.query_list.remove(next((x for x in self.query_list if x[0] == addr[0] and x[1] == addr[1])))  # Finds the next Address object with inet4==address and port==port and removes it from the query table.
                        #    except: pass

                        #    self.logger.info(Fore.CYAN + 'Data is a leave request')
                        #    t_leave.close()
                        #    del t_leave
                        '''

                        # Handle connection
                        handler = mpd.Pool(processes=1)
                        handler.apply_async(self.event_handler, (data, queue_mask_recv, queue_label_send))

                    # Handles a connection reset by peer error that I've noticed when running the code. For now it just catches 
                    # the exception and moves on to the next connection.
                    except socket.error as e:
                        if e.errno != ECONNRESET: raise
                        print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                        pass

    # Main loop + listening server initialisation
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv):
        """
        Main communication loop. Sets up the server process, sends out a join request to a known network and begins sending queries to agents in the network.
        Distributes queues for interactions between the communication and agent modules.
        
        Args:
            queue_label: A shared memory queue to send embeddings from the agent module to the communication module.
            queue_mask: A shared memory queue to send masks from the communication module to the agent module.
            queue_label_send: A shared memory queue to send embeddings from the communication module to the agent module for conversion.
            queue_loop: A shared memory queue to send iteration state variables from the communication module to the agent module. Currently the only variable that is sent over the queue is the agent module's iteration value.
            knowledge_base: A shared memory dictionary containing embedding-reward pairs for all observed tasks.
            world_size: A shared memory integer with value of the number of known agents in the network.
        """

        # Initialise the listening server
        p_server = mp.Process(target=self.server, args=(queue_mask_recv, queue_label_send))
        p_server.start()

        # Attempt to join an existing network.
        # TODO: Will have to figure how to heal a severed connection with the new method.
        #self.logger.info(Fore.GREEN + 'Attempting to discover peers from reference...')
        #p_discover = mp.Process(target=self.send_join_net)
        #p_discover.start()

        time.sleep(1)

        # Initialise the client loop
        while True:
            # Attempt to connect to reference agent and get latest table. If the query table is reduced to original state then try to reconnect previous agents
            # using the reference table.
            # Unless there is no reference.
            #try:
                print()
                self.logger.info(Fore.GREEN + f'Knowledge base in this iteration:')
                #for key, val in self.knowledge_base.items(): self.logger.info(f'{key[0:5]} : {val}')
                self.logger.info(Fore.GREEN + f'World size in comm: {self.world_size.value}')
                self.logger.info(Fore.GREEN + f'Query table in this iteration:')
                for addr in self.query_list: print(addr[0], addr[1])
                #self.logger.info(Fore.GREEN + f'Reference table this iteration:')
                #for addr in self.reference_list: print(addr[0], addr[1])


                # Block operation until an embedding is received to query for
                dict_to_query = queue_label.get()
                print(Fore.GREEN + f'RECEIVED DICT TO QUERY: {dict_to_query}')


                # Get the world size based on the number of addresses in the query list
                self.world_size.value = len(self.query_list) + 1


                # Send out a query when shell iterations matches mask interval if the agent is working on a task
                if self.world_size.value > 1:
                    if dict_to_query['reward'] < 0.9:
                        self.send_query(dict_to_query, queue_mask)



            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            #except (SystemExit, KeyboardInterrupt) as e:                           # Uncomment to enable the keyboard interrupt and system exit handling
            #    p_server.close()
            #    #p_discover.close()
            #    #self.send_exit_net()
            #    sys.exit()
    
    # Method for starting parallel comm process
    def parallel(self, manager):
        """
        Parallelization method for starting the communication loop inside a separate process.

        Args:
            manager: A multiprocessing Manager instance for creating shared resources.

        Returns:
            Four multiprocessing Queues for communication between processes.
        """

        queue_label = manager.Queue()
        queue_mask = manager.Queue()
        queue_label_send = manager.Queue()
        queue_mask_recv = manager.Queue()

        # Start the communication loop in a separate process
        p_client = mp.Process(
            target=self.communication,
            args=(queue_label, queue_mask, queue_label_send, queue_mask_recv)
        )
        p_client.start()

        return queue_label, queue_mask, queue_label_send, queue_mask_recv