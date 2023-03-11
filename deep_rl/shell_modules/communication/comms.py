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

import numpy as np
import torch
import traceback


class ParallelComm(object):
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
    def __init__(self, num_agents, embd_dim, mask_dim, logger, init_port, reference, knowledge_base, manager, localhost, mode, dropout):
        super(ParallelComm, self).__init__()
        self.embd_dim = embd_dim            # Dimensions of the the embedding
        self.mask_dim = mask_dim            # Dimensions of the mask for use in buffers. May no longer be needed
        self.logger = logger                # Logger object for logging CLI outputs.
        self.mode = mode                    # Communication operation mode. Currently only ondemand knowledge is implemented

        # Address and port for this agent
        if localhost: self.init_address = '127.0.0.1'
        else: self.init_address = self.init_address = urllib.request.urlopen('https://v4.ident.me').read().decode('utf8') # Use this to get the public ip of the host server.
        self.init_port = int(init_port)

        # Shared memory variables. Make these into attributes of the communication module to make it easier to use across the many sub processes of the module.
        self.query_list = manager.list([item for item in reference if item != (self.init_address, self.init_port)]) # manager.list(reference)
        self.reference_list = manager.list(deepcopy(self.query_list))   # Weird thing was happening here so used deepcopy to recreate the manager ListProxy with the addresses.
        self.knowledge_base = knowledge_base
        
        self.world_size = manager.Value('i', num_agents)
        self.metadata = manager.list()

        # COMMUNICATION DROPOUT
        # Used to simulate percentage communication dropout in the network. Currently only limits the amount of queries and not a total communication blackout.
        self.dropout = dropout  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout


        print(type(self.query_list))
        print(type(self.reference_list))

        # For debugging
        print('Query table:')
        for addr in self.query_list: print(addr[0], addr[1])

        print('\nReference table:')
        for addr in self.reference_list: print(addr[0], addr[1])

        print(f'\nlistening server params ->\naddress: {self.init_address}\nport: {self.init_port}\n')
        print('mask size:', self.mask_dim)
        print('embedding size:', self.embd_dim)

    def _null_message(self, msg):
        """
        Checks if a message contains null i.e, no data.

        Args:
            msg: A list received from another agent.

        Returns:
            A boolean indicating whether A list contains null data.
        """

        # check whether message sent denotes or is none.
        if bool(msg[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_NULL):
            return True

        else:
            return False

    # Client used by the server to send responses
    def client(self, data, address, port):
        """
        Client implementation. Begins a TCP connection secured using SSL/TLS to a trusted server ip-port. Attempts to send the serialized bytes.
        
        Args:
            data: A list to be sent to another agent.
            address: The ip of the destination.
            port: The port of the destination.
        """
        _data = pickle.dumps(data, protocol=5)

        # Attempt to send the data a number of times. If successful do not attempt to send again.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)

        try:
            sock.connect((address, port))

            #context = ssl.create_default_context()
            #context.load_cert_chain(ParallelComm.CERTPATH, ParallelComm.KEYPATH)
            #sock_ssl = context.wrap_socket(sock, server_side=False)

            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table
            #try:
            #    self.query_list.remove((address, port))
            #    self.world_size.value = len(self.query_list) + 1
            #except:
            #    pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
        finally: sock.close()
    
    # Modified version of the client used by the send_query function. Has an additional bit of code to handle the mask response before querying the next agent in the query list
    def query_client(self, data, address, port):
        # Attempt to send the data a number of times. If successful do not attempt to send again.
        
        _data = pickle.dumps(data, protocol=5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        
        try:
            if int(np.random.choice(2, 1, p=[self.dropout, 1 - self.dropout])) == 1:  # Condition to simulate % communication loss
                sock.connect((address, port))

                #context = ssl.create_default_context()
                #context.load_cert_chain(ParallelComm.CERTPATH, ParallelComm.KEYPATH)
                #sock = context.wrap_socket(sock, server_side=False)

                _data = struct.pack('>I', len(_data)) + _data
                sock.sendall(_data)
                self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table
            #try:
            #    self.query_list.remove((address, port))
            #    self.world_size.value = len(self.query_list) + 1
            #except:
            #    pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
        finally:
            sock.close()

    ### Query send and recv functions
    def send_query(self, embedding):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            embedding: A torch tensor containing an embedding
        """

        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        #self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        reward = 0.0
        if tuple(embedding.tolist()) in self.knowledge_base:
            reward = self.knowledge_base[tuple(embedding.tolist())]


        if embedding is None:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_QUERY, embedding, reward]

        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in list(self.query_list):
            self.query_client(data, addr[0], addr[1])

        time.sleep(0.2)
        self.pick_meta()
        
    def recv_mask(self, buffer):
        """
        Unpacks a received mask response from another agent.
        
        Args:
            buffer: A list received from another agent.
            best_agent_id: A shared memory variable of type dict() containing a ip-port pair for the best agent.
            
        Returns:
            received_mask: A torch tensor containing the continous mask parameters.
            received_label: A torch tensor containing the embedding.
        """
        
        received_mask = None
        received_label = None
        received_reward = None
        ip = None
        port = None

        if buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_NULL:
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
            received_mask = buffer[4]
            received_label = buffer[5]
            received_reward = buffer[6]
            ip = buffer[0]
            port = buffer[1]

        return received_mask, received_label, received_reward, ip, port


    def recv_query(self, buffer):
        """
        Unpacks the data buffer received from another agent for a query.
        
        Args:
            buffer: A list received from another agent.
            
        Returns:
            ret: A dictionary containing the unpacked data.
        """

        ret = {}
        ret['sender_address'] = str(buffer[ParallelComm.META_INF_IDX_ADDRESS])
        ret['sender_port'] = int(buffer[ParallelComm.META_INF_IDX_PORT])
        #ret['msg_type'] = int(buffer[ParallelComm.META_INF_IDX_MSG_TYPE])
        #ret['msg_data'] = int(buffer[ParallelComm.META_INF_IDX_MSG_DATA])
        ret['embedding'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ]
        ret['sender_reward'] = buffer[-1]

        # Handle when we receive a query from another agent that we do not have in our known list of agents (self.query_list)
        if (ret['sender_address'], ret['sender_port']) not in self.query_list:
            self.client([self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_TABLE, list(self.query_list)], ret['sender_address'], ret['sender_port'])
            self.query_list.append((ret['sender_address'], ret['sender_port']))

        self.world_size.value = len(self.query_list) + 1    # Refresh the world_size value

        return ret
    def proc_meta(self, other_agent_req):
        """
        Processes a query for an embedding and produces a response to send back to the requesting agent.
        
        Args:
            other_agent_req: A dictionary containing the information for the query request.
            knowledge_base: A shared memory variable consisting of a dictionary to store the task embeddings and rewards accumulated.
        
        Returns:
            meta_response: A dictionary containing the response information.
        """

        other_agent_req['response'] = False

        if other_agent_req is not None:
            np_embedding = other_agent_req['embedding'].detach().cpu().numpy()
            sender_reward = other_agent_req['sender_reward']

            # Iterate through the knowledge base and compute the distances
            # If reward greater than 0
            # If distance is less than or equal to threshold
            # response = True (false by default)
            for tlabel, treward in self.knowledge_base.items():
                if treward > np.around(0.0, decimals=6):
                    if 0.9 * round(treward, 6) > sender_reward:
                        tdist = np.sum(abs(np.subtract(np_embedding, np.array(tlabel))))
                        if tdist <= ParallelComm.THRESHOLD:
                            other_agent_req['response'] = True
                            other_agent_req['reward'] = treward         # Reward of the mask this agent has for the task
                            other_agent_req['dist'] = tdist             # Distance between the embedding of this agent's closest mask and the embedding from the querying agent
                            other_agent_req['resp_embedding'] = torch.tensor(tlabel)  # The closest embedding that this agent has to the one that the querying agent has queried for

        # Return the query request
        return other_agent_req
    def send_meta(self, meta_resp):
        if meta_resp:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_META]
            if meta_resp['response']:
                data.append(ParallelComm.MSG_DATA_META)
                data.append(meta_resp.get('reward', None))
                data.append(meta_resp.get('dist', None))
                data.append(meta_resp.get('resp_embedding', None))

            #else:
            #    data.append(ParallelComm.MSG_DATA_NULL)

                self.client(data, str(meta_resp['sender_address']), int(meta_resp['sender_port']))

    def proc_mask(self, mask_req, queue_label_send, queue_mask_recv):
        """
        Processes the mask response to send to another agent.
        
        Args:
            mask_req: A dictionary consisting of the response information to send to a specific agent.
            queue_label_send: A shared memory queue to send an embedding to be converted by the agent module.
            queue_mask_recv: A shared memory queue to receive a converted mask from the agent module.

        Returns:
            The mask_req dictionary with the converted mask now included. 
        """

        if mask_req['response']:
            self.logger.info('Sending mask request to be converted')
            queue_label_send.put((mask_req))
            self.logger.info('Mask request sent')
            return queue_mask_recv.get()        # Return the dictionary with the mask attached

    def send_mask(self, mask_resp):
        """
        Sends a mask response to a specific agent.
        
        Args:
            mask_resp: A dictionary consisting of the information to send to a specific agent.    
        """
        if mask_resp:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_MASK]

            # if response is True then send the mask
            if mask_resp['response']:
                data.append(ParallelComm.MSG_DATA_MSK)
                data.append(mask_resp.get('mask', None))
                data.append(mask_resp.get('embedding', None))
                data.append(mask_resp.get('reward', None))

                self.logger.info(f'Sending mask response: {data}')
                self.client(data, str(mask_resp['address']), int(mask_resp['port']))
            
            # otherwise send a null response
            #else:
            #    data.append(ParallelComm.MSG_DATA_NULL)
            
            
    
    # Event handler wrappers. This is done so the various functions for each event can be run in a single thread.
    def query(self, data):
        """
        Event handler for receiving a query from another agent. Unpacks the buffer received from another agent, processes the request and sends some response if necessary.
        
        Args:
            data: A list received from another agent.
            knowledge_base: A shared memory variable of type dict() containing embedding-reward pairs for task embeddings observed by the agent.    
        """

        # Get the query from the other agent
        other_agent_req = self.recv_query(data)
        self.logger.info(f'Received query: {other_agent_req}')

        # Check if this agent has any knowledge for the task
        meta_resp = self.proc_meta(other_agent_req)
        self.logger.info(f'Processes mask req: {meta_resp}')

        self.send_meta(meta_resp)
    def add_meta(self, data):
        print(Fore.YELLOW + 'Metadata:')
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Reward: {data[4]}')
        print(f'Distance: {data[5]}')
        print(f'Embedding: {data[6]}')
        # Append the received metadata to the global list
        self.metadata.append({'address':data[0], 'port':data[1], 'reward':data[4], 'dist':data[5], 'embedding':data[6]})
    def pick_meta(self):
        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_REQ]
        # Time to pick the best agent
        if len(self.metadata) > 0:
            meta_copy = list(self.metadata)
            self.metadata[:] = []   # Reset the metadata list now that we have a copy

            meta_copy = sorted(meta_copy, key=lambda d: (d['dist'], -d['reward']))      # bi-directional multikey sorting using the distance and reward

            for meta_dict in meta_copy:
                if meta_dict['reward'] == torch.inf: pass
                else:
                    recv_address = meta_dict['address']
                    recv_port = meta_dict['port']
                    recv_rw = meta_dict['reward']
                    recv_dist = meta_dict['dist']
                    recv_emb = meta_dict['embedding']

                    self.logger.info(recv_address)
                    self.logger.info(recv_port)
                    self.logger.info(recv_rw)
                    self.logger.info(recv_dist)
                    self.logger.info(recv_emb)
                    
                    if recv_rw != 0.0:
                        if recv_dist <= ParallelComm.THRESHOLD:
                            if tuple(recv_emb) in self.knowledge_base.keys():
                                if 0.9 * round(recv_rw, 6) > self.knowledge_base[tuple(recv_emb.tolist())]:
                                    data.append(ParallelComm.MSG_DATA_MSK_REQ)
                                    data.append(recv_emb)
                                    self.client(data, recv_address, recv_port)
                                    break
                                
                            else:
                                data.append(ParallelComm.MSG_DATA_MSK_REQ)
                                data.append(recv_emb)
                                self.client(data, recv_address, recv_port)
                                break
    def request(self, data, queue_label_send, queue_mask_recv):
        print(Fore.WHITE + 'Mask request:')
        print(data)
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Embedding: {data[4]}')
        mask_req = {'response': True, 'embedding': data[4], 'address':data[0], 'port':data[1]}
        #mask_req = {'response': False, 'address':data[0], 'port':data[1]}   # For the evaluation agent.

        # Get the label to mask conversion
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)     # This will be a NoneType if no mask is available to be sent back. Comment out for evaluation agent
        self.logger.info(f'Processes mask resp: {mask_resp}')

        mask_resp['reward'] = self.knowledge_base[tuple(mask_resp['embedding'].tolist())]

        # Send the mask response back to the querying agent
        self.send_mask(mask_resp)
    def update_params(self, data):
        _query_list = data[3]
        _query_list.reverse()
        for addr in _query_list:
            if addr not in self.query_list:
                self.query_list.insert(0, addr)

        self.world_size.value = len(self.query_list) + 1


    # Event handler
    def event(self, data, queue_mask, queue_mask_recv, queue_label_send):
        ### EVENT HANDLING
        # Agent is sending a query table
        if data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_TABLE:
            self.logger.info(Fore.CYAN + 'Data is a query table')
            self.update_params(data)
            for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')

        # An agent is sending a query
        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_QUERY:
            self.logger.info(Fore.CYAN + 'Data is a query')
            self.query(data)

        # An agent is sending meta information in response to a query
        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_META:
            self.logger.info(Fore.CYAN + 'Data is metadata')
            self.add_meta(data)

        # An agent is sending a mask request
        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_REQ:
            self.logger.info(Fore.CYAN + 'Data is mask request')
            self.request(data, queue_label_send, queue_mask_recv)

        # An agent is sending a mask
        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_MASK:
            self.logger.info(Fore.CYAN + 'Data is a mask')
            # Unpack the received data
            received_mask, received_label, received_reward, ip, port = self.recv_mask(data)

            self.logger.info(f'{received_mask, received_label, received_reward, ip, port}')
            # Send the reeceived information back to the agent process if condition met
            if received_mask is not None and received_label is not None and received_reward is not None:
                self.logger.info('Sending mask data to agent')
                queue_mask.put((received_mask, received_label, received_reward, ip, port))

        print('\n')

    # Listening server
    def server(self, queue_mask, queue_mask_recv, queue_label_send):
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
        
        while True:
            # Accept the connection
            conn, addr = sock.accept()
            #conn = context.wrap_socket(conn, server_side=True)

            with conn:
                self.logger.info('\n' + Fore.CYAN + f'Connected by {addr}')
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
                        event_handler = mpd.Pool(processes=1)
                        event_handler.apply_async(self.event, (data, queue_mask, queue_mask_recv, queue_label_send))

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
        p_server = mp.Process(target=self.server, args=(queue_mask, queue_mask_recv, queue_label_send))
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
                for key, val in self.knowledge_base.items(): print(f'{key} : {val}')
                self.logger.info(Fore.GREEN + f'World size in comm: {self.world_size.value}')
                #self.logger.info(Fore.GREEN + f'Query table in this iteration:')
                #for addr in self.query_list: print(addr[0], addr[1])
                #self.logger.info(Fore.GREEN + f'Reference table this iteration:')
                #for addr in self.reference_list: print(addr[0], addr[1])



                # Block operation until an embedding is received to query for
                msg = queue_label.get()


                # Get the world size based on the number of addresses in the query list
                self.world_size.value = len(self.query_list) + 1


                # Send out a query when shell iterations matches mask interval if the agent is working on a task
                if self.world_size.value > 1:
                    self.send_query(msg)



            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            #except (SystemExit, KeyboardInterrupt) as e:                           # Uncomment to enable the keyboard interrupt and system exit handling
            #    p_server.close()
            #    #p_discover.close()
            #    #self.send_exit_net()
            #    sys.exit()
                
    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv):
        """
        Parallelisation method for starting the communication loop inside a seperate process.
        """

        p_client = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv))
        p_client.start()


'''
Evaluation agent implementation of the communication module. Works the same way as the learner, only it does not 
respond with meta_responses as it is not designed to affect the network it is evaluating.

TODO: Needs to be updated once the learner code has been fixed up.
'''
class ParallelCommEval(object):
    ### COMMUNCIATION MODULE HYPERPARAMETERS
    # DETECT MODULE CONSTANTS
    # Threshold for embedding/tasklabel distance (similarity)
    # This should be taken from the detect module eventually
    THRESHOLD = 0.0

    # SSL/TLS PATHS
    # Paths to the SSL/TLS certificates and key
    CERTPATH = 'certificates/certificate.pem'
    KEYPATH = 'certificates/key.pem'

    # COMMUNICATION DROPOUT
    # Used to simulate percentage communication dropout in the network. Currently only limits the amount of queries and not a total communication blackout.
    DROPOUT = 0.0  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout

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
    def __init__(self, num_agents, embd_dim, mask_dim, logger, init_port, reference, knowledge_base, manager, localhost):
        super(ParallelCommEval, self).__init__()
        self.embd_dim = embd_dim            # Dimensions of the the embedding
        self.mask_dim = mask_dim            # Dimensions of the mask for use in buffers. May no longer be needed
        self.logger = logger                # Logger object for logging CLI outputs.

        # Address and port for this agent
        if localhost: self.init_address = '127.0.0.1'
        else: self.init_address = self.init_address = urllib.request.urlopen('https://v4.ident.me').read().decode('utf8') # Use this to get the public ip of the host server.
        self.init_port = int(init_port)

        # Shared memory variables. Make these into attributes of the communication module to make it easier to use across the many sub processes of the module.
        self.query_list = manager.list([item for item in reference if item != (self.init_address, self.init_port)]) # manager.list(reference)
        self.reference_list = manager.list(deepcopy(self.query_list))   # Weird thing was happening here so used deepcopy to recreate the manager ListProxy with the addresses.
        self.knowledge_base = knowledge_base
        
        self.world_size = manager.Value('i', num_agents)
        self.metadata = manager.list()


        print(type(self.query_list))
        print(type(self.reference_list))

        # For debugging
        print('Query table:')
        for addr in self.query_list: print(addr[0], addr[1])

        print('\nReference table:')
        for addr in self.reference_list: print(addr[0], addr[1])

        print(f'\nlistening server params ->\naddress: {self.init_address}\nport: {self.init_port}\n')
        print('mask size:', self.mask_dim)
        print('embedding size:', self.embd_dim)

    def _null_message(self, msg):
        """
        Checks if a message contains null i.e, no data.

        Args:
            msg: A list received from another agent.

        Returns:
            A boolean indicating whether A list contains null data.
        """

        # check whether message sent denotes or is none.
        if bool(msg[ParallelCommEval.META_INF_IDX_MSG_DATA] == ParallelCommEval.MSG_DATA_NULL):
            return True

        else:
            return False

    # Client used by the server to send responses
    def client(self, data, address, port):
        """
        Client implementation. Begins a TCP connection secured using SSL/TLS to a trusted server ip-port. Attempts to send the serialized bytes.
        
        Args:
            data: A list to be sent to another agent.
            address: The ip of the destination.
            port: The port of the destination.
        """
        _data = pickle.dumps(data, protocol=5)

        # Attempt to send the data a number of times. If successful do not attempt to send again.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)

        try:
            sock.connect((address, port))

            #context = ssl.create_default_context()
            #context.load_cert_chain(ParallelCommEval.CERTPATH, ParallelCommEval.KEYPATH)
            #sock_ssl = context.wrap_socket(sock, server_side=False)

            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')


        except:
            # Try to remove the ip and port that failed from the query table
            #try:
            #    self.query_list.remove((address, port))
            #    self.world_size.value = len(self.query_list) + 1
            #except:
            #    pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
        finally: sock.close()
    
    # Modified version of the client used by the send_query function. Has an additional bit of code to handle the mask response before querying the next agent in the query list
    def query_client(self, data, address, port):
        # Attempt to send the data a number of times. If successful do not attempt to send again.
        
        _data = pickle.dumps(data, protocol=5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        
        try:
            sock.connect((address, port))

            #context = ssl.create_default_context()
            #context.load_cert_chain(ParallelCommEval.CERTPATH, ParallelCommEval.KEYPATH)
            #sock = context.wrap_socket(sock, server_side=False)

            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table
            #try:
            #    self.query_list.remove((address, port))
            #    self.world_size.value = len(self.query_list) + 1
            #except:
            #    pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
        finally:
            sock.close()

    ### Query send and recv functions
    def send_query(self, embedding):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            embedding: A torch tensor containing an embedding
        """

        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        #self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        reward = 0.0
        if tuple(embedding.tolist()) in self.knowledge_base:
            reward = self.knowledge_base[tuple(embedding.tolist())]


        if embedding is None:
            data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_QUERY, ParallelCommEval.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_QUERY, ParallelCommEval.MSG_DATA_QUERY, embedding, reward]

        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in list(self.query_list):
            self.query_client(data, addr[0], addr[1])

        time.sleep(0.2)
        self.pick_meta()
        
    def recv_mask(self, buffer):
        """
        Unpacks a received mask response from another agent.
        
        Args:
            buffer: A list received from another agent.
            best_agent_id: A shared memory variable of type dict() containing a ip-port pair for the best agent.
            
        Returns:
            received_mask: A torch tensor containing the continous mask parameters.
            received_label: A torch tensor containing the embedding.
        """
        
        received_mask = None
        received_label = None
        received_reward = None
        ip = None
        port = None

        if buffer[ParallelCommEval.META_INF_IDX_MSG_DATA] == ParallelCommEval.MSG_DATA_NULL:
            pass

        elif buffer[ParallelCommEval.META_INF_IDX_MSG_DATA] == ParallelCommEval.MSG_DATA_MSK:
            received_mask = buffer[4]
            received_label = buffer[5]
            received_reward = buffer[6]
            ip = buffer[0]
            port = buffer[1]

        return received_mask, received_label, received_reward, ip, port


    def recv_query(self, buffer):
        """
        Unpacks the data buffer received from another agent for a query.
        
        Args:
            buffer: A list received from another agent.
            
        Returns:
            ret: A dictionary containing the unpacked data.
        """

        ret = {}
        ret['sender_address'] = str(buffer[ParallelCommEval.META_INF_IDX_ADDRESS])
        ret['sender_port'] = int(buffer[ParallelCommEval.META_INF_IDX_PORT])
        #ret['msg_type'] = int(buffer[ParallelCommEval.META_INF_IDX_MSG_TYPE])
        #ret['msg_data'] = int(buffer[ParallelCommEval.META_INF_IDX_MSG_DATA])
        ret['embedding'] = buffer[ParallelCommEval.META_INF_IDX_TASK_SZ]
        ret['sender_reward'] = buffer[-1]

        # Handle when we receive a query from another agent that we do not have in our known list of agents (self.query_list)
        if (ret['sender_address'], ret['sender_port']) not in self.query_list:
            self.client([self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_TABLE, list(self.query_list)], ret['sender_address'], ret['sender_port'])
            self.query_list.append((ret['sender_address'], ret['sender_port']))

        self.world_size.value = len(self.query_list) + 1    # Refresh the world_size value

        return ret
    def proc_meta(self, other_agent_req):
        """
        Processes a query for an embedding and produces a response to send back to the requesting agent.
        
        Args:
            other_agent_req: A dictionary containing the information for the query request.
            knowledge_base: A shared memory variable consisting of a dictionary to store the task embeddings and rewards accumulated.
        
        Returns:
            meta_response: A dictionary containing the response information.
        """

        other_agent_req['response'] = False

        '''if other_agent_req is not None:
            np_embedding = other_agent_req['embedding'].detach().cpu().numpy()
            sender_reward = other_agent_req['sender_reward']

            # Iterate through the knowledge base and compute the distances
            # If reward greater than 0
            # If distance is less than or equal to threshold
            # response = True (false by default)
            for tlabel, treward in self.knowledge_base.items():
                if treward > np.around(0.0, decimals=6):
                    if 0.9 * round(treward, 6) > sender_reward:
                        tdist = np.sum(abs(np.subtract(np_embedding, np.array(tlabel))))
                        if tdist <= ParallelCommEval.THRESHOLD:
                            other_agent_req['response'] = True
                            other_agent_req['reward'] = treward         # Reward of the mask this agent has for the task
                            other_agent_req['dist'] = tdist             # Distance between the embedding of this agent's closest mask and the embedding from the querying agent
                            other_agent_req['resp_embedding'] = torch.tensor(tlabel)  # The closest embedding that this agent has to the one that the querying agent has queried for'''

        # Return the query request
        return other_agent_req
    def send_meta(self, meta_resp):
        if meta_resp:
            data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_META]
            if meta_resp['response']:
                data.append(ParallelCommEval.MSG_DATA_META)
                data.append(meta_resp.get('reward', None))
                data.append(meta_resp.get('dist', None))
                data.append(meta_resp.get('resp_embedding', None))

            #else:
            #    data.append(ParallelCommEval.MSG_DATA_NULL)

                self.client(data, str(meta_resp['sender_address']), int(meta_resp['sender_port']))

    def proc_mask(self, mask_req, queue_label_send, queue_mask_recv):
        """
        Processes the mask response to send to another agent.
        
        Args:
            mask_req: A dictionary consisting of the response information to send to a specific agent.
            queue_label_send: A shared memory queue to send an embedding to be converted by the agent module.
            queue_mask_recv: A shared memory queue to receive a converted mask from the agent module.

        Returns:
            The mask_req dictionary with the converted mask now included. 
        """

        if mask_req['response']:
            self.logger.info('Sending mask request to be converted')
            queue_label_send.put((mask_req))
            self.logger.info('Mask request sent')
            return queue_mask_recv.get()        # Return the dictionary with the mask attached

    def send_mask(self, mask_resp):
        """
        Sends a mask response to a specific agent.
        
        Args:
            mask_resp: A dictionary consisting of the information to send to a specific agent.    
        """
        if mask_resp:
            data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_MASK]

            # if response is True then send the mask
            if mask_resp['response']:
                data.append(ParallelCommEval.MSG_DATA_MSK)
                data.append(mask_resp.get('mask', None))
                data.append(mask_resp.get('embedding', None))
                data.append(mask_resp.get('reward', None))

                self.logger.info(f'Sending mask response: {data}')
                self.client(data, str(mask_resp['address']), int(mask_resp['port']))
            
            # otherwise send a null response
            #else:
            #    data.append(ParallelCommEval.MSG_DATA_NULL)
            
            
    
    # Event handler wrappers. This is done so the various functions for each event can be run in a single thread.
    def query(self, data):
        """
        Event handler for receiving a query from another agent. Unpacks the buffer received from another agent, processes the request and sends some response if necessary.
        
        Args:
            data: A list received from another agent.
            knowledge_base: A shared memory variable of type dict() containing embedding-reward pairs for task embeddings observed by the agent.    
        """

        # Get the query from the other agent
        other_agent_req = self.recv_query(data)
        self.logger.info(f'Received query: {other_agent_req}')

        # Check if this agent has any knowledge for the task
        meta_resp = self.proc_meta(other_agent_req)
        self.logger.info(f'Processes mask req: {meta_resp}')

        self.send_meta(meta_resp)
    def add_meta(self, data):
        print(Fore.YELLOW + 'Metadata:')
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Reward: {data[4]}')
        print(f'Distance: {data[5]}')
        print(f'Embedding: {data[6]}')
        # Append the received metadata to the global list
        self.metadata.append({'address':data[0], 'port':data[1], 'reward':data[4], 'dist':data[5], 'embedding':data[6]})
    def pick_meta(self):
        data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_REQ]
        # Time to pick the best agent
        if len(self.metadata) > 0:
            meta_copy = list(self.metadata)
            self.metadata[:] = []   # Reset the metadata list now that we have a copy

            meta_copy = sorted(meta_copy, key=lambda d: (d['dist'], -d['reward']))      # bi-directional multikey sorting using the distance and reward

            for meta_dict in meta_copy:
                if meta_dict['reward'] == torch.inf: pass
                else:
                    recv_address = meta_dict['address']
                    recv_port = meta_dict['port']
                    recv_rw = meta_dict['reward']
                    recv_dist = meta_dict['dist']
                    recv_emb = meta_dict['embedding']

                    self.logger.info(recv_address)
                    self.logger.info(recv_port)
                    self.logger.info(recv_rw)
                    self.logger.info(recv_dist)
                    self.logger.info(recv_emb)
                    
                    if recv_rw != 0.0:
                        if recv_dist <= ParallelCommEval.THRESHOLD:
                            if tuple(recv_emb) in self.knowledge_base.keys():
                                if 0.9 * round(recv_rw, 6) > self.knowledge_base[tuple(recv_emb.tolist())]:
                                    data.append(ParallelCommEval.MSG_DATA_MSK_REQ)
                                    data.append(recv_emb)
                                    self.client(data, recv_address, recv_port)
                                    break
                                
                            else:
                                data.append(ParallelCommEval.MSG_DATA_MSK_REQ)
                                data.append(recv_emb)
                                self.client(data, recv_address, recv_port)
                                break
    def request(self, data, queue_label_send, queue_mask_recv):
        print(Fore.WHITE + 'Mask request:')
        print(data)
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Embedding: {data[4]}')
        mask_req = {'response': True, 'embedding': data[4], 'address':data[0], 'port':data[1]}
        #mask_req = {'response': False, 'address':data[0], 'port':data[1]}   # For the evaluation agent.

        # Get the label to mask conversion
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)     # This will be a NoneType if no mask is available to be sent back. Comment out for evaluation agent
        self.logger.info(f'Processes mask resp: {mask_resp}')

        mask_resp['reward'] = self.knowledge_base[tuple(mask_resp['embedding'].tolist())]

        # Send the mask response back to the querying agent
        self.send_mask(mask_resp)
    def update_params(self, data):
        _query_list = data[3]
        _query_list.reverse()
        for addr in _query_list:
            if addr not in self.query_list:
                self.query_list.insert(0, addr)

        self.world_size.value = len(self.query_list) + 1


    # Event handler
    def event(self, data, queue_mask, queue_mask_recv, queue_label_send):
        ### EVENT HANDLING
        # Agent is sending a query table
        if data[ParallelCommEval.META_INF_IDX_MSG_TYPE] == ParallelCommEval.MSG_TYPE_SEND_TABLE:
            self.logger.info(Fore.CYAN + 'Data is a query table')
            self.update_params(data)
            for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')

        # An agent is sending a query
        elif data[ParallelCommEval.META_INF_IDX_MSG_TYPE] == ParallelCommEval.MSG_TYPE_SEND_QUERY:
            self.logger.info(Fore.CYAN + 'Data is a query')
            self.query(data)

        # An agent is sending meta information in response to a query
        elif data[ParallelCommEval.META_INF_IDX_MSG_TYPE] == ParallelCommEval.MSG_TYPE_SEND_META:
            self.logger.info(Fore.CYAN + 'Data is metadata')
            self.add_meta(data)

        # An agent is sending a mask request
        elif data[ParallelCommEval.META_INF_IDX_MSG_TYPE] == ParallelCommEval.MSG_TYPE_SEND_REQ:
            self.logger.info(Fore.CYAN + 'Data is mask request')
            self.request(data, queue_label_send, queue_mask_recv)

        # An agent is sending a mask
        elif data[ParallelCommEval.META_INF_IDX_MSG_TYPE] == ParallelCommEval.MSG_TYPE_SEND_MASK:
            self.logger.info(Fore.CYAN + 'Data is a mask')
            # Unpack the received data
            received_mask, received_label, received_reward, ip, port = self.recv_mask(data)

            self.logger.info(f'{received_mask, received_label, received_reward, ip, port}')
            # Send the reeceived information back to the agent process if condition met
            if received_mask is not None and received_label is not None and received_reward is not None:
                self.logger.info('Sending mask data to agent')
                queue_mask.put((received_mask, received_label, received_reward, ip, port))

        print('\n')

    # Listening server
    def server(self, queue_mask, queue_mask_recv, queue_label_send):
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
        
        while True:
            # Accept the connection
            conn, addr = sock.accept()
            #conn = context.wrap_socket(conn, server_side=True)

            with conn:
                self.logger.info('\n' + Fore.CYAN + f'Connected by {addr}')
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
                        event_handler = mpd.Pool(processes=1)
                        event_handler.apply_async(self.event, (data, queue_mask, queue_mask_recv, queue_label_send))

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
        p_server = mp.Process(target=self.server, args=(queue_mask, queue_mask_recv, queue_label_send))
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
                for key, val in self.knowledge_base.items(): print(f'{key} : {val}')
                self.logger.info(Fore.GREEN + f'World size in comm: {self.world_size.value}')
                #self.logger.info(Fore.GREEN + f'Query table in this iteration:')
                #for addr in self.query_list: print(addr[0], addr[1])
                #self.logger.info(Fore.GREEN + f'Reference table this iteration:')
                #for addr in self.reference_list: print(addr[0], addr[1])



                # Block operation until an embedding is received to query for
                msg = queue_label.get()


                # Get the world size based on the number of addresses in the query list
                self.world_size.value = len(self.query_list) + 1


                # Send out a query when shell iterations matches mask interval if the agent is working on a task
                if self.world_size.value > 1:
                    if int(np.random.choice(2, 1, p=[ParallelCommEval.DROPOUT, 1 - ParallelCommEval.DROPOUT])) == 1:  # Condition to simulate % communication loss
                        self.send_query(msg)



            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            #except (SystemExit, KeyboardInterrupt) as e:                           # Uncomment to enable the keyboard interrupt and system exit handling
            #    p_server.close()
            #    #p_discover.close()
            #    #self.send_exit_net()
            #    sys.exit()
                
    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv):
        """
        Parallelisation method for starting the communication loop inside a seperate process.
        """

        p_client = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv))
        p_client.start() 


'''
Omniscient agent implementation of the communication module. Works in a similar fashion as the learner. Has an
additional querying component that handles the collection of all knowledge from the collective based on what has
been observed by the network. This is done by tracking all the embeddings from incoming queries. Using this, the agent
queries for the knowledge at the communication interval.

TODO: Needs to be tested with a modified trainer_shell script. Also need to modify the code to be able to switch between omniscient mode and standard learner mode on the fly.
'''
class ParallelCommOmniscient(object):
    ### COMMUNCIATION MODULE HYPERPARAMETERS
    # DETECT MODULE CONSTANTS
    # Threshold for embedding/tasklabel distance (similarity)
    # This should be taken from the detect module eventually
    THRESHOLD = 0.0

    # SSL/TLS PATHS
    # Paths to the SSL/TLS certificates and key
    CERTPATH = 'certificates/certificate.pem'
    KEYPATH = 'certificates/key.pem'

    # COMMUNICATION DROPOUT
    # Used to simulate percentage communication dropout in the network. Currently only limits the amount of queries and not a total communication blackout.
    DROPOUT = 0.0  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout

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
    def __init__(self, num_agents, embd_dim, mask_dim, logger, init_port, reference, knowledge_base, manager, localhost, mode, dropout):
        super(ParallelCommOmniscient, self).__init__()
        self.embd_dim = embd_dim            # Dimensions of the the embedding
        self.mask_dim = mask_dim            # Dimensions of the mask for use in buffers. May no longer be needed
        self.logger = logger                # Logger object for logging CLI outputs.
        self.mode = mode                    # Toggle flag for omniscient mode

        # Address and port for this agent
        if localhost: self.init_address = '127.0.0.1'
        else: self.init_address = self.init_address = urllib.request.urlopen('https://v4.ident.me').read().decode('utf8') # Use this to get the public ip of the host server.
        self.init_port = int(init_port)

        # Shared memory variables. Make these into attributes of the communication module to make it easier to use across the many sub processes of the module.
        self.query_list = manager.list([item for item in reference if item != (self.init_address, self.init_port)]) # manager.list(reference)
        self.reference_list = manager.list(deepcopy(self.query_list))   # Weird thing was happening here so used deepcopy to recreate the manager ListProxy with the addresses.
        self.knowledge_base = knowledge_base
        
        self.world_size = manager.Value('i', num_agents)
        self.metadata = manager.list()

        # COMMUNICATION DROPOUT
        # Used to simulate percentage communication dropout in the network. Currently only limits the amount of queries and not a total communication blackout.
        self.dropout = dropout  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout

        # Omniscient task tracker. Makes a collection of all observed tasks by the visible network.
        # When a query is received by this agent, the list is updated with the embedding.
        # The list is also updated when this agent observes a new task.
        # The agent then uses this to query for everything after querying for its current task at every interval
        # This is because the gather all operation is not as time sensitive as the query specific to this agent's task
        self.task_tracker = manager.dict()

        print('RUNNING IN OMNISCIENT MODE :D')

        print(type(self.query_list))
        print(type(self.reference_list))

        # For debugging
        print('Query table:')
        for addr in self.query_list: print(addr[0], addr[1])

        print('\nReference table:')
        for addr in self.reference_list: print(addr[0], addr[1])

        print(f'\nlistening server params ->\naddress: {self.init_address}\nport: {self.init_port}\n')
        print('mask size:', self.mask_dim)
        print('embedding size:', self.embd_dim)

    def _null_message(self, msg):
        """
        Checks if a message contains null i.e, no data.

        Args:
            msg: A list received from another agent.

        Returns:
            A boolean indicating whether A list contains null data.
        """

        # check whether message sent denotes or is none.
        if bool(msg[ParallelCommOmniscient.META_INF_IDX_MSG_DATA] == ParallelCommOmniscient.MSG_DATA_NULL):
            return True

        else:
            return False

    # Client used by the server to send responses
    def client(self, data, address, port):
        """
        Client implementation. Begins a TCP connection secured using SSL/TLS to a trusted server ip-port. Attempts to send the serialized bytes.
        
        Args:
            data: A list to be sent to another agent.
            address: The ip of the destination.
            port: The port of the destination.
        """
        _data = pickle.dumps(data, protocol=5)

        # Attempt to send the data a number of times. If successful do not attempt to send again.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)

        try:
            sock.connect((address, port))

            #context = ssl.create_default_context()
            #context.load_cert_chain(ParallelCommOmniscient.CERTPATH, ParallelCommOmniscient.KEYPATH)
            #sock_ssl = context.wrap_socket(sock, server_side=False)

            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')


        except:
            # Try to remove the ip and port that failed from the query table
            try:
                self.query_list.remove((address, port))
                self.world_size.value = len(self.query_list) + 1
            except:
                pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
        finally: sock.close()
    
    # Modified version of the client used by the send_query function. Has an additional bit of code to handle the mask response before querying the next agent in the query list
    def query_client(self, data, address, port):
        # Attempt to send the data a number of times. If successful do not attempt to send again.
        
        _data = pickle.dumps(data, protocol=5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        
        try:
            sock.connect((address, port))

            #context = ssl.create_default_context()
            #context.load_cert_chain(ParallelCommOmniscient.CERTPATH, ParallelCommOmniscient.KEYPATH)
            #sock = context.wrap_socket(sock, server_side=False)

            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table
            try:
                self.query_list.remove((address, port))
                self.world_size.value = len(self.query_list) + 1
            except:
                pass
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
        finally:
            sock.close()

    ### Query send and recv functions
    def send_query(self, embedding):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            embedding: A torch tensor containing an embedding
        """

        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        #self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        reward = 0.0
        if tuple(embedding.tolist()) in self.knowledge_base:
            reward = self.knowledge_base[tuple(embedding.tolist())]


        if embedding is None:
            data = [self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_QUERY, ParallelCommOmniscient.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_QUERY, ParallelCommOmniscient.MSG_DATA_QUERY, embedding, reward]

        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in list(self.query_list):
            self.query_client(data, addr[0], addr[1])

        time.sleep(0.2)        # Originally 2
        self.pick_meta()
        
    def recv_mask(self, buffer):
        """
        Unpacks a received mask response from another agent.
        
        Args:
            buffer: A list received from another agent.
            best_agent_id: A shared memory variable of type dict() containing a ip-port pair for the best agent.
            
        Returns:
            received_mask: A torch tensor containing the continous mask parameters.
            received_label: A torch tensor containing the embedding.
        """
        
        received_mask = None
        received_label = None
        received_reward = None
        ip = None
        port = None

        if buffer[ParallelCommOmniscient.META_INF_IDX_MSG_DATA] == ParallelCommOmniscient.MSG_DATA_NULL:
            pass

        elif buffer[ParallelCommOmniscient.META_INF_IDX_MSG_DATA] == ParallelCommOmniscient.MSG_DATA_MSK:
            received_mask = buffer[4]
            received_label = buffer[5]
            received_reward = buffer[6]
            ip = buffer[0]
            port = buffer[1]

        return received_mask, received_label, received_reward, ip, port


    def recv_query(self, buffer):
        """
        Unpacks the data buffer received from another agent for a query.
        
        Args:
            buffer: A list received from another agent.
            
        Returns:
            ret: A dictionary containing the unpacked data.
        """

        ret = {}
        ret['sender_address'] = str(buffer[ParallelCommOmniscient.META_INF_IDX_ADDRESS])
        ret['sender_port'] = int(buffer[ParallelCommOmniscient.META_INF_IDX_PORT])
        #ret['msg_type'] = int(buffer[ParallelCommOmniscient.META_INF_IDX_MSG_TYPE])
        #ret['msg_data'] = int(buffer[ParallelCommOmniscient.META_INF_IDX_MSG_DATA])
        ret['embedding'] = buffer[ParallelCommOmniscient.META_INF_IDX_TASK_SZ]
        ret['sender_reward'] = buffer[-1]

        if ret['embedding'] not in self.task_tracker:
            self.task_tracker[f"{ret['sender_address']}, {ret['sender_port']}"] = (ret['embedding'], ret['sender_reward'])      # A hacky way to store unique embeddings in a 'set' using a dictionary instead of a set. Python multiprocessing does not have a set proxy in manager objects.

        # Handle when we receive a query from another agent that we do not have in our known list of agents (self.query_list)
        if (ret['sender_address'], ret['sender_port']) not in self.query_list:
            self.client([self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_TABLE, list(self.query_list)], ret['sender_address'], ret['sender_port'])
            self.query_list.append((ret['sender_address'], ret['sender_port']))

        self.world_size.value = len(self.query_list) + 1    # Refresh the world_size value

        return ret
    def proc_meta(self, other_agent_req):
        """
        Processes a query for an embedding and produces a response to send back to the requesting agent.
        
        Args:
            other_agent_req: A dictionary containing the information for the query request.
            knowledge_base: A shared memory variable consisting of a dictionary to store the task embeddings and rewards accumulated.
        
        Returns:
            meta_response: A dictionary containing the response information.
        """

        other_agent_req['response'] = False

        if other_agent_req is not None:
            np_embedding = other_agent_req['embedding'].detach().cpu().numpy()
            sender_reward = other_agent_req['sender_reward']

            # Iterate through the knowledge base and compute the distances
            # If reward greater than 0
            # If distance is less than or equal to threshold
            # response = True (false by default)
            for tlabel, treward in self.knowledge_base.items():
                if treward > np.around(0.0, decimals=6):
                    if 0.9 * round(treward, 6) > sender_reward:
                        tdist = np.sum(abs(np.subtract(np_embedding, np.array(tlabel))))
                        if tdist <= ParallelCommOmniscient.THRESHOLD:
                            other_agent_req['response'] = True
                            other_agent_req['reward'] = treward         # Reward of the mask this agent has for the task
                            other_agent_req['dist'] = tdist             # Distance between the embedding of this agent's closest mask and the embedding from the querying agent
                            other_agent_req['resp_embedding'] = torch.tensor(tlabel)  # The closest embedding that this agent has to the one that the querying agent has queried for

        # Return the query request
        return other_agent_req
    def send_meta(self, meta_resp):
        if meta_resp:
            data = [self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_META]
            if meta_resp['response']:
                data.append(ParallelCommOmniscient.MSG_DATA_META)
                data.append(meta_resp.get('reward', None))
                data.append(meta_resp.get('dist', None))
                data.append(meta_resp.get('resp_embedding', None))

            #else:
            #    data.append(ParallelCommOmniscient.MSG_DATA_NULL)

                self.client(data, str(meta_resp['sender_address']), int(meta_resp['sender_port']))

    def proc_mask(self, mask_req, queue_label_send, queue_mask_recv):
        """
        Processes the mask response to send to another agent.
        
        Args:
            mask_req: A dictionary consisting of the response information to send to a specific agent.
            queue_label_send: A shared memory queue to send an embedding to be converted by the agent module.
            queue_mask_recv: A shared memory queue to receive a converted mask from the agent module.

        Returns:
            The mask_req dictionary with the converted mask now included. 
        """

        if mask_req['response']:
            self.logger.info('Sending mask request to be converted')
            queue_label_send.put((mask_req))
            self.logger.info('Mask request sent')
            return queue_mask_recv.get()        # Return the dictionary with the mask attached

    def send_mask(self, mask_resp):
        """
        Sends a mask response to a specific agent.
        
        Args:
            mask_resp: A dictionary consisting of the information to send to a specific agent.    
        """
        if mask_resp:
            data = [self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_MASK]

            # if response is True then send the mask
            if mask_resp['response']:
                data.append(ParallelCommOmniscient.MSG_DATA_MSK)
                data.append(mask_resp.get('mask', None))
                data.append(mask_resp.get('embedding', None))
                data.append(mask_resp.get('reward', None))

                self.logger.info(f'Sending mask response: {data}')
                self.client(data, str(mask_resp['address']), int(mask_resp['port']))
            
            # otherwise send a null response
            #else:
            #    data.append(ParallelCommOmniscient.MSG_DATA_NULL)
            
    def gather_all(self, msg):
        #if msg is not None:
        #    self.logger.info('QUERYING FOR CURRENT TASK')
        #    self.send_query(msg)

        # Lots of queries :'))) Lets hope the system doesnt die. Perform a query with all agents for every embedding in that has been tracked.
        for destination, (embedding, reward) in self.task_tracker.items():
            src_addr = str(destination.split(', ')[0])
            src_port = int(destination.split(', ')[1])

            if src_addr == self.init_address and src_port == self.init_port: continue # If we were the last one to learn this task then there is no need to get new knowledge


            if reward <= self.knowledge_base.get(tuple(embedding.tolist()), 0.0): continue
            
            #reward = 0.0
            #if tuple(embedding.tolist()) in self.knowledge_base:
            #    reward = self.knowledge_base[tuple(embedding.tolist())]

            data = [self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_REQ, ParallelCommOmniscient.MSG_DATA_MSK_REQ, embedding]

            try:
                self.logger.info(f'{Fore.YELLOW}Running self.client() for {src_addr}, {src_port}')
                self.client(data, src_addr, src_port)
            except Exception as e:
                self.logger.info(traceback.format_exc())
        
    
    # Event handler wrappers. This is done so the various functions for each event can be run in a single thread.
    def query(self, data):
        """
        Event handler for receiving a query from another agent. Unpacks the buffer received from another agent, processes the request and sends some response if necessary.
        
        Args:
            data: A list received from another agent.
            knowledge_base: A shared memory variable of type dict() containing embedding-reward pairs for task embeddings observed by the agent.    
        """

        # Get the query from the other agent
        other_agent_req = self.recv_query(data)
        self.logger.info(f'Received query: {other_agent_req}')

        # Check if this agent has any knowledge for the task
        meta_resp = self.proc_meta(other_agent_req)
        self.logger.info(f'Processes mask req: {meta_resp}')

        self.send_meta(meta_resp)
    def add_meta(self, data):
        print(Fore.YELLOW + 'Metadata:')
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Reward: {data[4]}')
        print(f'Distance: {data[5]}')
        print(f'Embedding: {data[6]}')
        # Append the received metadata to the global list
        self.metadata.append({'address':data[0], 'port':data[1], 'reward':data[4], 'dist':data[5], 'embedding':data[6]})
    def pick_meta(self):
        data = [self.init_address, self.init_port, ParallelCommOmniscient.MSG_TYPE_SEND_REQ]
        # Time to pick the best agent
        if len(self.metadata) > 0:
            meta_copy = list(self.metadata)
            #self.metadata[:] = []   # Reset the metadata list now that we have a copy

            meta_copy = sorted(meta_copy, key=lambda d: (d['dist'], -d['reward']))      # bi-directional multikey sorting using the distance and reward

            for meta_dict in meta_copy:
                if meta_dict['reward'] == torch.inf: pass
                else:
                    recv_address = meta_dict['address']
                    recv_port = meta_dict['port']
                    recv_rw = meta_dict['reward']
                    recv_dist = meta_dict['dist']
                    recv_emb = meta_dict['embedding']

                    self.logger.info(recv_address)
                    self.logger.info(recv_port)
                    self.logger.info(recv_rw)
                    self.logger.info(recv_dist)
                    self.logger.info(recv_emb)
                    
                    if recv_rw != 0.0:
                        if recv_dist <= ParallelCommOmniscient.THRESHOLD:
                            if tuple(recv_emb) in self.knowledge_base.keys():
                                if 0.9 * round(recv_rw, 6) > self.knowledge_base[tuple(recv_emb.tolist())]:
                                    data.append(ParallelCommOmniscient.MSG_DATA_MSK_REQ)
                                    data.append(recv_emb)
                                    self.client(data, recv_address, recv_port)
                                    break
                                
                            else:
                                data.append(ParallelCommOmniscient.MSG_DATA_MSK_REQ)
                                data.append(recv_emb)
                                self.client(data, recv_address, recv_port)
                                break
    def request(self, data, queue_label_send, queue_mask_recv):
        print(Fore.WHITE + 'Mask request:')
        print(data)
        print(f'Address: {data[0]}')
        print(f'Port: {data[1]}')
        print(f'Embedding: {data[4]}')
        mask_req = {'response': True, 'embedding': data[4], 'address':data[0], 'port':data[1]}
        #mask_req = {'response': False, 'address':data[0], 'port':data[1]}   # For the evaluation agent.

        # Get the label to mask conversion
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)     # This will be a NoneType if no mask is available to be sent back. Comment out for evaluation agent
        self.logger.info(f'Processes mask resp: {mask_resp}')

        mask_resp['reward'] = self.knowledge_base[tuple(mask_resp['embedding'].tolist())]

        # Send the mask response back to the querying agent
        self.send_mask(mask_resp)
    def update_params(self, data):
        _query_list = data[3]
        _query_list.reverse()
        for addr in _query_list:
            if addr not in self.query_list:
                self.query_list.insert(0, addr)

        self.world_size.value = len(self.query_list) + 1

    # Listening server
    def server(self, queue_mask, queue_mask_recv, queue_label_send):
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
        
        while True:
            # Accept the connection
            conn, addr = sock.accept()
            #conn = context.wrap_socket(conn, server_side=True)

            with conn:
                self.logger.info('\n' + Fore.CYAN + f'Connected by {addr}')
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

                        ### EVENT HANDLING
                        # Agent is sending a query table
                        if data[ParallelCommOmniscient.META_INF_IDX_MSG_TYPE] == ParallelCommOmniscient.MSG_TYPE_SEND_TABLE:
                            self.logger.info(Fore.CYAN + 'Data is a query table')
                            self.update_params(data)
                            for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')

                        # An agent is sending a query
                        elif data[ParallelCommOmniscient.META_INF_IDX_MSG_TYPE] == ParallelCommOmniscient.MSG_TYPE_SEND_QUERY:
                            self.logger.info(Fore.CYAN + 'Data is a query')
                            self.query(data)

                        # An agent is sending meta information in response to a query
                        elif data[ParallelCommOmniscient.META_INF_IDX_MSG_TYPE] == ParallelCommOmniscient.MSG_TYPE_SEND_META:
                            self.logger.info(Fore.CYAN + 'Data is metadata')
                            self.add_meta(data)

                        # An agent is sending a mask request
                        elif data[ParallelCommOmniscient.META_INF_IDX_MSG_TYPE] == ParallelCommOmniscient.MSG_TYPE_SEND_REQ:
                            self.logger.info(Fore.CYAN + 'Data is mask request')
                            self.request(data, queue_label_send, queue_mask_recv)

                        # An agent is sending a mask
                        elif data[ParallelCommOmniscient.META_INF_IDX_MSG_TYPE] == ParallelCommOmniscient.MSG_TYPE_SEND_MASK:
                            self.logger.info(Fore.CYAN + 'Data is a mask')
                            # Unpack the received data
                            received_mask, received_label, received_reward, ip, port = self.recv_mask(data)

                            
                            self.knowledge_base[tuple(received_label.tolist())] = received_reward

                            self.logger.info(f'{received_mask, received_label, received_reward, ip, port}')
                            # Send the reeceived information back to the agent process if condition met
                            if received_mask is not None and received_label is not None and received_reward is not None:
                                self.logger.info('Sending mask data to agent')
                                queue_mask.put((received_mask, received_label, received_reward, ip, port))

                        print('\n')

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
            queue_mask_recv: A shared memory queue to send masks from the communication module to the agent module for distillation.
        """

        # Initialise the listening server
        p_server = mp.Process(target=self.server, args=(queue_mask, queue_mask_recv, queue_label_send))
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
            try:
                print()
                self.logger.info(Fore.GREEN + f'Knowledge base in this iteration:')
                for key, val in self.knowledge_base.items(): self.logger.info(f'{key} : {val}')
                self.logger.info(Fore.GREEN + f'World size in comm: {self.world_size.value}')
                #self.logger.info(Fore.GREEN + f'Query table in this iteration:')
                #for addr in self.query_list: print(addr[0], addr[1])
                #self.logger.info(Fore.GREEN + f'Reference table this iteration:')
                #for addr in self.reference_list: print(addr[0], addr[1])



                # Block operation until an embedding is received to query for
                msg = queue_label.get()

                #print(msg, type(msg))
                # Update the tracked tasks with the ones that this agent also observes
                #self.task_tracker[f"{self.init_address}, {self.init_port}"] = (torch.tensor(msg), self.knowledge_base.get(tuple(msg), 0.0))

                # Get the world size based on the number of addresses in the query list
                self.world_size.value = len(self.query_list) + 1


                # Send out a query when shell iterations matches mask interval if the agent is working on a task
                if self.world_size.value > 1:
                    if int(np.random.choice(2, 1, p=[ParallelCommOmniscient.DROPOUT, 1 - ParallelCommOmniscient.DROPOUT])) == 1:  # Condition to simulate % communication loss
                        #if msg is not None:
                        #    self.send_query(msg)

                        
                        try:
                            if self.mode.value:
                                self.logger.info(f'{Fore.YELLOW}ACTIVATING GATHER ALL OPERATION')
                                self.logger.info(f'{Fore.YELLOW}TRACKED TASK LABELS: {self.task_tracker}')
                                self.gather_all(msg)
                        except Exception as e:
                            self.logger.info(traceback.format_exc())


                        time.sleep(1)
                        self.metadata[:] = []


            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            except (SystemExit, KeyboardInterrupt) as e:                           # Uncomment to enable the keyboard interrupt and system exit handling
                p_server.close()
                #p_discover.close()
                #self.send_exit_net()
                sys.exit()
                
    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv):
        """
        Parallelisation method for starting the communication loop inside a seperate process.
        """

        p_client = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv))
        p_client.start()