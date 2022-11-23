# -*- coding: utf-8 -*-

#   _________                                           .__                  __   .__                 
#   \_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
#   /    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
#   \     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
#    \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
#           \/               \/       \/             \/          \/      \/                        \/ 
#
#                          Now featuring shared memory variables. Have fun tracking those :)
#                                                 (╯°□°)╯︵ ┻━┻
from colorama import Fore
import copy
import datetime
import multiprocessing as mp
import multiprocessing.dummy as mpd
import os
import pickle
import socket
import ssl
import time
from errno import ECONNRESET
from queue import Empty

import numpy as np
import torch

'''
Revise communication class. Contains the new implementation of the communication module
Improves on the bandwidth efficiency by some margin by reducing the amount of masks
that are communicated over the network, but will likely take longer to complete

Is currently the communication method is used in the parallelisation wrapper

TODO: Implement a protocol to work without a look-up table.
      Standardise the communication buffer.
'''
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
    MSG_TYPE_SEND_JOIN = 4
    MSG_TYPE_SEND_LEAVE = 5

    # message data (META_INF_IDX_MSG_DATA) values
    MSG_DATA_NULL = 0 # an empty message
    MSG_DATA_QUERY = 1
    MSG_DATA_MSK_REQ = 2
    MSG_DATA_MSK = 3
    MSG_DATA_META = 4

    # number of seconds to sleep/wait
    SLEEP_DURATION = 1

    # constants for the client-server
    TIMEOUT = 5 # Might not be used anymore
    TRIES = 1   # Handles how many times client will attempt to send some data

    # Task label size can be replaced with the embedding size.
    def __init__(self, embd_dim, mask_dim, logger, init_address, init_port, mode, mask_interval, addresses, ports):
        super(ParallelComm, self).__init__()
        self.embd_dim = embd_dim            # Dimensions of the the embedding
        self.mask_dim = mask_dim            # Dimensions of the mask for use in buffers. May no longer be needed
        self.logger = logger                # Logger object for logging CLI outputs.
        self.mode = mode                    # Communication operation mode. Currently only ondemand knowledge is implemented
        self.mask_interval = mask_interval  # Interval for mask communication in synchronised learning.

        # Address and port for this agent
        self.init_address = init_address
        self.init_port = int(init_port)

        # Address-port lookup table
        self.other_address = addresses
        self.other_ports = ports

        # Debugging prints
        print('ports:', self.other_ports)
        print('addresses:', self.other_address)
        print('mask size:', self.mask_dim)
        print('embedding size:', self.embd_dim)

    def _null_message(self, msg):
        # check whether message sent denotes or is none.
        if bool(msg[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_NULL):
            return True

        else:
            return False

    def client(self, data, address, port):
        '''
        Client implementation. Begins a TCP connection secured using SSL/TLS implementation to a trusted server host-port. Attempts to send the data.
        '''
        attempts = 0
        while attempts < ParallelComm.TRIES:        # Attempt to send the data a number of times. If successful, do not attempt to send again.
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s = ssl.wrap_socket(s, keyfile="key.pem", certfile="certificate.pem")
            data = pickle.dumps(data)
            try:
                s.connect((address, port))
                s.sendall(data)
                attempts += 1

            except:
                attempts += 1

            finally:
                s.close()

    # Peer joining network methods
    def send_join_net(self):
        '''
        Tells all agents in the existing network that this agent is joining. 
        '''
        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_JOIN]

        for i in range(len(self.other_address)):
            if self.other_address[i] == self.init_address and self.other_ports[i] == self.init_port: continue
            self.client(data, self.other_address[i], self.other_ports[i])
    def recv_join_net(self, data, world_size):
        '''
        If this agent receives data that states a new agent is joining the network. Update known peers and update the world size
        '''
        # In dynaminc implementation this will be updated to the trusted connections after some validation.
        address = data[0]           # new peer address
        port = data[1]              # new peer port

        # Update the world size
        world_size.value += 1
    
    # Peer leaving network methods
    def send_exit_net(self):
        '''
        Tells all other known agents in the network that this agent is about to leave
        '''
        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_LEAVE]

        for i in range(len(self.other_address)):
            if self.other_address[i] == self.init_address and self.other_ports[i] == self.init_port: continue
            self.client(data, self.other_address[i], self.other_ports[i])
    def recv_exit_net(self, data, world_size):
        '''
        If this agent receives data that states an agent is leaving the network. Update the known peers and update the world size
        '''
        # In dynamic implementation the known peers will be updated to remove the leaving agent
        address = data[0]           # leaving peer address
        port = data[1]              # leaving peer port

        # Update the world size
        world_size.value -= 1

    # Query send and recv functions
    def send_query(self, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        if embedding is None:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_QUERY, embedding]

        # Try to send a query to all known destinations. Skip the ones that don't work
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
        # if populated prepare metadata responses
        if other_agent_req is not None:
            # If the req is none, which it usually will be, just skip.
            #if other_agent_req['msg_data'] is None: pass

            #else:
            req_label_as_np = other_agent_req['embedding'].detach().cpu().numpy()
            #print(req_label_as_np, type(req_label_as_np), flush=True)

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

        if not meta_response:
            meta_response['dst_address'] = other_agent_req['sender_address']
            meta_response['dst_port'] = other_agent_req['sender_port']
            meta_response['mask_reward'] = torch.inf
            meta_response['dist'] = torch.inf
            meta_response['resp_embedding'] = None

        return meta_response
    def send_meta(self, meta_response):
        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_META]

        if meta_response:
            dst_address = meta_response['dst_address']
            dst_port = meta_response['dst_port']
            mask_reward = meta_response['mask_reward']
            distance = np.float64(meta_response['dist'])
            embedding = meta_response['resp_embedding']

            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            if distance == torch.inf and mask_reward == torch.inf:
                data.append(ParallelComm.MSG_DATA_NULL)
            
            else:
                data.append(ParallelComm.MSG_DATA_META)
                data.append(mask_reward)
                data.append(distance)
                data.append(embedding)

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

        print(Fore.YELLOW + f'{metadata}')
        print(Fore.YELLOW + f'{knowledge_base}')
            
        # if not results something bad has happened
        if len(metadata) > 0:
            # Sort received meta data by smallest distance (primary) and highest reward (secondary)
            # using full bidirectional multikey sorting (fancy words for such a simple concept)
            metadata = {k: metadata[k] for k in sorted(metadata, key=lambda d: (metadata[d]['dist'], -metadata[d]['mask_reward']))}
            print(Fore.YELLOW + 'Metadata responses sorted:')
            for item in metadata:
                print(Fore.YELLOW + f'{item}')

            
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
        print(Fore.YELLOW + f'SEND_MSK_REQ: {send_msk_requests}', flush=True)

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
                    data.append(ParallelComm.MSG_DATA_NULL)
                        
                else:
                    # Otherwise we want the agent's mask
                    data.append(ParallelComm.MSG_DATA_MSK_REQ)
                    data.append(embedding) # NOTE deepcopy?

                # Send out the mask request or rejection to each agent that sent metadata
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


    # Mask response processing, send and recv functions
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


    # Event handler wrappers. This is done so the various functions for each event can be run in a single thread.
    def validate_connection(self, data, world_size):
        '''
        Event handler for validating a new agent that is attempting to join the existing network.
        '''
        self.recv_join_net(data, world_size)
        return
    def update_network(self, data, world_size):
        '''
        Event handler for an existing agent exiting the network
        '''
        self.recv_exit_net(data, world_size)
        return
    def query(self, data, knowledge_base):
        '''
        Event handler for receiving a query from another agent. Unpacks the data buffer, processes the response and sends some response if necessary.
        '''
        other_agent_req = self.recv_query(data)
        self.logger.info(Fore.GREEN + f'other agent request: {other_agent_req}')
        meta_response = self.proc_meta(other_agent_req, knowledge_base)
        self.send_meta(meta_response)
        return
    def add_meta(self, data, metadata):
        '''
        Event handler for receiving some metadata. Appends some new metadata to the collection.
        '''
        other_agent_meta, address, port = self.recv_meta(data)
        if address is not None and port is not None:
            metadata[address + ':' + str(port)] = other_agent_meta

        self.logger.info(Fore.YELLOW + f'Metadata collected: {metadata}')
        return
    def pick_meta(self, metadata, knowledge_base):
        '''
        Event handler for picking the best agent based on whatever metadata it has collected.
        '''
        self.logger.info(Fore.YELLOW + f'Time to select best agent!! :DDD')
        mask_req, best_agent_id, best_agent_rw = self.proc_mask_req(metadata, knowledge_base)
        metadata = {} #reset metadata dictionary
        self.send_mask_req(mask_req)
        return best_agent_id, best_agent_rw 
    def req(self, data, queue_label_send, queue_mask_recv):
        '''
        Event handler for mask requests. Unpacks the data buffer, processes the response and sends mask.
        '''
        mask_req = self.recv_mask_req(data)
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
        self.send_mask(mask_resp)
        return
    
    def server(self, knowledge_base, queue_mask, queue_mask_recv, queue_label_send, world_size):
        '''
        Server implementation. Binds a socket to a specified port and listens for incoming communication requests. If the connection is accepted using SSL/TLS handshake
        then the connection is secured and data is transferred. Once the data is recevied, an event is triggered based on the contents of the deserialised data.
        '''

        # Initialise a socket and wrap it with SSL
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s = ssl.wrap_socket(s, server_side=True, keyfile='key.pem', certfile='certificate.pem')

        # Bind the socket to the chosen address-port and start listening for connections
        s.bind((self.init_address, self.init_port))
        s.listen(1)

        metadata = {}

        while True:
            try:
                recv_mask = None
                best_agent_rw = None
                best_agent_id = None
                recv_embedding = None

                # Accept the connection
                conn, addr = s.accept()
                with conn:
                    self.logger.info(Fore.CYAN + f'\nConnected by {addr}')
                    while True:
                        try:
                            # Receive the data onto a buffer
                            data = conn.recv(4096)
                            if not data: break

                            # Deseralize the data
                            data = pickle.loads(data)
                            self.logger.info(Fore.CYAN + f'Received {data!r}')

                            # Event handling
                            # Agent attempting to join the network
                            if data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_JOIN:
                                with mpd.Pool(processes=1) as t_validation:
                                    _ = t_validation.apply_async(self.validate_connection, (data, world_size))
                                
                                #t_validation = mpd.Pool(processes=1)
                                #_ = t_validation.apply_async(self.validate_connection, (data, world_size))
                                #t_validation.close()

                            # Agent is leaving the network
                            #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_LEAVE:
                            #    with mpd.Pool(processes=1) as t_leave:
                            #        _ = t_leave.apply_async(self.update_network, (data, world_size))

                            # Agent is sending a query
                            elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_QUERY:
                                with mpd.Pool(processes=1) as t_query:
                                    _ = t_query.apply_async(self.query, (data, knowledge_base))

                            # Agent is sending some task distance and reward information
                            #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_META:
                            #    # Select best agent to get mask from
                            #    if len(metadata) == world_size.value - 1:
                            #        with mpd.Pool(processes=1) as t_pick:
                            #            result = t_pick.apply_async(self.pick_meta, (metadata, knowledge_base))
                            #            best_agent_id, best_agent_rw = result.get()

                                # Update the list of data
                            #    with mpd.Pool(processes=1) as t_meta:
                            #        _ = t_meta.apply_async(self.add_meta, (data, metadata))

                            # Agent is sending a direct request for a mask
                            #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_REQ:
                            #    with mpd.Pool(processes=1) as t_req:
                            #        _ = t_req.apply_async(self.req, (data, queue_label_send, queue_mask_recv))

                            # Another agent is sending a mask to this agent
                            #elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_MASK:
                            #    with mpd.Pool(processes=1) as t_msk:
                            #        result = t_msk.apply_async(self.recv_mask, (data, best_agent_id))
                            #        recv_mask, recv_embedding = result.get()
                            #        best_agent_id = None
                        
                        # Handles a connection reset by peer error that I've noticed when running the code. For now it just catches 
                        # the exception and moves on to the next connection.
                        except socket.error as e:
                            if e.errno != ECONNRESET: raise
                            print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                            pass
                            
                # Send knowledge to the agent if anything is available otherwise do nothing.
                if recv_mask is not None and best_agent_rw is not None and best_agent_id is not None and recv_embedding is not None:
                    queue_mask.put((recv_mask, best_agent_rw, best_agent_id, recv_embedding))

            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            except KeyboardInterrupt as e:
                self.send_exit_net()
    
    # Main loop + listening server initialisation
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base, world_size):
        '''
        Main communication function. Sets up the server and client. Distributes queues for interactions between the communication and agent processes.
        '''
        # Initialise the listening server
        self.logger.info('Starting the server...')
        p_server = mp.Process(target=self.server, args=(knowledge_base, queue_mask, queue_mask_recv, queue_label_send, world_size))
        p_server.start()

        # Attempt to join an existing network.
        self.logger.info('Attempting to join an existing network...')
        self.send_join_net()

        # Initialise the client loop
        self.logger.info('Starting the client...')
        while True:
            print()
            # Do some checks on the agent/communication interaction queues and perform actions based on those
            shell_iterations = queue_loop.get()     # This makes the communication module synchronised to the agent. If we remove this the communication module will be on speed. Nobody knows what will happen if this is removed. Do not remove... or maybe do. Idk. Users discretion. Good luck.
            self.logger.info(Fore.GREEN + f'Knowledge base in this iteration: {knowledge_base}')
            self.logger.info(Fore.GREEN + f'World size in this iteration: {world_size.value}')

            # Get an embedding to query for. If no embedding to query for then do nothing.
            try:
                msg = queue_label.get_nowait()
                # Send out a query when shell iterations matches mask interval if the agent is working on a task
                if world_size.value > 1:
                    print('Sending msg to other servers')
                    if shell_iterations % 1 == 0:
                        self.send_query(msg)

            except Empty: continue

            '''try:
                masks_list = queue_mask_recv.get_nowait()
            except Empty:
                masks_list = []

            if len(masks_list) > 0:
                pass
                # Send some masks?'''
            
            

    ### Old functions and methods. For reference
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
    
    # meta() with timer
    '''
    def meta(self, data, metadata, knowledge_base, select):
        other_agent_meta, address, port = self.recv_meta(data)
        if address is not None and port is not None:
            metadata[address + ':' + str(port)] = other_agent_meta

        self.logger.info(Fore.YELLOW + f'Metadata collected: {metadata}')

        if select == True:
            self.logger.info(Fore.YELLOW + f'Time to select best agent!! :DDD')
            mask_req, best_agent_id, best_agent_rw = self.proc_mask_req(metadata, knowledge_base)
            self.send_mask_req(mask_req)
            return False, best_agent_id, best_agent_rw
        
        return True, None, None
    '''

    # server() with meta timeout
    '''
    def server(self, knowledge_base, queue_mask, queue_mask_recv, queue_label_send):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s = ssl.wrap_socket(s, server_side=True, keyfile='key.pem', certfile='certificate.pem')

        s.bind((self.init_address, self.init_port))
        s.listen(1)

        timer = False
        select = False
        start_time = 0

        metadata = {}

        while True:
            recv_mask = None
            best_agent_rw = None
            best_agent_id = None
            recv_embedding = None

            conn, addr = s.accept()
            with conn:
                self.logger.info(Fore.CYAN + f'\nConnected by {addr}')
                while True:
                    try:
                        data = conn.recv(4096)
                        if not data: break

                        data = pickle.loads(data)
                        self.logger.info(Fore.CYAN + f'Received {data!r}')

                        # Event handling
                        if data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_QUERY:
                            t_query = mpd.Pool(processes=1)
                            _ = t_query.apply_async(self.query, (data, knowledge_base))
                            t_query.close()

                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_META:
                            if timer == False:
                                start_time = time.time()
                                timer = True

                            else:
                                time_elapsed = time.time() - start_time
                                if time_elapsed >= 5000:
                                    select = True
                                    timer = False


                            #if select == False:
                            #    t_pick = mpd.Pool(processes=1)
                            #    result = t_pick.apply_async(self.pick_meta, (data, metadata))
                            #    best_agent_id, best_agent_rw = result.get()
                            #    t_pick.close()
                            #    launched = True

                            # Wait for 10 seconds or 0.5s*num_agents. Whichever comes first
                            # Maybe fork a new thread here to start the sorting process after time and use the metadata list.
                            # thread will start, and wait for x amount of time. i.e., sleep(x) and then carry out the sort-select process and return this to the agent.
                            # Will have to figure out how to get the information back.
                            # We are going to run out of cores/threads very soon :^)

                            t_meta = mpd.Pool(processes=1)
                            result = t_meta.apply_async(self.meta, (data, metadata, knowledge_base, select))
                            select, best_agent_id, best_agent_rw = result.get()
                            t_meta.close()

                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_REQ:
                            t_req = mpd.Pool(processes=1)
                            _ = t_req.apply_async(self.req, (data, queue_label_send, queue_mask_recv))
                            t_req.close()

                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_MASK:
                            recv_mask, recv_embedding = self.recv_mask(data, best_agent_id)
                            best_agent_id = None
                    
                    except socket.error as e:
                        if e.errno != ECONNRESET: raise
                        print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                        pass
                        
            if recv_mask is not None and best_agent_rw is not None and best_agent_id is not None and recv_embedding is not None:
                queue_mask.put((recv_mask, best_agent_rw, best_agent_id, recv_embedding))
    '''

    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base, world_size):
        '''
        Parallelisation function for communication loop.
        '''
        p_client = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base, world_size))
        p_client.start()

'''
Evaluation variation of the standard communication module.

TODO: Implement the evaluation variation of the module.
'''
class ParallelCommEval(object):
    def __init__(self):
        pass

'''
Omniscent agent variation of the standard communication module

Currently the idea is to absorb the attributes of the standard communication module and then replace it in a new process
when an agent has been promoted to become to new omniscent.

Ideally the omniscent should be the strongest agent both computationally and in terms of bandwidth/networking capabilities.
Not sure how it will work but we will see.

TODO: Figure out what the omniscent agent actually is and implement this variation of the module.
'''
class ParallelCommOmniscent(object):
    def __init__(self):
        pass