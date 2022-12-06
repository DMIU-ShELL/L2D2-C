# -*- coding: utf-8 -*-
#   _________                                           .__                  __   .__                 
#   \_   ___ \   ____    _____    _____   __ __   ____  |__|  ____  _____  _/  |_ |__|  ____    ____  
#   /    \  \/  /  _ \  /     \  /     \ |  |  \ /    \ |  |_/ ___\ \__  \ \   __\|  | /  _ \  /    \ 
#   \     \____(  <_> )|  Y Y  \|  Y Y  \|  |  /|   |  \|  |\  \___  / __ \_|  |  |  |(  <_> )|   |  \
#    \______  / \____/ |__|_|  /|__|_|  /|____/ |___|  /|__| \___  >(____  /|__|  |__| \____/ |___|  /
#           \/               \/       \/             \/          \/      \/                        \/ 
#
#                                      Now featuring shared memory variables.
#                                                 (╯°□°)╯︵ ┻━┻
from colorama import Fore
from copy import deepcopy
import datetime
import multiprocessing as mp
import multiprocessing.dummy as mpd
import os
import pickle
import socket
import ssl
import struct
import time
from errno import ECONNRESET
from queue import Empty

import numpy as np
import torch



class Address(object):
    """
    Helper class for storing the address and port for individual agents. Makes it easier to store the information.
    """
    def __init__(self, inet4, port):
        self.inet4 = inet4
        self.port = port


'''
Revised communication class. Contains the new implementation of the communication module
Improves on the bandwidth efficiency by reducing the amount of mask transfers. Performance 
improved using concurrency implemented using multiprocessing/multithreading.

SSL/TLS is implemented but disabled currently, as it unfortunately causes issues with the mask
transfer for the moment.

Further changes have been made to implement a client-server event-based architecture which 
improves the flexibility of the system.

TODO: Implement a protocol to work without a look-up table.
      Standardise the communication buffer.
      Figure out how to re-introduce SSL/TLS without compromising mask transfer.
      Can probably reduce alot of the code in terms of memory consumption by reusing 
       alot of the lists instead of creating new dictionaries everywhere, although 
       it is more readable that way.
'''
class ParallelComm(object):
    # DETECT MODULE CONSTANTS
    # Threshold for embedding/tasklabel distance (similarity)
    # This should be taken from the detect module eventually
    THRESHOLD = 0.0

    # SSL/TLS Parameters
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

    # number of seconds to sleep/wait
    SLEEP_DURATION = 1

    # Task label size can be replaced with the embedding size.
    def __init__(self, embd_dim, mask_dim, logger, init_address, init_port, mode, mask_interval, ref_address=None, ref_port=None):
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
        # If there is a reference then try to connect
        if ref_address is not None and ref_port is not None:
            self.reference_table = [Address(inet4=ref_address, port=ref_port)]

        self.query_table = []   # Initialise an empty list to store Address() objects in for querying and communication

        # Debugging prints
        print(f'listening server params ->\naddress: {self.init_address}\nport: {self.init_port}\n')


        print(f'Reference address: {self.reference_table[0].inet4}, {self.reference_table[0].port}')
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

    def client(self, data, address, port, world_size):
        """
        Client implementation. Begins a TCP connection secured using SSL/TLS to a trusted server ip-port. Attempts to send the serialized bytes.
        
        Args:
            data: A list to be sent to another agent.
            address: The ip of the destination.
            port: The port of the destination.
        """
        _data = pickle.dumps(data, protocol=5)

        # Attempt to send the data a number of times. If successful do not attempt to send again.     
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.settimeout(2.0)
        #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #s = ssl.wrap_socket(s, keyfile=ParallelComm.KEYPATH, certfile=ParallelComm.CERTPATH, ssl_version=ssl.PROTOCOL_TLSv1_2)      # Uncomment to enable SSL/TLS security. Currently breaks when transferring masks.

        try:
            s.connect((address, port))
            _data = struct.pack('>I', len(_data)) + _data
            s.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            # Try to remove the ip and port that failed from the query table
            try: self.query_table.remove(next((x for x in self.query_table if x.inet4 == address and x.port == port)))  # Finds the next Address object with inet4==address and port==port and removes it from the query table.
            except: pass
            world_size -= 1

        finally: s.close()


    ### Methods for agents joining a network
    def send_join_net(self):
        """
        Sends a join request to agents in an existing network. Sends a buffer containign this agent's ip and port to the reference destinations.
        """

        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_JOIN]

        for addr in self.reference_table:
            self.client(data, addr.inet4, addr.port)
    def recv_join_net(self, data, world_size):
        """
        Event handler. Updates the known peers and world size if a network join request is received from another agent. Sends back
        the current query table.

        Args:
            data: A list received from another agent.
            world_size: A shared memory variable of type int() used to keep track of the size of the fully connected network.
        """
        address = data[ParallelComm.META_INF_IDX_ADDRESS]        # new peer address
        port = data[ParallelComm.META_INF_IDX_PORT]              # new peer port

        # Update the world size, query and reference table.
        world_size.value += 1



        # Send the current query table to the newly joining agent without the ip-port of the new agent.
        response = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_TABLE, self.query_table]
        self.client(response, address, port)


    ### Methods for agents leaving a network
    def send_exit_net(self):
        """
        Sends a leave notification to all other known agents in the network.
        """

        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_LEAVE]

        for i in range(len(self.other_address)):
            if self.other_address[i] == self.init_address and self.other_ports[i] == self.init_port: continue
            self.client(data, self.other_address[i], self.other_ports[i])
    def recv_exit_net(self, data, world_size):
        """
        Updates the known peers and world size if a network leave notification is recieved from another agent.
        
        Args:
            data: A list received from another agent.
            world_size: A shared memory variable of type int() used to keep track of the size of the fully connected network.
        """

        # In dynamic implementation the known peers will be updated to remove the leaving agent
        address = data[0]           # leaving peer address
        port = data[1]              # leaving peer port

        # Update the world size
        world_size.value -= 1

        return address, port

    ### Query send and recv functions
    def send_query(self, embedding, world_size):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            embedding: A torch tensor containing an embedding
        """

        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        if embedding is None:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_QUERY, ParallelComm.MSG_DATA_QUERY, np.array(embedding)]

        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in self.query_table:
            self.client(data, addr.inet4, addr.port, world_size)

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
        ret['msg_type'] = int(buffer[ParallelComm.META_INF_IDX_MSG_TYPE])
        ret['msg_data'] = int(buffer[ParallelComm.META_INF_IDX_MSG_DATA])
        ret['embedding'] = torch.tensor(buffer[ParallelComm.META_INF_IDX_TASK_SZ])

        return ret
    

    ### Metadata pre-processing, send and recv functions
    def proc_meta(self, other_agent_req, knowledge_base):
        """
        Processes a query for an embedding and produces a response to send back to the requesting agent.
        
        Args:
            other_agent_req: A dictionary containing the information for the query request.
            knowledge_base: A shared memory variable consisting of a dictionary to store the task embeddings and rewards accumulated.
        
        Returns:
            meta_response: A dictionary containing the response information.
        """

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
    def send_meta(self, meta_response, world_size):
        """
        Sends a response to a query with the distance, reward and this agent's embedding to the requesting agent.
        
        Args:
            meta_response: A dictionary containing the response information.
        """

        data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_META]

        if meta_response:
            dst_address = str(meta_response['dst_address'])
            dst_port = int(meta_response['dst_port'])
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
                data.append(np.array(embedding))

            self.client(data, dst_address, dst_port, world_size)
    def recv_meta(self, buffer):
        """
        Unpacks a response containing the distance, mask reward and an embedding from another agent.

        Args:
            buffer: A list received from another agent.

        Returns:
            ret: A dictionary containing the unpacked information. Default {None, None, 0.0, torch.inf, None}
            ret['address']: A string indicating the ip address of the sending agent.
            ret['port']: An integer indicating the port of the sending agent.
        """

        ret = {}
        if self._null_message(buffer):
            ret['address'] = str(buffer[ParallelComm.META_INF_IDX_ADDRESS])
            ret['port'] = int(buffer[ParallelComm.META_INF_IDX_PORT])
            ret['mask_reward'] = 0.0
            ret['dist'] = torch.inf
            ret['embedding'] = None
        
        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            # Code should never reach this point. If it does then reality has broken.
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_META:
            ret['address'] = str(buffer[ParallelComm.META_INF_IDX_ADDRESS])
            ret['port'] = int(buffer[ParallelComm.META_INF_IDX_PORT])
            ret['mask_reward'] = float(buffer[ParallelComm.META_INF_IDX_MSK_RW])
            ret['dist'] = float(buffer[ParallelComm.META_INF_IDX_DIST])
            ret['embedding'] = buffer[ParallelComm.META_INF_IDX_TASK_SZ_]   # This is an np array

        return ret, ret['address'], ret['port']


    ### Mask request pre-processing, send and recv functions
    def proc_mask_req(self, metadata, knowledge_base):
        """
        Processes a response to any received distance, mask reward, embedding information.
        
        Args:
            metadata: A dictionary containing the unpacked information from another agent.
            knowledge_base: A shared memory variable containing a dictionary which consists of all the task embeddings and corresponding reward.

        Returns:
            send_msk_requests: A dictionary containing the response to the information received.
            best_agent_id: A dictionary containing the ip-port pair for the selected agent.
            best_agent_rw: A dictionary containing the embedding-reward pair from the information received.
        """

        send_msk_requests = []
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

            for key, data_dict in metadata.items():
                # Do some checks to remove to useless results
                if key == str(self.init_address + ':' + str(self.init_port)): continue
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
                                    send_msk_requests.append(data_dict)
                                    # Make a note of the best agent id in memory of this agent
                                    # We will use this later to get the mask from the best agent
                                    best_agent_id = {recv_address: recv_port}
                                    best_agent_rw[tuple(recv_label)] = np.around(recv_msk_rw, 6)
                                    break

                            # If we don't have any knowledge present for the task then get the mask 
                            # anyway from the best agent.
                            else:
                                send_msk_requests.append(data_dict)
                                best_agent_id = {recv_address: recv_port}
                                best_agent_rw[tuple(recv_label)] = np.around(recv_msk_rw, 6)
                                break

        return send_msk_requests, best_agent_id, best_agent_rw
    def send_mask_req(self, send_msk_requests, world_size):
        """
        Sends a request for a specific mask to a specific agent.
        
        Args:
            send_msk_requests: A dictionary containing the information required send a request to an agent.
        """

        print(Fore.YELLOW + f'SEND_MSK_REQ: {send_msk_requests}', flush=True)
        if send_msk_requests:
            for data_dict in send_msk_requests:
                dst_address = str(data_dict['address'])
                dst_port = int(data_dict['port'])
                embedding = data_dict.get('embedding', None)

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
                    data.append(np.array(embedding)) # NOTE deepcopy?

                print(Fore.YELLOW + f'Buffer to send: {data}')
                # Send out the mask request or rejection to each agent that sent metadata
                self.client(data, dst_address, dst_port, world_size)
    def recv_mask_req(self, buffer):
        """
        Unpacks a mask request data buffer received from another agent.
        
        Args:
            buffer: A list received from another agent.

        Returns:
            ret: A dictionary containing the unpacked mask request from another agent.
        """

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
            ret['embedding'] = torch.tensor(buffer[ParallelComm.META_INF_IDX_TASK_SZ])

        return ret


    ### Mask response processing, send and recv functions
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
        print(f'\n{Fore.CYAN} Inside proc_mask()')
        if mask_req:
            print(Fore.CYAN + f'Mask request: {mask_req}', flush=True)
            queue_label_send.put((mask_req))
            return queue_mask_recv.get()
    def send_mask(self, mask_resp, world_size):
        """
        Sends a mask response to a specific agent.
        
        Args:
            mask_resp: A dictionary consisting of the information to send to a specific agent.    
        """
        if mask_resp:
            dst_address = str(mask_resp['sender_address'])
            dst_port = int(mask_resp['sender_port'])
            embedding = mask_resp.get('embedding', None)
            mask = mask_resp.get('mask', None)
            print(mask.dtype)

            data = [self.init_address, self.init_port, ParallelComm.MSG_TYPE_SEND_MASK]

            if mask is None or embedding is None:
                data.append(ParallelComm.MSG_DATA_NULL)

            else:
                data.append(ParallelComm.MSG_DATA_MSK)
                data.append(np.array(mask))
                data.append(np.array(embedding))

            print(f'{Fore.CYAN}Mask buffer to send: {data}')
            
            self.client(data, dst_address, dst_port, world_size)
    def recv_mask(self, buffer, best_agent_id):
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

        print()
        print(buffer)
        print(best_agent_id)

        if buffer[ParallelComm.META_INF_IDX_MSG_DATA] == torch.inf:
            pass

        elif buffer[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
            if {buffer[ParallelComm.META_INF_IDX_ADDRESS]: buffer[ParallelComm.META_INF_IDX_PORT]} == best_agent_id:
                received_mask = torch.tensor(buffer[4])
                received_label = torch.tensor(buffer[5])
        else:
            pass

        return received_mask, received_label


    # Event handler wrappers. This is done so the various functions for each event can be run in a single thread.
    def query(self, data, knowledge_base, world_size):
        """
        Event handler for receiving a query from another agent. Unpacks the buffer received from another agent, processes the request and sends some response if necessary.
        
        Args:
            data: A list received from another agent.
            knowledge_base: A shared memory variable of type dict() containing embedding-reward pairs for task embeddings observed by the agent.    
        """

        other_agent_req = self.recv_query(data)
        self.logger.info(Fore.GREEN + f'other agent request: {other_agent_req}')
        meta_response = self.proc_meta(other_agent_req, knowledge_base)
        self.send_meta(meta_response, world_size)

        return other_agent_req['sender_address'], other_agent_req['sender_port']
        
    def add_meta(self, data, metadata, knowledge_base, world_size):
        """
        Event handler for receving some distance, mask reward and embedding information. Appends the data to a collection of data received from all other agents.

        Args:
            data: A list received from another agent.
            metadata: A list to store responses from all known agents.
        """

        other_agent_meta, address, port = self.recv_meta(data)
        if address is not None and port is not None:
            metadata[address + ':' + str(port)] = other_agent_meta

        self.logger.info(Fore.YELLOW + f'Metadata collected: {metadata}')

        # Select best agent to get mask from
        if len(metadata) == world_size.value:
            t_pick = mpd.Pool(processes=1)
            best_agent_id, best_agent_rw = t_pick.apply_async(self.pick_meta, (metadata, knowledge_base)).get()
            t_pick.close()
            del t_pick

            print(Fore.CYAN + f'State after picking best agent {metadata}')
            return best_agent_id, best_agent_rw
        
        return None, None
    def pick_meta(self, metadata, knowledge_base):
        """
        Event handler to pick a best agent from the information it has collected at this point. Resets the metadata list.
        
        Args:
            metadata: A list containing the responses to a query from all other agents.
            knowledge_base: A shared memory dictionary containing the embedding-reward pairs for all observed task embeddings.
            
        Returns:
            best_agent_id: A dictionary containing a ip-port pair for the selected agent.
            best_agent_rw: A dictionary containing an embedding-reward pair from the selected agent.
        """
        
        self.logger.info(Fore.YELLOW + f'Time to select best agent!! :DDD')
        mask_req, best_agent_id, best_agent_rw = self.proc_mask_req(metadata, knowledge_base)
        metadata.clear() #reset metadata dictionary
        self.send_mask_req(mask_req)
        return best_agent_id, best_agent_rw 
    def req(self, data, queue_label_send, queue_mask_recv, world_size):
        '''
        Event handler for mask requests. Unpacks the data buffer, processes the response and sends mask.
        '''
        """
        Event hander for mask requests. Unpacks the data buffer, processes the response and sends a mask.
        
        Args:
            data: A list received from another agent.
            queue_label_send: A shared memory queue to send an embedding to be converted by the agent module.
            queue_mask_recv: A shared memory queue to received a mask from the agent module.
        """

        mask_req = self.recv_mask_req(data)
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
        self.send_mask(mask_resp, world_size)


    # Listening server
    def server(self, knowledge_base, queue_mask, queue_mask_recv, queue_label_send, world_size):
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

        # Initialise a socket and wrap it with SSL
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #s = ssl.wrap_socket(s, server_side=True, keyfile=ParallelComm.KEYPATH, certfile=ParallelComm.CERTPATH, ssl_version=ssl.PROTOCOL_TLSv1_2)    # Uncomment to enable SSL/TLS security. Currently breaks when transferring masks.

        # Bind the socket to the chosen address-port and start listening for connections
        s.bind(('', self.init_port))
        s.listen(1)

        metadata = {}


        recv_mask = None
        best_agent_rw = None
        best_agent_id = None
        recv_embedding = None
        while True:
            # Accept the connection
            conn, addr = s.accept()
            with conn:
                self.logger.info(Fore.CYAN + f'\nConnected by {addr}')
                while True:
                    try:
                        # Receive the data onto a buffer
                        data = recv_msg(conn)
                        if not data: break
                        # Deseralize the data
                        data = pickle.loads(data)
                        self.logger.info(Fore.CYAN + f'Received {data!r}')

                        ### EVENT HANDLING
                        # Agent attempting to join the network
                        if data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_JOIN:
                            t_validation = mpd.Pool(processes=1)
                            t_validation.apply_async(self.recv_join_net, (data, world_size))
                            self.logger.info(Fore.CYAN + f'Data is a join req')
                            t_validation.close()
                            del t_validation    # saving memory? Not sure if this will have any major effect on memory consumption.

                        # Agent is sending a query table
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_TABLE:
                            self.logger.info(Fore.CYAN + f'Data is a query table')
                            # Update this agent's query table
                            self.query_table = data[3]  # Overwrite the empty table with the received query table. It should ideally contain all the ip-ports of agents in the network, except it's own ip and port.
                            self.query_table.append(Address(inet4=data[ParallelComm.META_INF_IDX_ADDRESS], port=data[ParallelComm.META_INF_IDX_PORT]))   # Append the reference agent ip-port
                            
                            # Update the reference table with any previously unseen ip-ports
                            #for addr in self.query_table:
                            #    if not any(x.inet4 == addr.inet4 and x.port == addr.port for x in self.reference_table):           # Uncomment to log new connections from the received query table to the reference table as well
                            #        self.reference_table.append(addr)                                                              # Currently handled by the query.

                            # Ideally this agent will never have a table with its own ip-port

                        # Another agent is leaving the network
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_LEAVE:
                            t_leave = mpd.Pool(processes=1)
                            _address, _port = t_leave.apply_async(self.recv_exit_net, (data, world_size)).get()

                            # Remove the ip-port from the query table for the agent that is leaving
                            try: self.query_table.remove(next((x for x in self.query_table if x.inet4 == addr.inet4 and x.port == addr.port)))  # Finds the next Address object with inet4==address and port==port and removes it from the query table.
                            except: pass

                            self.logger.info(Fore.CYAN + f'Data is a leave req')
                            t_leave.close()
                            del t_leave

                        # An agent is sending a query
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_QUERY:
                            t_query = mpd.Pool(processes=1)
                            _address, _port = t_query.apply_async(self.query, (data, knowledge_base, world_size)).get()

                            # If the agent receives a query from a location it has not previously encountered then
                            # update the query and reference tables.
                            if not any(x.inet4 == _address and x.port == _port for x in self.query_table):
                                self.query_table.append(Address(_address, _port))
                                self.reference_table.append(Address(_address, _port))

                            self.logger.info(Fore.CYAN + f'Data is a query')
                            t_query.close()
                            del t_query

                        # An agent is sending some task distance and reward information in response to a query
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_META:
                            # Update the list of data
                            t_meta = mpd.Pool(processes=1)
                            best_agent_id, best_agent_rw = t_meta.apply_async(self.add_meta, (data, metadata, knowledge_base, world_size)).get()
                            self.logger.info(Fore.CYAN + f'Data is metadata')
                            t_meta.close()
                            del t_meta

                        # An agent is sending a direct request for a mask
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_REQ:
                            t_req = mpd.Pool(processes=1)
                            t_req.apply_async(self.req, (data, queue_label_send, queue_mask_recv, world_size))
                            self.logger.info(Fore.CYAN + f'Data is a mask req')
                            t_req.close()
                            del t_req

                        # An agent is sending a mask
                        elif data[ParallelComm.META_INF_IDX_MSG_TYPE] == ParallelComm.MSG_TYPE_SEND_MASK:
                            t_msk = mpd.Pool(processes=1)
                            recv_mask, recv_embedding = t_msk.apply_async(self.recv_mask, (data, best_agent_id)).get()
                            self.logger.info(Fore.CYAN + f'Data is a mask')
                            t_msk.close()
                            del t_msk

                            self.logger.info(f'\n{Fore.WHITE}Mask: {recv_mask}\nReward: {best_agent_rw}\nSrc: {best_agent_id}\nEmbedding: {recv_embedding}\n')
                            # Send knowledge to the agent if anything is available otherwise do nothing.
                            if recv_mask is not None and best_agent_rw is not None and best_agent_id is not None and recv_embedding is not None:
                                print('SENDING DATA TO AGENT')
                                queue_mask.put((recv_mask, best_agent_rw, best_agent_id, recv_embedding))
                                recv_mask = None
                                best_agent_rw = None
                                best_agent_id = None
                                recv_embedding = None

                    # Handles a connection reset by peer error that I've noticed when running the code. For now it just catches 
                    # the exception and moves on to the next connection.
                    except socket.error as e:
                        if e.errno != ECONNRESET: raise
                        print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                        pass
    
    # Main loop + listening server initialisation
    def communication(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base, world_size):
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
        self.logger.info('Starting the server...')
        p_server = mp.Process(target=self.server, args=(knowledge_base, queue_mask, queue_mask_recv, queue_label_send, world_size, response_counter))
        p_server.start()

        # Attempt to join an existing network.
        

        # Initialise the client loop
        self.logger.info('Starting the client...')
        while True:

            # Attempt to connect to reference agent and get latest table. If the query table is reduced to original state then try to reconnect previous agents
            # using the reference table.
            # Unless there is no reference.
            if len(self.query_table) == 0:  # If query table is empty
                if len(self.reference_table) != 0:  # If there are reference locations to try to communicate to
                    self.logger.info('Attempting to join an existing network...')          # Uncomment to for the join/leave protocol
                    self.send_join_net()    # Send out a message to the reference(s)

            time.sleep(3)  # Wait some time before trying to perform queries.  

            try:
                print()
                # Do some checks on the agent/communication interaction queues and perform actions based on those
                shell_iterations = queue_loop.get() # Synchronises the querying client with the agent module.
                self.logger.info(Fore.GREEN + f'Knowledge base in this iteration: {knowledge_base}')
                self.logger.info(Fore.GREEN + f'World size in this iteration: {world_size.value}')

                # Get an embedding to query for. If no embedding to query for then do nothing.
                try:
                    msg = queue_label.get_nowait()
                    # Send out a query when shell iterations matches mask interval if the agent is working on a task
                    if world_size.value > 1 and shell_iterations % self.mask_interval == 0:
                        #self.logger.info(Fore.MAGENTA + f'Sending msg to other servers')
                        self.send_query(msg, world_size)

                except Empty: continue

            # Handles the agent crashing or stopping or whatever. Not sure if this is the right way to do this. Come back to this later.
            except (SystemExit, KeyboardInterrupt) as e:                           # Uncomment to enable the keyboard interrupt and system exit handling
                pass
                self.send_exit_net()
            

    def parallel(self, queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base, world_size):
        """
        Parallelisation method for starting the communication loop inside a seperate process.
        """

        p_client = mp.Process(target=self.communication, args=(queue_label, queue_mask, queue_label_send, queue_mask_recv, queue_loop, knowledge_base, world_size))
        p_client.start()

        #return p_client # consider returning the process object


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