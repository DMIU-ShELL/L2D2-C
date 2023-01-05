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
    DROPOUT = 0  # Value between 0 and 1 i.e, 0.25=25% dropout, 1=100% dropout, 0=no dropout

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
    def __init__(self, num_agents, embd_dim, mask_dim, logger, init_port, mode, reference, knowledge_base, manager, localhost):
        super(ParallelCommEval, self).__init__()
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
        self.expecting = manager.Value('b', True)


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

        
        try:
            sock.connect((address, port))
            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

            data = recv_msg(sock)


        except:
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
            # Try to remove the ip and port that failed from the query table
            #try: self.query_list.remove(next(item for item in self.query_list if item == (address, port)))
            #except: print('FAILED :(((((((')
            #self.world_size.value = len(self.query_list) + 1

            #print(f'New query list: {self.query_list}')
            #print(f'New world size: {self.world_size.value}')

        finally: sock.close()
    
    # Modified version of the client used by the send_query function. Has an additional bit of code to handle the mask response before querying the next agent in the query list
    def query_client(self, data, address, port, queue_mask):
        
        _data = pickle.dumps(data, protocol=5)

        # Attempt to send the data a number of times. If successful do not attempt to send again.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)

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

        received_mask, received_label, received_reward = None, None, None
        
        try:
            # Connect to target server
            sock.connect((address, port))
            _data = struct.pack('>I', len(_data)) + _data
            
            # Send formatted query
            sock.sendall(_data)
            self.logger.info(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

        except:
            self.logger.info(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')
            # Try to remove the ip and port that failed from the query table
            #try: self.query_list.remove(next(item for item in self.query_list if item == (address, port)))
            #except: print('FAILED :(((((((')
            #self.world_size.value = len(self.query_list) + 1

            #print(f'New query list: {self.query_list}')
            #print(f'New world size: {self.world_size.value}')
        finally: sock.close()

    ### Query send and recv functions
    def send_query(self, embedding, queue_mask):
        """
        Sends a query for knowledge for a given embedding to other agents known to this agent.
        
        Args:
            embedding: A torch tensor containing an embedding
        """

        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        #self.logger.info(Fore.GREEN + 'send_recv_req, req data: {0}'.format(embedding))

        if embedding is None:
            data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_QUERY, ParallelCommEval.MSG_DATA_NULL]

        else:
            data = [self.init_address, self.init_port, ParallelCommEval.MSG_TYPE_SEND_QUERY, ParallelCommEval.MSG_DATA_QUERY, np.array(embedding)]

        self.expecting.value = True
        # Try to send a query to all known destinations. Skip the ones that don't work
        for addr in self.query_list:
            if not self.expecting.value: break
            self.query_client(data, addr[0], addr[1], queue_mask)
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
        ret['embedding'] = torch.tensor(buffer[ParallelCommEval.META_INF_IDX_TASK_SZ])

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

            # Iterate through the knowledge base and compute the distances
            # If reward greater than 0
            # If distance is less than or equal to threshold
            # response = True (false by default)
            for tlabel, treward in self.knowledge_base.items():
                if treward > np.around(0.0, decimals=6):
                    if np.sum(abs(np.subtract(np_embedding, np.asarray(tlabel)))) <= ParallelCommEval.THRESHOLD:
                        other_agent_req['response'] = True
                        other_agent_req['reward'] = treward

        # Return the query request
        return other_agent_req
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
            dst_address = str(mask_resp['sender_address'])
            dst_port = int(mask_resp['sender_port'])
            embedding = mask_resp.get('embedding', None)
            mask = mask_resp.get('mask', None)
            reward = mask_resp.get('reward', None)
            response = mask_resp['response']
            #print(f'{Fore.RED}Mask type: {mask.dtype}{type(mask)}')

            data = [self.init_address, self.init_port, ParallelCommV2.MSG_TYPE_SEND_MASK]

            if response:
                data.append(ParallelCommV2.MSG_DATA_MSK)
                data.append(mask)
                data.append(embedding)
                data.append(reward)
            
            else:
                data.append(ParallelCommV2.MSG_DATA_NULL)
            
            self.logger.info(f'Sending mask response: {data}')
            self.client(data, dst_address, dst_port)
    
    # Event handler wrappers. This is done so the various functions for each event can be run in a single thread.
    def query(self, data, queue_label_send, queue_mask_recv):
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
        mask_req = self.proc_meta(other_agent_req)
        self.logger.info(f'Processes mask req: {mask_req}')

        # Get the label to mask conversion
        mask_resp = self.proc_mask(mask_req, queue_label_send, queue_mask_recv)
        self.logger.info(f'Processes mask resp: {mask_resp}')

        # Send the mask response back to the querying agent
        self.send_mask(mask_resp)
    def update_params(self, data):
        temp = list(self.query_list)
        self.query_list[:] = []
        self.query_list.extend(data[3])
        self.query_list.extend(temp)





    # Potentially deprecated
    ### Mask request pre-processing, send and recv functions
    ### Methods for agents joining a network
    def send_join_net(self):
        """
        Sends a join request to agents in an existing network. Sends a buffer containign this agent's ip and port to the reference destinations.
        """

        data = [self.init_address, self.init_port, ParallelCommV1.MSG_TYPE_SEND_JOIN]

        # Continue to loop through the reference table from start to end until one of the connects
        # TODO: In the future the table should include a last active date to sort by and remove pointless connections, otherwise the system
        #       will spend forever trying to find an active agent.


        # If we manage to connect to one of the peers in the reference_list then stop looking for connections. Best case this agent 
        # reconnects to the full network, worst case this agent reconnects to a single agent. Potentially, this new network will be 
        # re-discovered by some other agents and eventually the network will slowly heal again and reform into a complete network.
        while len(self.query_list) == 0 and len(self.reference_list) != 0:
            for addr in cycle(self.reference_list):
                #print(f"Reaching out to: {addr[0]}, {addr[1]}")
                if self.client(data, addr[0], addr[1]):
                    self.query_list.append((addr[0], addr[1]))
                    break
    def recv_join_net(self, data):
        """
        Event handler. Updates the known peers and world size if a network join request is received from another agent. Sends back
        the current query table.

        Args:
            data: A list received from another agent.
            world_size: A shared memory variable of type int() used to keep track of the size of the fully connected network.
        """
        address = data[ParallelCommV1.META_INF_IDX_ADDRESS]        # new peer address
        port = data[ParallelCommV1.META_INF_IDX_PORT]              # new peer port

        # Send this agent's query table and world size to the new agent.
        response = [self.init_address, self.init_port, ParallelCommV1.MSG_TYPE_SEND_TABLE, list(self.query_list)]
        self.client(response, address, port)

        # Update the query and reference lists
        self.query_list.append((address, port))
        self.reference_list.append((address, port))

    ### Methods for agents leaving a network
    def send_exit_net(self):
        """
        Sends a leave notification to all other known agents in the network.
        """

        data = [self.init_address, self.init_port, ParallelCommV1.MSG_TYPE_SEND_LEAVE]


        for addr in self.query_list:
            self.client(data, addr[0], addr[1])
    def recv_exit_net(self, data):
        """
        Updates the known peers and world size if a network leave notification is recieved from another agent.
        
        Args:
            data: A list received from another agent.
            world_size: A shared memory variable of type int() used to keep track of the size of the fully connected network.
        """

        # In dynamic implementation the known peers will be updated to remove the leaving agent
        address = data[0]           # leaving peer address
        port = data[1]              # leaving peer port


        # Remove address to from query list
        try: self.query_list.remove(next((x for x in self.query_list if x[0] == address and x[1] == port)))  # Finds the next Address object with inet4==address and port==port and removes it from the query table.
        except: pass

        return address, port
    def proc_mask_req(self, metadata):
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

        #self.logger.info(Fore.YELLOW + f'{metadata}')
        #self.logger.info(Fore.YELLOW + f'{self.knowledge_base}')
            
        # if not results something bad has happened
        if len(metadata) > 0:
            # Sort received meta data by smallest distance (primary) and highest reward (secondary)
            # using full bidirectional multikey sorting (fancy words for such a simple concept)
            metadata = {k: metadata[k] for k in sorted(metadata, key=lambda d: (metadata[d]['dist'], -metadata[d]['mask_reward']))}
            #print(Fore.YELLOW + 'Metadata responses sorted:')
            #for item in metadata:
            #    print(Fore.YELLOW + f'{item}')

            
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
                    
                    #self.logger.info(Fore.YELLOW + f'{recv_address}\n{recv_port}\n{recv_msk_rw}\n{recv_dist}\n{recv_label}\n')

                    # If the recv_dist is lower or equal to the threshold and a best agent
                    # hasn't been selected yet then continue
                    if recv_msk_rw != 0.0:
                        if recv_dist <= ParallelCommV2.THRESHOLD:
                            # Check if the reward is greater than the current reward for the task
                            # or if the knowledge even exists.
                            if tuple(recv_label) in self.knowledge_base.keys():
                                #self.logger.info(f'COMPARISON TAKES PLACE FOR CASE 1: {round(recv_msk_rw, 6)} > {0.75 * self.knowledge_base[tuple(recv_label)]}')
                                if 0.9 * round(recv_msk_rw, 6) > self.knowledge_base[tuple(recv_label)]:
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
                                #self.logger.info(f'NO COMPARISON. RUNNING CASE 2 AS KNOWLEDGE NOT IN KNOWLEDGE BASE YET')
                                send_msk_requests.append(data_dict)
                                best_agent_id = {recv_address: recv_port}
                                best_agent_rw[tuple(recv_label)] = np.around(recv_msk_rw, 6)
                                break

        self.logger.info(f'\n{Fore.YELLOW}Sending mask request: {send_msk_requests}')
        return send_msk_requests, best_agent_id, best_agent_rw
    def send_mask_req(self, send_msk_requests):
        """
        Sends a request for a specific mask to a specific agent.
        
        Args:
            send_msk_requests: A dictionary containing the information required send a request to an agent.
        """

        #print(Fore.YELLOW + f'SEND_MSK_REQ: {send_msk_requests}', flush=True)
        if send_msk_requests:
            for data_dict in send_msk_requests:
                dst_address = str(data_dict['address'])
                dst_port = int(data_dict['port'])
                embedding = data_dict.get('embedding', None)

                # Convert embedding label to tensor
                if isinstance(embedding, np.ndarray):
                    embedding = torch.tensor(embedding, dtype=torch.float32)

                data = [self.init_address, self.init_port, ParallelCommV2.MSG_TYPE_SEND_REQ]
                
                if embedding is None:
                    # If emb_label is none it means we reject the agent
                    data.append(ParallelCommV2.MSG_DATA_NULL)
                        
                else:
                    # Otherwise we want the agent's mask
                    data.append(ParallelCommV2.MSG_DATA_MSK_REQ)
                    data.append(np.array(embedding)) # NOTE deepcopy?

                #print(Fore.YELLOW + f'Buffer to send: {data}')
                # Send out the mask request or rejection to each agent that sent metadata
                self.client(data, dst_address, dst_port)
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

        elif buffer[ParallelCommV2.META_INF_IDX_MSG_DATA] == torch.inf:
            pass

        else:
            ret['sender_address'] = str(buffer[ParallelCommV2.META_INF_IDX_ADDRESS])
            ret['sender_port'] = int(buffer[ParallelCommV2.META_INF_IDX_PORT])
            ret['msg_type'] = int(buffer[ParallelCommV2.META_INF_IDX_MSG_TYPE])
            ret['msg_data'] = int(buffer[ParallelCommV2.META_INF_IDX_MSG_DATA])
            ret['embedding'] = torch.tensor(buffer[ParallelCommV2.META_INF_IDX_TASK_SZ])

        return ret
    def add_meta(self, data, metadata):
        """
        Event handler for receving some distance, mask reward and embedding information. Appends the data to a collection of data received from all other agents.

        Args:
            data: A list received from another agent.
            metadata: A list to store responses from all known agents.
        """

        other_agent_meta, address, port = self.recv_meta(data)
        if address is not None and port is not None:
            metadata[address + ':' + str(port)] = other_agent_meta

        #self.logger.info(Fore.YELLOW + f'Metadata collected: {metadata}')

        # Select best agent to get mask from
        if len(metadata) >= self.world_size.value - 1:
            best_agent_id, best_agent_rw = self.pick_meta(metadata)
            return best_agent_id, best_agent_rw
        
        return None, None
    def pick_meta(self, metadata):
        """
        Event handler to pick a best agent from the information it has collected at this point. Resets the metadata list.
        
        Args:
            metadata: A list containing the responses to a query from all other agents.
            knowledge_base: A shared memory dictionary containing the embedding-reward pairs for all observed task embeddings.
            
        Returns:
            best_agent_id: A dictionary containing a ip-port pair for the selected agent.
            best_agent_rw: A dictionary containing an embedding-reward pair from the selected agent.
        """
        
        #self.logger.info(Fore.YELLOW + f'Time to select best agent!! :DDD')
        mask_req, best_agent_id, best_agent_rw = self.proc_mask_req(metadata)
        metadata.clear() #reset metadata dictionary
        self.send_mask_req(mask_req)
        return best_agent_id, best_agent_rw 
    def req(self, data, queue_label_send, queue_mask_recv):
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
        self.send_mask(mask_resp)
    
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


        #context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        #context.load_cert_chain(certfile='certificates/certificate.pem', keyfile='certificates/key.pem')
        # Initialise a socket and wrap it with SSL

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
                        if data[ParallelCommV2.META_INF_IDX_MSG_TYPE] == ParallelCommV2.MSG_TYPE_SEND_TABLE:
                            self.logger.info(Fore.CYAN + 'Data is a query table')
                            self.update_params(data)
                            for addr in self.query_list: print(f'{Fore.GREEN}{addr[0], addr[1]}')

                        # An agent is sending a query
                        elif data[ParallelCommV2.META_INF_IDX_MSG_TYPE] == ParallelCommV2.MSG_TYPE_SEND_QUERY:
                            self.logger.info(Fore.CYAN + 'Data is a query')
                            self.query(data, queue_label_send, queue_mask_recv)


                        elif data[ParallelCommV2.META_INF_IDX_MSG_TYPE] == ParallelCommV2.MSG_TYPE_SEND_MASK:
                            self.logger.info(Fore.CYAN + 'Data is a mask')
                            # Unpack the received data
                            received_mask, received_label, received_reward, ip, port = self.recv_mask(data)

                            self.logger.info(f'{received_mask, received_label, received_reward, ip, port}')
                            # Send the reeceived information back to the agent process if condition met
                            if received_mask is not None and received_label is not None and received_reward is not None:
                                self.expecting.value = False
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
            try:
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
                    if int(np.random.choice(2, 1, p=[ParallelCommV2.DROPOUT, 1 - ParallelCommV2.DROPOUT])) == 1:  # Condition to simulate % communication loss
                        self.send_query(msg, queue_mask)



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

