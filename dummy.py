import socket
import struct
from colorama import Fore
import pickle
import multiprocessing as mp
import multiprocessing.dummy as mpd
from errno import ECONNRESET
import torch.nn.functional as F
import torch
import numpy as np

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
MSG_TYPE_SEND_QUERY_EVAL = 7

# message data (META_INF_IDX_MSG_DATA) values
MSG_DATA_NULL = 0 # an empty message
MSG_DATA_QUERY = 1
MSG_DATA_MSK_REQ = 2
MSG_DATA_MSK = 3
MSG_DATA_META = 4

init_address = '127.0.0.1'
init_port = 29501

query_list = []
query_list.append(('127.0.0.1', 29500))
world_size = 0


def client(data, address, port, is_query=False):
    _data = pickle.dumps(data, protocol=5)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            print('ATTEMPTING CONNECTION TO DESTINATION')
            sock.connect((address, port))
            _data = struct.pack('>I', len(_data)) + _data
            sock.sendall(_data)
            print(Fore.MAGENTA + f'Sending {data} of length {len(_data)} to {address}:{port}')

    except:
        print(Fore.MAGENTA + f'Failed to send {data} of length {len(_data)} to {address}:{port}')

def received_query(data):
    # Query to mask response pipeline
    def recv_query(buffer):
        sender_address = str(buffer[META_INF_IDX_ADDRESS])
        sender_port = int(buffer[META_INF_IDX_PORT])
        embedding = buffer[META_INF_IDX_TASK_SZ]
        sender_reward = buffer[-1]

        # Create a dictionary with the unpacked data
        ret = {
            'sender_address': sender_address,
            'sender_port': sender_port,
            'sender_embedding': embedding,
            'sender_reward': sender_reward
        }

        # Handle when receiving a query from an unknown agent
        if (sender_address, sender_port) not in query_list:
            client([init_address, init_port, MSG_TYPE_SEND_TABLE, list(query_list)], sender_address, sender_port)
            query_list.append((sender_address, sender_port))

        # Refresh the world_size value
        world_size = len(query_list) + 1

        return ret
    
    def proc_mask(query):
        """
        Find the most similar task record and get the internal task id if any satisfying entries found. Create response dictionary and return otherwise return NoneType.
        
        Args:
            query: A dictionary consisting of the response information to send to a specific agent.

        Returns:
            The mask_req dictionary with the converted mask now included.
        """

        # Get information from the registry for the closest key (internal task id)
        query['response_reward'] = 1.0
        query['response_similarity'] = 0.0
        query['response_task_id'] = 1
        query['response_embedding'] = torch.rand(7400)
        query['response_label'] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        query['response_mask'] = torch.FloatTensor(1, 109600).uniform_(-1, 1)

        return query
    
    def send_mask(response):
        """
        Sends a mask response to a specific agent.
        
        Args:
            mask_resp: A dictionary consisting of the information to send to a specific agent.    
        """
        data = [
            init_address,
            init_port,
            MSG_TYPE_SEND_MASK,
            MSG_DATA_MSK,
            response.get('response_mask', None),
            response.get('response_embedding', None),
            response.get('response_reward', None),
            response.get('response_label', None)
        ]

        print(f'Sending mask response: {data}')
        client(data, str(response['sender_address']), int(response['sender_port']))


    # Unpack the query from the other agent
    query = recv_query(data)
    print(Fore.MAGENTA + f'Received query: {query}')

    # Get the mask with the most task similarity, if any such mask exists in the network.
    response = proc_mask(query)
    print(Fore.WHITE + f'Processed mask request: {response}')

    # Send mask to querying agent if response is not NoneType
    if response is not None:
        send_mask(response)

    print('\n') 

def event_handler(data):
    ### EVENT HANDLING
    # An agent is sending a query
    if data[META_INF_IDX_MSG_TYPE] == MSG_TYPE_SEND_QUERY:
        print(Fore.YELLOW + 'Data is a query')
        received_query(data)

def server():
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

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the chosen address-port and start listening for connections
    sock.bind(('127.0.0.1', 29501))

    # Set backlog to the world size
    sock.listen(2)
    print('SERVER STARTED')

    while True:
        # Accept the connection
        conn, addr = sock.accept()

        with conn:
            print(Fore.CYAN + f'Connected by {addr}')
            while True:
                try:
                    # Receive the data onto a buffer
                    data = recv_msg(conn)
                    if not data: break
                    data = pickle.loads(data)
                    print(Fore.CYAN + f'Received {data!r}')

                    # Handle connection
                    handler = mpd.Pool(processes=1)
                    handler.apply_async(event_handler, (data, ))

                # Handles a connection reset by peer error that I've noticed when running the code. For now it just catches 
                # the exception and moves on to the next connection.
                except socket.error as e:
                    if e.errno != ECONNRESET: raise
                    print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                    pass


if __name__ == '__main__':
    server()