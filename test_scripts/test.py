import multiprocess as mp
import torch.distributed as dist
import random
import time
import socket
import argparse
import torch
import os
from _thread import *


class Communication(object):
    META_INF_IDX_PROC_ID = 0
    META_INF_IDX_MSG_TYPE = 1
    META_INF_IDX_MSG_DATA = 2

    META_INF_IDX_MSK_RW = 3
    META_INF_IDX_TASK_SZ = 3 # only for the send_recv_request buffer

    META_INF_IDX_DIST = 4
    META_INF_IDX_TASK_SZ_ = 5
    

    META_INF_IDX_MASK_SZ = 6
    
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

    def __init__(self, agent_id, num_agents, add_to_network, init_address):
        super(Communication, self).__init__()
        # System attributes
        self.agent_id = agent_id
        self.num_agents = num_agents

        # Networking attributes
        # If the agent has created the network then add_to_network is false
        # if so then the agent is already is in the network
        self.in_network = not add_to_network
        self.comm_init_str = init_address
        self.group = None
        

        emb_label_sz = 3
        # Communciation buffer attributes
        self.buff_send_recv_req = [torch.ones(Communication.META_INF_IDX_TASK_SZ + emb_label_sz, ) * torch.inf]
        #self.buff_recv_resp = [torch.ones(Communication.META_INF_IDX_MASK_SZ + mask_sz, ) * torch.inf]
        #self.buff_send_resp = [torch.ones(Communication.META_INF_IDX_MASK_SZ + mask_sz, ) * torch.inf]
        #self.buff_send_recv_msk_req = torch.ones(Communication.META_INF_IDX_TASK_SZ + emb_label_sz, ) * torch.inf

    def init_dist():
        '''
        Initialise the process group. Used if the agent is creating a new network
        '''
        self.group = dist.init_process_group(backend='gloo', init_method=self.comm_init_str, rank=self.agent_id, world_size=self.num_agents)

    def re_init_dist():
        '''
        Re-initialise the process group. Used if the agent is re-initialising the network with
        update parameters. (i.e., adding or removing agents)
        '''
        # Update the internal knowledge about the network.
        # Increase the num_agents to match the new world size when a new agent is added
        # Rank will remain the same
        # Each agent will receive their rank and the new world size.
        self.num_agents += 1

        # Destroy the original process group and initialise a new one.
        # Each agent will have to wait for other agents to get setup with the new process group
        # within the network. Once all agents are in the new process group. Then the system can continue
        # Highly delicate process.
        dist.destroy_process_group(self.group)
        self.init_dist()

    def send_receive_request(self, emb_label):
        if isinstance(emb_label, np.ndarray):
            emb_label = torch.tensor(emb_label, dtype=torch.float32)

            
        self.logger.info('send_recv_req, req data: {0}'.format(emb_label))
        # from message to send from agent (current node), can be NULL message or a valid
        # request based on given task label
        data = torch.ones_like(self.buff_send_recv_req[0]) * torch.inf

        # Send the task label or embedding to the other agents.
        data[Communication.META_INF_IDX_PROC_ID] = self.agent_id
        data[Communication.META_INF_IDX_MSG_TYPE] = Communication.MSG_TYPE_SEND_REQ

        if emb_label is None:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_NULL
        else:
            data[Communication.META_INF_IDX_MSG_DATA] = Communication.MSG_DATA_TSK
            data[Communication.META_INF_IDX_TASK_SZ : ] = emb_label # NOTE deepcopy?


        # actual send/receive
        self.handle_send_recv_req = dist.all_gather(tensor_list=self.buff_send_recv_req, \
            tensor=data, async_op=True)

        self.handle_send_recv_req.wait()


        # briefly wait to see if other agents will send their request
        #time.sleep(Communication.SLEEP_DURATION)
        
        # check buffer for incoming requests
        idxs = list(range(len(self.buff_send_recv_req)))
        idxs.remove(self.agent_id)
        ret = []
        for idx in idxs :
            buff = self.buff_send_recv_req[idx]

            if self._null_message(buff):
                ret.append(None)
            else:
                self.logger.info('send_recv_req: request received from agent {0}'.format(idx))
                d = {}
                d['sender_agent_id'] = int(buff[Communication.META_INF_IDX_PROC_ID])
                d['msg_type'] = int(buff[Communication.META_INF_IDX_MSG_TYPE])
                d['msg_data'] = int(buff[Communication.META_INF_IDX_MSG_DATA])
                d['task_label'] = buff[Communication.META_INF_IDX_TASK_SZ : ]
                ret.append(d)
        return ret

class CommProcess(mp.Process):
    SR_REQUEST = 0
    INIT = 1
    REINIT = 2


    def __init__(self, pipe, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.comm = Communication(agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port)

    def run(self):
        while True:
            op, data = self.pipe.recv()
            if op == self.SR_REQUEST:
                self.pipe.send(self.comm.send_receive_request(data))

            elif op == self.INIT:
                self.comm.init_dist()

            elif op == self.REINIT:
                self.comm.re_init_dist()

            else:
                raise Exception('Unknown command')
                
class ParallelizedComm:
    def __init__(self, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port):
        self.pipe, worker_pipe = mp.Pipe([True])
        self.worker = CommProcess(worker_pipe, agent_id, num_agents, task_label_sz, mask_sz, logger, init_address, init_port)
        self.worker.start()

    def send_receive_request(self, data):
        self.pipe.send([CommProcess.SR_REQUEST, data])
        return self.pipe.recv()

    def init_dist(self):
        self.pipe.send([CommProcess.INIT, None])

    def re_init_dist(self):
        self.pipe.send([CommProcess.REINIT, None])

def train():
    # Pseudo trainer

    # Set initial task
    msg = random.choice([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])])
    print('TASK CHANGE: ', msg)
    iteration = 0

    # Start iteration training
    while True:
        # Do communication
        other_agents_request = comm.send_receive_request(msg)
        print(other_agents_request)
        msg = None

        # Increment iteration
        iteration += 1
        time.sleep(1)

        # Check if its time to switch task
        if iteration == 512:
            msg = random.choice([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])])
            print('TASK CHANGE: ', msg)
            iteration = 0

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int)
    parser.add_argument('num_agents', help='world: total number of agents', type=int)
    parser.add_argument('server', help='should this agent create a new network or join an existing one', type=int, default=0)
    args = parser.parse_args()

    server = args.server
    num_agents = args.num_agents
    agent_id = args.agent_id

    print(agent_id, num_agents, server)

    comm = Communication(args.agent_id, args.num_agents, args.server, '127.0.0.1')



    # If agent initialises the network then create the listening server
    if server == 1:
        ServerSideSocket = socket.socket()
        host = '127.0.0.1'
        port = 5000
        ThreadCount = 0

        try:
            ServerSideSocket.bind((host, port))

        except socket.error as e:
            print(str(e))

        print('Socket is listening...')
        ServerSideSocket.listen(5)

        def multi_threaded_client(connection):
            connection.send(str.encode('Server is working:'))
            while True:
                data = connection.recv(2048)
                print(data.decode('utf-8'))
                response = 'Server message: ' + data.decode('utf-8')
                if not data:
                    break
                connection.sendall(str.encode(response))
            connection.close()

        print("Initial world size: ", num_agents)
        while True:
            Client, address = ServerSideSocket.accept()
            print('Client @ ' + address[0] + ':' + str(address[1]), ' joined the network')
            start_new_thread(multi_threaded_client, (Client, ))
            ThreadCount += 1
            print('Thread Number: ' + str(ThreadCount))
            
            # When a conenction is made, then update the worldsize parameters
            num_agents += 1
            print("New world size: ", num_agents)

        ServerSideSocket.close()

    # If the agent acts as a client to the server then run the client code
    if server == 0:
        ClientMultiSocket = socket.socket()
        host = '127.0.0.1'
        port = 5000
        print('Waiting for connection response')
        try:
            ClientMultiSocket.connect((host, port))

        except socket.error as e:
            print(str(e))

        res = ClientMultiSocket.recv(1024)
        ClientMultiSocket.send(str.encode("Hello!"))
        res = ClientMultiSocket.recv(1024)
        print(res.decode('utf-8'))
        ClientMultiSocket.close()


    
    # Listen for new agent broadcasts

    # if message received
    # re-initialise

    # else we want to broadcast to network ip that we want to connect to
    #else:
        #comm.init_dist()
        # sockets code goes here
        # The message that is received can be literally anything so long as it is verifiable
        # so we don't have to have too much bandwidth usage here.

        # More or less the algorithm we want
        # Listen for a particular message:
        #       if message is discovered
        #               re-initialise the process group
        #               and wait for the new agent to join the
        #               process group
        #               continue listening for more agents.
       # pass
        
    



