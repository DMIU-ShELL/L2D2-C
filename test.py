import time
import pickle
import requests
import ssl
from multiprocessing import Process, Queue
from socket import socket, AF_INET, SOCK_STREAM

class Communication:
    def __init__(self, port, certfile, keyfile):
        # Create a socket
        self.server_socket = socket(AF_INET, SOCK_STREAM)

        # Get the external IP of the device
        external_ip = requests.get('http://checkip.dyndns.org/').text.split(': ')[-1]

        # Bind the socket to the external IP and port
        self.server_socket.bind((external_ip, port))

        # Initialize a list to store the IP and port of each of the known peers
        self.peers = []

        # Load the certificate and key files for TLS
        self.certfile = certfile
        self.keyfile = keyfile

    def send_list(self, conn, lst):
        # Pickle the list and send it over the connection
        data = pickle.dumps(lst)
        conn.sendall(data)

    def listen_for_response(self, conn):
        while True:
            # Wait for a response
            response = conn.recv(4096)
            if not response:
                break

            # Unpickle the response
            response_list = pickle.loads(response)

            # Trigger the event
            print(response_list)

    def send_list_process(self, conn, queue, external_ip, port):
        while True:
            # Get the list from the queue
            lst = queue.get()

            # Append the external IP and port to the list
            lst = [external_ip, port, lst]

            # Send the list over the connection
            self.send_list(conn, lst)

    def start(self, queue):
        # Start listening for incoming connections
        self.server_socket.listen()

        while True:
            # Accept an incoming connection
            conn, address = self.server_socket.accept()

            # Wrap the connection in a TLS layer
            conn = ssl.wrap_socket(conn, certfile=self.certfile, keyfile=self.keyfile, server_side=True)

            # Append the IP and port of the peer to the list of peers
            if (address[0], address[1]) not in self.peers:
                self.peers.append((address[0], address[1]))

            # Get the external IP of the device
            external_ip = requests.get('http://checkip.dyndns.org/').text.split(': ')[-1]

            # Start a separate process to listen for responses
            listener_process = Process(target=self.listen_for_response, args=(conn,))
            listener_process.start()

            # Start a separate process to send the list
            sender_process = Process(target=self.send_list_process, args=(conn, queue, external_ip, address[1]))
            sender_process.start()

# Create a queue to pass the list to be sent
queue = Queue()

# Create an instance of the Communication class, passing the port and certificate/key files as arguments
communication = Communication(12345, 'cert.pem', 'key.pem')

# Start the communication
communication.start(queue)

# Put a list in the queue every three seconds
while True:
    queue.put([1, 0, 0])  # modified to send a list with only one