import multiprocessing as mp
import multiprocessing.dummy as mpd
import pickle
import socket
from torch import tensor
import argparse
from time import sleep
import struct
import time
import ssl

from errno import ECONNRESET
from colorama import Fore

HOST = 'lnx-grid-19.lboro.ac.uk'
OTHER_PORTS = [29500+i for i in range(0, 5)]

# TCP + TLS v2
class TCP_TLSv2:
    def unpack(self, data):
        emb_sz = 3
        try:
            address = f'{int(data[0])}.{int(data[1])}.{int(data[2])}.{int(data[3])}'
            port = int(data[4])
            msg_type = int(data[5])
            msg_data = int(data[6])
            embedding = tensor(data[7:7+emb_sz])

            return address, port, msg_type, msg_data, embedding
        except:
            return None, None, None, None, None

    def server(self, port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server = ssl.wrap_socket(server, server_side=True, keyfile="key.pem", certfile="certificate.pem")

        server.bind((HOST, port))
        server.listen(0)

        while True:
            conn, addr = server.accept()
            with conn:
                print(Fore.CYAN + f"\nConnected by {addr}")
                while True:
                    try:
                        data = conn.recv(4096)
                        if not data: break

                        #data = pickle.loads(data)
                        data = list(struct.unpack('11d', data))
                        _address, _port, _msg_type, _msg_data, _embedding = self.unpack(data)
                        print(Fore.CYAN + f"Received{data!r}. Time taken: {(time.time()-data[-1])*(10**3):.3f}µs")
                        print(Fore.CYAN + f"Address: {_address} \nPort: {_port} \nType: {_msg_type} \nData: {_msg_data} \nEmbedding: {_embedding}")
                    except socket.error as e:
                        if e.errno != ECONNRESET: raise
                        print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                        pass

    def client(self, port, buffer):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client = ssl.wrap_socket(client, keyfile="key.pem", certfile="certificate.pem")

        #buffer = pickle.dumps(buffer)
        buffer = struct.pack('11d', *buffer)
        try:
            client.connect((HOST, port))
            client.sendall(buffer)
            client.close()

        except:
            pass


# HTTPS + TLS
import http.server
import requests
class HTTPS_TLS:
    def server(self):
        httpd = http.server.HTTPServer(('localhost', 443), http.server.SimpleHTTPRequestHandler)
        httpd.socket = ssl.wrap_socket (httpd.socket, certfile='/certificate.pem', server_side=True, ssl_version=ssl.PROTOCOL_TLS)
        httpd.serve_forever()

    def client(self):
        res = requests.get('https://localhost:443')
        print(res)


# TCP + TLS
class TCP_TLS:
    def server(self, port):
        while True:
            print('Listening...')
            with ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), server_side=True, keyfile='key.pem', certfile='certificate.pem') as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, port))
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    print(f"\nConnected by {addr}")
                    while True:
                        data = conn.recv(4096)
                        if not data: break
                        data = pickle.loads(data)
                        print(f"Received{data!r}. Time taken to complete transfer: {(time.time()-data[-1])*(10**3):.3f}µs")

    def client(self, port, buffer):
        try:
            with ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), keyfile='key.pem', certfile='certificate.pem') as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.connect(HOST, port)
                s.sendall(buffer)

                return True
        except:
            return False


# TCP
class TCP:
    def handle(self, conn, addr):
        with conn:
            print(f"\nConnected by {addr}")
            while True:
                data = conn.recv(4096)
                if not data: break
                data = pickle.loads(data)
                #data = list(struct.unpack('11d', data))
                print(f"Received {data!r}. Time taken to complete transfer: {(time.time()-data[-1])*(10**3):.3f}µs")

    def server(self, port):
        while True:
            print('Listening...')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, port))
                s.listen(1)
                conn, addr = s.accept()
                self.handle(conn, addr)

    def client(self, buffer):
        buffer = pickle.dumps(buffer)
        #buffer = struct.pack('11d', *buffer)
        print(len(buffer))
        for port in OTHER_PORTS:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((HOST, port))
                    s.sendall(buffer)
            except:
                pass



# UDP
class UDP:
    def server(self, port):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind((HOST, port))
                while True:
                    data, addr = s.recvfrom(4096)
                    data = pickle.loads(data)
                    #data = list(struct.unpack('10i', data))
                    print(f'Received {data!r}, {type(data)}')

    def client(self, buffer):
        buffer = pickle.dumps(buffer)
        #buffer = struct.pack('10i', *buffer)

        for port in OTHER_PORTS:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.sendto(buffer, (HOST, port))

            except:
                pass


if __name__ == '__main__':
    print('Running TCP + TLS/SSL in Peer to Peer')
    parser = argparse.ArgumentParser()
    parser.add_argument('port', help='port for this agent', type=int)
    #parser.add_argument('link', help='address of an agent in the network', type=int)
    args = parser.parse_args()
    OTHER_PORTS.remove(args.port)
    print(OTHER_PORTS)

    TL = TCP_TLSv2()

    p_server = mp.Process(target=TL.server, args=(args.port,))
    p_server.start()

    #client(b'')

    #if args.port == 29500:
    count = 0
    while True:
        count += 1
        print(Fore.GREEN + f'\n\nIteration: {count}')
        for port in OTHER_PORTS:
            print(Fore.GREEN + f'Attempting to send to port: {port}')
            TL.client(port, [127, 0, 0, 1, args.port, 2, 1, 1, 0, 0, time.time()])

        sleep(3)
        