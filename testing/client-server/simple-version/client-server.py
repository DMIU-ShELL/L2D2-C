import multiprocessing as mp
import multiprocessing.dummy as mpd
import pickle
import socket
from torch import tensor, rand
import argparse
from time import sleep
import struct
import time
import ssl
import socketserver
import sys

from errno import ECONNRESET
from colorama import Fore

HOST = ''

OTHER_DST = {29500: 'lnx-grid-19.lboro.ac.uk', 29501: 'lnx-grid-19.lboro.ac.uk'}



# SOCKET SERVER IMPLEMENTATION
class SS_HANDLER(socketserver.BaseRequestHandler):
    def handle(self):
        self.data = self.request.recv(400000).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        self.request.sendall(self.data.upper())

class SS_SERVER:
    def server(self, port):
        print(f'Attempting to make server on {HOST}, {port}')
        with socketserver.TCPServer((HOST, port), SS_HANDLER) as s:
            s.serve_forever()

    def client(self, address, port, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
            try:
                c.connect((address, port))
                c.sendall(bytes(data + '\n', 'utf-8'))

                received = str(c.recv(400000), 'utf-8')

                print("Sent:     {}".format(data))
                print("Received: {}".format(received))
                
            except:
                print('failed')




# TCP + TLS v3
class TCP_TLSv3:
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

    def send_msg(self, sock, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        sock.sendall(msg)

    def recv_msg(self, sock):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen: return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        print(msglen)
        # Read the message data
        return self.recvall(sock, msglen)

    def recvall(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return data

    def server(self, port):
        print(f'Attempting to start server on port {port}')
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(certfile='certificates/certificate.pem', keyfile='certificates/key.pem')
        context.load_verify_locations('certificates/certificate.pem')
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('', port))
        server_socket.listen(0)

        #server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #server = ssl.wrap_socket(server, server_side=True, keyfile="certificates/key.pem", certfile="certificates/certificate.pem", ssl_version=ssl.PROTOCOL_TLS_SERVER)

        #server.bind((HOST, port))
        #server.listen(0)

        while True:
            conn, addr = server_socket.accept()
            with context.wrap_socket(conn, server_side=True) as conn:
                print(Fore.CYAN + f"\nConnected by {addr}")
                #print(f'Peer cert: {conn.getpeercert()}')
                while True:
                    #try:
                        #data = conn.recv(400000)
                        data = self.recv_msg(conn)
                        if not data: break

                        data = pickle.loads(data)
                        #data = list(struct.unpack('11d', data))
                        #_address, _port, _msg_type, _msg_data, _embedding = self.unpack(data)
                        print(Fore.CYAN + f"Received{data!r}. Time taken: {(time.time()-data[-1])*(10**3):.3f}µs")
                        print(Fore.CYAN + f"Address: {data[0]} \nPort: {data[1]} \nType: {data[2]} \nData: {data[3]} \nMask: {data[4]} \nEmbedding: {data[5]}")
                    
                        #print(Fore.CYAN + f"Address: {_address} \nPort: {_port} \nType: {_msg_type} \nData: {_msg_data} \nEmbedding: {_embedding}")
                    #except socket.error as e:
                        #if e.errno != ECONNRESET: raise
                        #print(Fore.RED + f'Error raised while attempting to receive data from {addr}')
                        #pass

    def client(self, address, port, buffer):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_cert_chain(certfile='certificates/certificate.pem', keyfile='certificates/key.pem')
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.load_verify_locations('certificates/certificate.pem')
        client_socket = context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), server_hostname=address)


        #client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #client = ssl.wrap_socket(client, keyfile="certificates/key.pem", certfile="certificates/certificate.pem", ssl_version=ssl.PROTOCOL_TLS_CLIENT)

        buffer = pickle.dumps(buffer)
        #print(buffer)
        #buffer = struct.pack('11d', *buffer)
        try:
            client_socket.connect((address, port))
            #client_socket.sendall(buffer)
            self.send_msg(client_socket, buffer)
            #print(f'Peer cert: {client_socket.getpeercert()}')
            client_socket.shutdown(socket.SHUT_RDWR)

        except:
            print('Failed :(')

        finally:
            client_socket.close()


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

    print(args.port)

    del OTHER_DST[args.port]

    TL = SS_SERVER()

    #if args.port == 29500:
    p_server = mp.Process(target=TL.server, args=(args.port,))
    p_server.start()

    #client(b'')

    #if args.port == 29500:
    #else:
    count = 0
    while True:
        count += 1
        print(Fore.GREEN + f'\n\nIteration: {count}')
        for port, address in OTHER_DST.items():
            print(Fore.GREEN + f'Attempting to send to port: {address}:{port}')
            #TL.client(address, port, ['hello'])
            #TL.client(port, [127, 0, 0, 1, args.port, 2, 1, 1, 0, 0, time.time()])
            TL.client(address, port, ['lnx-grid-19.lboro.ac.uk', args.port, 4, 4, rand(110800), rand(3), time.time()])
            #TL.client(port, ['127.0.0.1', args.port, 4, 4, rand(3), time.time()])
            #TL.client(address, port, 'hello world')

        sleep(3)
        