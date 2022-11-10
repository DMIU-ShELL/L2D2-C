import multiprocessing as mp
import pickle
import socket
import torch
import argparse

HOST = '127.0.0.1'

OTHER_PORTS = []

def server(port):
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, port))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                
                while True:
                    data = conn.recv(512)
                    if not data:
                        break
                    data = pickle.loads(data)
                    print(f"\nReceived {data!r}", flush=True)
                    #conn.sendall(data)

                # Handle terminated connection to another agent
                # if connection terminated:
                #   conn.close()

def client():
    for port in OTHER_PORTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, port))
                buffer = torch.tensor([0., 2., 1., 0., 0., 1.])
                buffer = pickle.dumps(buffer)
                s.sendall(buffer)
        except:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('port', help='port for this agent', type=int)
    args = parser.parse_args()
    OTHER_PORTS.remove(args.port)

    p_server = mp.Process(target=server, args=(args.port,))
    p_server.start()

    while True:
        trigger = input('\nSend data?: ')
        if trigger:
            client()

