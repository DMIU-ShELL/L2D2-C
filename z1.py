import socket
import pickle
import time
import torch
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 29500))
sock.listen()
while True:
    conn, addr = sock.accept()
    with conn:
        print(f'Connected by {addr}')
        while True:
            data = conn.recv(700000)
            if not data: break
            data = pickle.loads(data)
            print(data)
            conn.sendall(pickle.dumps(data))