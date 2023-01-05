import socket
import pickle
import time
import torch
import numpy as np

while True:
    time.sleep(1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('', 29500))
        s.sendall(pickle.dumps([0, 1, 2, time.time()]))
        resp = s.recv(700000)
        resp = pickle.loads(resp)
        print(f'Time taken: {time.time() - resp[-1]}')