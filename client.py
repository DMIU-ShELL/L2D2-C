import socket
import torch
import torch.distributed as dist
import struct
import pickle

HOST = "127.0.01"
PORT = 25900

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = torch.tensor([0., 2., 1., 0., 0., 1.])
    #buffer = [0., 0., 1.]
    buffer = pickle.dumps(buffer)
    s.sendall(buffer)


    