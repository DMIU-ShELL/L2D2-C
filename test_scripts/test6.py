import socket
import sys
from time import sleep
import random
from struct import pack, unpack
import multiprocess as mp


def receive():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    host, port = '127.0.0.1', 65000
    server_address = (host, port)

    print(f'Starting UDP server on {host} port {port}')
    sock.bind(server_address)

    while True:
        # Wait for message
        message, address = sock.recvfrom(4096)

        print(f'Received {len(message)} bytes:')
        x, y, z = unpack('3f', message)
        print(f'X: {x}, Y: {y}, Z: {z}')

def send():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    host, port = '127.0.0.1', 65000
    server_address = (host, port)

    # Generate some random start values
    x, y, z = random.random(), random.random(), random.random()

    # Send a few messages
    for i in range(10):

        # Pack three 32-bit floats into message and send
        message = pack('3f', x, y, z)
        sock.sendto(message, server_address)

        sleep(1)
        x += 1
        y += 1
        z += 1


if __name__ == '__main__':
    processes = []
    workers = 4

    for i in range(workers):
        p = mp.Process(target=receive)
        p.start()
        processes.append(p)


    p = mp.Process(target=send)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()