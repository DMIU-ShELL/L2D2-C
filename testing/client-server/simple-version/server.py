import socket
import pickle

HOST = '127.0.0.1'
PORT = 25900

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                #conn.sendall(data)
                data = s.recv(1024)
                data = pickle.loads(data)

                print(f"Received {data!r}")