import socket
import threading
import multiprocessing
import time

def handle_client(client_socket):
    # Receive data from the client
    request = client_socket.recv(1024).decode()
    print(f"Received request: {request}")

    # Send a response back to the client
    response = "Hello from the server!"
    client_socket.sendall(response.encode())

    # Close the client socket
    client_socket.close()

def start_server():
    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_address = ("127.0.0.1", 8080)
    server_socket.bind(server_address)

    # Listen for incoming connections (max queue size is set to 5)
    server_socket.listen(5)
    print(f"Server listening on {server_address}")

    try:
        while True:
            # Accept a new connection
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")

            # Create a new thread to handle the client
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.start()

    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        # Close the server socket
        server_socket.close()

def send_query(ip, port, query):
    # Establish a TCP connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip, port))

        # Send the query
        s.sendall(query.encode())

        # Receive the response
        response = s.recv(1024).decode()

    print(f"Received response from server: {response}")

def run_server_and_client():
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=2) as pool:
        # Start the server in one process
        server_process = pool.apply_async(start_server)

        # Allow some time for the server to start before launching the client
        time.sleep(2)

        # Start the querying client in another process
        client_process = pool.apply_async(send_query, args=("127.0.0.1", 8080, "Sample Query"))

        # Wait for both processes to complete
        server_process.get()
        client_process.get()

if __name__ == "__main__":
    run_server_and_client()
