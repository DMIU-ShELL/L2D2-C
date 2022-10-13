import socket, sys               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.
s.bind((host, port))        # Bind to the port

connect = True
if connect == True:
    host = sys.argv[2]
    s.connect((host, port))
    s.close 
    
else:
    s.listen(5)                 # Now wait for client connection.
    while True:
       c, addr = s.accept()     # Establish connection with client.
       print('Got connection from', addr)
       c.send('Thank you for connecting')
       c.close()