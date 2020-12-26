import socket, sys
import comm
import time

# data can be whatever you want (even just sys.argv)
data = sys.argv

print( "sending to server:")
print( data)

# send data to the server and receive a result
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# disable Nagle algorithm (probably only needed over a network) 
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
sock.settimeout(20)
watchguard=True
while watchguard:
    try:
        sock.connect((comm.server_host, comm.server_port))
        comm.send_data(data, sock)
        result = comm.receive_data(sock)
        sock.close()
        watchguard=False
    except:
        print("Error, sleeping...")
        time.sleep(5)

# do something with the result...
print( "result from server:")
print( result)
