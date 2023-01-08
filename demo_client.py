import socket


def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 5000)
    print(f"connecting to {server_address}")
    sock.connect(server_address)

    try:
        data = sock.recv(1024)
        print(data.decode())

        model = input("Choose a model: ")

        sock.sendall(model.encode())

        while True:
            data = sock.recv(1024)
            print(data.decode())
            if "closing connection..." in data.decode():
                break

    except Exception as e:
        print("Error: ", e)
        sock.close()
        return

    finally:
        print("Send 'close' to close the connection")
        sock.sendall("close".encode())
        print("closing socket")
        sock.close()


if __name__ == "__main__":
    main()
