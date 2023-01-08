import socket
import threading
from prediction_methods.prediction import cry_detection
from recorder.record import record


class Server:
    def __init__(self, port=5000, host=""):
        self.port = port
        self.host = host
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.close_connection = False

    def __del__(self):
        self.server_socket.close()

    def run(self):
        # Bind the socket to the port
        self.server_socket.bind((self.host, self.port))

        # Listen for incoming connections
        print("Listening for incoming connections...")
        self.server_socket.listen(1)

        while True:
            # Wait for a connection
            connection, client_address = self.server_socket.accept()
            print("Connection from: ", client_address)

            # Create a new thread to handle the connection
            client_thread = threading.Thread(target=Server.handle_client, args=(self, connection, client_address))
            client_thread.start()

    def handle_client(self, connection, client_address):
        try:
            msg = "Connected!\nI will predict if the audio file is a baby cry or not.\nChoose a model:\n1. CNN\n2. SVC\n"
            connection.sendall(msg.encode())

            data = connection.recv(1024)
            model = data.decode()

            if model == "1" or model.upper() == "CNN":
                model = "cnn"
                connection.sendall("You chose CNN\n".encode())
            elif model == "2" or model.upper() == "SVC":
                model = "svc"
                connection.sendall("You chose SVC\n".encode())
            else:
                connection.sendall("The model type is not supported.\nclosing connection...\n".encode())
                return

            check_close_socket_thread = threading.Thread(target=Server.end_conversation, args=(self, connection))
            check_close_socket_thread.start()

            # Receive the data in small chunks and retransmit it
            while not self.close_connection:
                filename = record()
                result = cry_detection(filename, model, num_of_frames=5)
                if result == 1:
                    print("Baby cry detected! --> Send message to the parent")
                    connection.sendall("Baby cry detected!\n".encode())

        except Exception as e:
            print("No connection from parent's application. Error: ", e)
            connection.close()
            return

        finally:
            print("Closing connection with: ", client_address)
            # Clean up the connection
            connection.close()
            print("Connection closed.\n Waiting for new connection...\n")

    def end_conversation(self, connection):
        try:
            data = connection.recv(1024)
            if data.decode() == "close":
                connection.sendall("closing connection...\n".encode())
                self.close_connection = True
        except Exception as e:
            print("Error: ", e)
            self.close_connection = True
