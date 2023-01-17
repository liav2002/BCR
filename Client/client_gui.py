import threading
import tkinter as tk
import socket
import os
from time import sleep

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class App:
    def __init__(self):
        # Flag for connection
        self.keep_alive = False

        # Create a window
        self.root = tk.Tk()
        self.root.configure(background="lightblue", width=800, height=600)
        self.root.minsize(800, 600)
        self.root.maxsize(800, 600)
        self.root.title("Client")

        # add BCR_Title Image to the GUI
        self.bcr_title = tk.PhotoImage(file=os.path.join(PROJECT_PATH, "Client", "Images", "BCR_title.png"))
        self.bcr_title_label = tk.Label(self.root, image=self.bcr_title, bg="lightblue")
        self.bcr_title_label.place(x=200, y=0)

        # Add a label for "Choose a model"
        self.label = tk.Label(self.root, text="Choose a model", bg="lightblue", fg="black", font=("Arial", 14))
        self.label.place(x=310, y=170)

        # Create select button for select a model
        self.model = tk.StringVar(self.root)
        self.model.set("SVC")  # default value
        self.select_model = tk.OptionMenu(self.root, self.model, "SVC", "CNN")
        self.select_model.config(width=20, height=1)
        self.select_model.place(x=300, y=200)

        # Create a button for start the model
        self.start_button = tk.Button(self.root, text="Start", width=20, height=1, bg="silver", fg="black",
                                      font=("Arial", 14),
                                      command=lambda: self.connect_server(self.model.get()))
        self.start_button.place(x=270, y=250)

        # Create label text for errors
        self.error_label = tk.Label(self.root, text="", bg="lightblue", fg="red", font=("Arial", 14))
        self.error_label.place(x=250, y=300)

        # Create an image of the smile emoji
        self.smile_emoji = tk.PhotoImage(file=os.path.join(PROJECT_PATH, "Client", "Images", "smile.png"))
        self.smile_emoji_label = tk.Label(self.root, image=self.smile_emoji, bg="lightblue")

        # Create an image of the cry emoji
        self.cry_emoji = tk.PhotoImage(file=os.path.join(PROJECT_PATH, "Client", "Images", "cry.png"))
        self.cry_emoji_label = tk.Label(self.root, image=self.cry_emoji, bg="lightblue")

    def startApp(self):
        self.root.mainloop()

    def connect_server(self, model):
        # Clear the error label
        self.error_label.config(text="")

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Connect the socket to the port where the server is listening
            server_address = ('localhost', 5000)
            print(f"connecting to {server_address}")
            sock.connect(server_address)
            self.keep_alive = True

        except Exception as e:
            print("Error {connect_server}: ", e)
            sock.close()
            self.error_label.config(text="Failed to connect to the server.")
            return

        # Hide start button and select model button and 'Choose a model' label
        self.label.place_forget()
        self.select_model.place_forget()
        self.start_button.place_forget()

        # Show the smile emoji
        self.smile_emoji_label.place(x=250, y=200)

        # Create a button for close connection
        close_connection_button = tk.Button(self.root, text="Close", width=20, height=1, bg="silver", fg="black",
                                            font=("Arial", 14), command=lambda: self.closeConnection(sock, close_connection_button))
        close_connection_button.place(x=270, y=550)

        # Send the model name to the server
        data = sock.recv(1024)
        print("Server sent: " + data.decode())
        print("Chosen model is: " + model)
        sock.sendall(model.encode())

        # Last message from the server before starting the model
        data = sock.recv(1024)
        print("Server sent: " + data.decode())

        # Receive the result from the server --> model is started
        threading.Thread(target=self.get_result_from_server, args=(sock,)).start()

    def get_result_from_server(self, sock):
        try:
            while self.keep_alive:
                data = sock.recv(1024) if self.keep_alive else None
                print("Server sent: " + data.decode())

                # Check if the connection is alive.
                sock.sendall("".encode()) if self.keep_alive else None

                if "Baby cry detected!" in data.decode():
                    # Show the cry emoji
                    self.cry_emoji_label.place(x=250, y=200)
                    sleep(5)
                    self.cry_emoji_label.place_forget()

        except Exception as e:
            print("Error {get_result_from_Server}: ", e)

            # Reveal start button and select model button and 'Choose a model' label
            self.label.place(x=310, y=170)
            self.select_model.place(x=300, y=200)
            self.start_button.place(x=270, y=250)

            # Hide the smile emoji
            self.smile_emoji_label.place_forget()

    def closeConnection(self, sock, close_connection_button):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        close_connection_button.place_forget()
        sock.sendall("close".encode())
        self.keep_alive = False
        sock.close()

        # Reveal start button and select model button and 'Choose a model' label
        self.label.place(x=310, y=170)
        self.select_model.place(x=300, y=200)
        self.start_button.place(x=270, y=250)

        # Hide the smile emoji
        self.smile_emoji_label.place_forget()


def main():
    print("Starting the client...")
    print("Press Ctrl+C to stop the client.\n")
    app = App()
    app.startApp()


if __name__ == "__main__":
    main()
