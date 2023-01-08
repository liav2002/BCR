from Server.server import Server


def main():
    print("Starting the server...")
    print("Press Ctrl+C to stop the server.")
    server = Server()
    server.run()


if __name__ == "__main__":
    main()
