import sys
import socket 
from monarch.actor import run_worker_loop_forever 


def main():
    hostname = socket.gethostname(); 
    run_worker_loop_forever(address=f"tcp://{hostname}:22222", ca="trust_all_connections")

if __name__ == "__main__":
    main()