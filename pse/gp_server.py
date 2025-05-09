from contextlib import closing
from flask import Flask
from flask import abort, request
from multiprocessing import Process, Manager, Queue
import os
import requests
import socket
import sys
import threading
import time
from werkzeug.serving import run_simple

from pse import gp

app = Flask(__name__)
gpo = None
port = None
task_dict = None


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


@app.route("/", methods=['GET'])
def default():
    global port
    print("Serving on port {}".format(port))
    return "Server is running on port {}".format(port)


def start_server(storage_dir):
    global app
    global port
    global task_dict

    print(f"Using storage directory: {storage_dir} for gp.")

    port = find_free_port()
    # save port number to file for streamlit to read
    fp = os.path.join(storage_dir, "service_port.txt")
    with open(fp, "w") as f:
        f.write(str(port))
    print(f"Starting Phase Space Explorer Flask server on port {port}")

    manager = Manager()
    # shared dict between server and subprocesses using the Manager funcitonality
    task_dict = manager.dict({"status": "idle", "progress": "0%", "cancelled": False})

    app.run(port=port)


@app.route('/start_pse', methods=['POST'])
def start_pse():
    """
    POST request function that starts a PSE task.

    The POST data must dictioinary must contain the keyword arguments passed to gp init

    :return: status message.
    """
    global gpo
    global task_dict

    if request.method != 'POST':
        abort(400, description='Request method is not POST.')

    data = request.get_json()
    if data is None or not isinstance(data, dict):
        abort(400, description='No valid data received.')

    gpo = gp.Gp(**data)
    task_dict = {"status": "running", "progress": "0%", "cancelled": False}
    p = Process(target=gpo.run, args=(task_dict, ))
    p.start()

    return "PSE started"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gp_server.py <storage_directory>")
        sys.exit(1)

    storage_dir = sys.argv[1]
    start_server(storage_dir)
