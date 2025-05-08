import gp
from flask import Flask
from flask import abort, request
import json
import os
from multiprocessing import Process, Manager
from threading import Thread
from typing import Optional
import time
import uuid
from werkzeug.serving import run_simple

import socket
from contextlib import closing

app = Flask(__name__)

manager = Manager()
# shared dict between server and subprocesses using the Manager funcitonality
task_dict = manager.dict({"status": "idle", "progress": "0%", "cancelled": False})
gpo = None
port = None


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


@app.route("/")
def default():
    return "Server is running on port {}".format(port)


def start_server():
    global port
    port = find_free_port()
    # save port number to file for streamlit to read
    with open("my_service_port.txt", "w") as f:
        f.write(str(port))
    print(f"Starting Phase Space Explorer Flask server on port {port}")

    global app
    run_simple("localhost", port, app)


@app.route('/start_pse', methods=['POST'])
def start_pse():
    """
    POST request function that starts a PSE task.

    The POST data must dictioinary must contain the keyword arguments passed to gp init

    :return: status message.
    """

    if request.method != 'POST':
        abort(400, description='Request method is not POST.')

    data = request.get_json()
    if data is None or not isinstance(data, dict):
        abort(400, description='No valid data received.')

    global gpo
    gpo = gp.Gp(**data)
    global task_dict
    task_dict = {"status": "running", "progress": "0%", "cancelled": False}
    p = Process(target=gpo.run, args=task_dict)
    p.start()

    return "PSE started"


if __name__ == "__main__":
    start_server()
