from contextlib import closing
from flask import Flask
from flask import abort, request
# from multiprocessing import Process, Manager
from threading import Thread
import os
import socket
import sys

app = Flask(__name__)
gpo = None
port = None
p = None
task_dict = {}


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


@app.route("/", methods=['GET'])
def default():
    global port
    print("Serving on port {}".format(port))
    return "Server is running on port {}".format(port)


@app.route("/get_status", methods=['GET'])
def get_status():
    global task_dict
    if task_dict:
        return task_dict["status"]
    else:
        return "down"


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
    global p

    if request.method != 'POST':
        abort(400, description='Request method is not POST.')

    data = request.get_json()
    if data is None or not isinstance(data, dict):
        abort(400, description='No valid data received.')

    if p is not None:
        abort(400, description='Another gp instance is already running.')

    if 'client' in data:
        if data['client'] == 'ROADMAP':
            from pse.roadmap import ROADMAP_Gp as gpobject
        elif data['client'] == 'Test Ackley Function':
            from pse.gp import Gp as gpobject
        del data['client']
    else:
        from pse.roadmap import ROADMAP_Gp as gpobject

    gpo = gpobject(**data)
    # manager = Manager()
    # task_dict = manager.dict()
    task_dict["status"] = "running"
    task_dict["progress"] = "0%"
    task_dict["cancelled"] = False
    # gp client needs to be a process, otherwise the server will stop responding during computation intensive
    # tasks in the client.
    p = Thread(target=gpo.run, args=(task_dict, ))
    p.start()

    return "PSE started"


@app.route('/stop_pse', methods=['GET'])
def stop_pse():
    global task_dict
    global p
    global gpo

    if task_dict:
        task_dict["cancelled"] = True
    if p is not None:
        p.join()
        gpo = None

    return "PSE stopped"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gp_server.py <storage_directory>")
        sys.exit(1)

    storage_dir = sys.argv[1]
    start_server(storage_dir)
