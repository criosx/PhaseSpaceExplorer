from contextlib import closing
from flask import Flask
from flask import abort, request
from multiprocessing import Process, Manager
from threading import Thread
import os
import socket
import sys


class GpServer:
    def __init__(self):
        self.gpo = None
        self.port = None
        self.p = None
        self.task_dict = {}
        self.app = Flask(__name__)
        # add routes
        self.add_routes()

    def add_routes(self):
        self.app.add_url_rule("/", view_func=self.default, methods=['GET'])
        self.app.add_url_rule("/get_status", view_func=self.get_status, methods=['GET'])
        self.app.add_url_rule('/resume_pse', view_func=self.resume_pse, methods=['POST'])
        self.app.add_url_rule('/pause_pse', view_func=self.pause_pse, methods=['GET'])
        self.app.add_url_rule('/start_pse', view_func=self.start_pse, methods=['POST'])
        self.app.add_url_rule('/stop_pse', view_func=self.stop_pse, methods=['GET'])

    def default(self):
        print("Serving on port {}".format(self.port))
        return "Server is running on port {}".format(self.port)

    def get_status(self):
        if "status" in self.task_dict:
            return self.task_dict["status"]
        else:
            return "down"

    def pause_pse(self):
        if self.task_dict:
            self.task_dict["cancelled"] = True
            self.task_dict['paused'] = True
        if self.p is not None:
            self.p.join()
            self.gpo = None
            self.p = None
        return "PSE paused"

    def resume_pse(self):
        if self.task_dict and self.task_dict['paused']:
            self.task_dict["cancelled"] = False
            # call start PSE function as functionally a restart from Pause is only different in that the hardware
            # is not being reiniatialized, which will be decided based on inside gp.py the 'paused' flag in the
            # task_dict
            self.start_pse()
            return "PSE resumed"
        else:
            return "PSE was not paused."

    def run(self, port=None):
        if port is None:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                # Bind to a free port provided by the host.
                s.bind(('', 0))
                port = s.getsockname()[1]
        self.port = port
        print(f"Starting Phase Space Explorer Flask server on port {self.port}")
        self.app.run(port=self.port)

    def start_pse(self):
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

        if self.p is not None:
            abort(400, description='Another gp instance is already running.')

        if 'client' in data:
            if data['client'] == 'ROADMAP':
                from pse.roadmap import ROADMAP_Gp as gpobject
            elif data['client'] == 'Test Ackley Function':
                from pse.gp import Gp as gpobject
            del data['client']
        else:
            from pse.roadmap import ROADMAP_Gp as gpobject

        self.gpo = gpobject(**data)
        self.task_dict["progress"] = "0%"
        self.task_dict["cancelled"] = False
        self.p = Thread(target=self.gpo.run, args=(self.task_dict, ))
        self.p.start()

        return "PSE started"

    def stop_pse(self):
        if self.task_dict:
            self.task_dict["cancelled"] = True
        if self.p is not None:
            self.p.join()
            self.gpo = None
            self.p = None
        return "PSE stopped"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gp_server.py <storage_directory>")
        sys.exit(1)

    port = sys.argv[1]
    gp_server = GpServer()
    gp_server.run(port)
    _ = input("Press enter to exit...")
