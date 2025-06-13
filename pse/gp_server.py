from contextlib import closing
from flask import Flask
from flask import abort, request
from pse.gp import Gp as GpParent
from threading import Thread
import os
import socket
import sys


class GpServer:
    def __init__(self):
        self.gpo = None
        self.port = None
        self.p = None
        self.task_dict = {
            'progress': "0%",
            'cancelled': True,
            'paused': False,
        }
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

    def check_post(self):
        if request.method != 'POST':
            abort(400, description='Request method is not POST.')
        data = request.get_json()
        if data is None or not isinstance(data, dict):
            abort(400, description='No valid data received.')
        if self.p is not None:
            abort(400, description='Another PSE optimization is already running.')
        return data

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
            self.task_dict['paused'] = True
        if self.p is not None:
            self.p.join()
            # keep the gpo alive
            # self.gpo = None
            self.p = None
        return "PSE paused"

    def resume_pse(self):
        data = self.check_post()
        if self.task_dict['paused']:
            self.task_dict['paused'] = False
            if not self.task_dict['cancelled']:
                # call start PSE function as functionally a restart from Pause is only different in that the hardware
                # is not being reiniatialized, which will be decided based on inside gp.py the 'paused' flag in the
                # task_dict
                self.pse_go(data, from_pause=True)
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
        data = self.check_post()
        if not self.task_dict['cancelled']:
            return "PSE was already running."
        self.task_dict['cancelled'] = False
        self.pse_go(data, from_pause=False)
        return "PSE started"

    def pse_go(self, data, from_pause=False):
        if 'client' in data:
            if data['client'] == 'ROADMAP':
                from pse.roadmap import ROADMAP_Gp as GpObject
            elif data['client'] == 'Test Ackley Function':
                from pse.gp import Gp as GpObject
            del data['client']
        else:
            from pse.roadmap import ROADMAP_Gp as GpObject

        if from_pause:
            # just reinitialize the object with updated arguments, keep inits from children untouched
            GpParent.__init__(self.gpo, **data)
        else:
            self.gpo = GpObject(**data)

        self.task_dict["progress"] = "0%"
        self.p = Thread(target=self.gpo.run, args=(self.task_dict, from_pause))
        self.p.start()

        return "PSE started"

    def stop_pse(self):
        if self.task_dict['cancelled']:
            return "PSE was already stopped."
        self.task_dict["cancelled"] = True
        if self.p is not None:
            self.p.join()
            self.gpo = None
            self.p = None
        else:
            # PSE not cancelled, but process not alive -> gp is in pause
            # shut down hardware
            if self.gpo is not None:
                self.gpo.gp_hardware_shutdown()
                self.gpo = None
        return "PSE stopped"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gp_server.py <storage_directory>")
        sys.exit(1)

    port = sys.argv[1]
    gp_server = GpServer()
    gp_server.run(port)
    _ = input("Press enter to exit...")
