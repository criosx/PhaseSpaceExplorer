from flask import Flask, jsonify
from flask import abort, request
from pse.broker_worker import PSEBrokerWorker, PSEPointService
import sys

DEFAULT_PORT = 5025


class GpServerBase:
    def __init__(self):
        self.gpo: PSEPointService | None = None
        self.port = None
        self.task_dict = {
            'progress': "0%",
            'cancelled': True,
            'paused': False,
        }
        self.app = Flask(__name__)
        self.add_routes()

    def _on_service_started(self, service):
        pass

    def _on_service_stopped(self):
        pass

    def add_routes(self):
        self.app.add_url_rule("/", view_func=self.default, methods=['GET'])
        self.app.add_url_rule("/get_status", view_func=self.get_status, methods=['GET'])
        self.app.add_url_rule("/get_info", view_func=self.get_info, methods=['GET'])
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
        if self.gpo is not None and not self.task_dict.get('cancelled', True):
            abort(400, description='Another PSE optimization is already running.')
        return data

    def default(self):
        return "Server is running on port {}".format(self.port)

    def get_status(self):
        if self.gpo is not None:
            return self.gpo.task_dict.get('status', 'running')
        return "idle"

    def get_info(self):
        from pse.broker_worker import ACQUISITION_FUNCTIONS
        return jsonify({
            "has_service": self.gpo is not None,
            "status": self.gpo.task_dict.get('status', 'running') if self.gpo else "idle",
            "storage_path": str(self.gpo.spath) if self.gpo else None,
            "acquisition_functions": list(ACQUISITION_FUNCTIONS.keys()),
        })

    def pause_pse(self):
        if self.gpo is not None:
            self.gpo._paused = True
            self.task_dict['paused'] = True
        return "PSE paused"

    def resume_pse(self):
        self.check_post()
        if self.task_dict.get('paused') and self.gpo is not None:
            self.gpo._paused = False
            self.task_dict['paused'] = False
            return "PSE resumed"
        return "PSE was not paused."

    def run(self, port=None):
        if port is None:
            port = DEFAULT_PORT
        self.port = port
        print(f"Starting Phase Space Explorer Flask server on port {self.port}")
        self.app.run(port=self.port)

    def start_pse(self):
        data = self.check_post()
        self.task_dict['cancelled'] = False
        return self.pse_go(data, from_pause=False)

    def pse_go(self, data, from_pause=False):
        data.pop('client', None)

        if from_pause:
            if self.gpo is not None:
                self.gpo._paused = False
                self.task_dict['paused'] = False
            return "PSE resumed"

        return self._start_service(data)

    def _start_service(self, data):
        try:
            service = PSEPointService(**data)
            service.initialize()
        except ValueError as e:
            self.task_dict['cancelled'] = True
            self.task_dict['progress'] = '100%'
            return str(e)

        self.gpo = service
        self.task_dict['progress'] = '0%'
        self.task_dict['cancelled'] = False
        self.task_dict['paused'] = False
        self._on_service_started(service)
        return "PSE started"

    def stop_pse(self):
        if self.task_dict.get('cancelled', True):
            return "PSE was already stopped."
        self.task_dict['cancelled'] = True
        self._on_service_stopped()
        if self.gpo is not None:
            self.gpo.gp_hardware_shutdown()
            self.gpo = None
        return "PSE stopped"


class BrokerGpServer(GpServerBase):
    def __init__(self):
        self._broker = PSEBrokerWorker()
        self._broker._on_service_changed = self._sync_gpo
        self._broker.start()
        super().__init__()

    def _sync_gpo(self, service):
        """Called by PSEBrokerWorker when a service is configured or cleared via broker."""
        self.gpo = service
        if service is not None:
            self.task_dict.update({'cancelled': False, 'progress': '0%', 'paused': False})
        else:
            self.task_dict['cancelled'] = True

    def _on_service_started(self, service):
        self._broker.set_service(service)

    def _on_service_stopped(self):
        self._broker.set_service(None)


# Backward-compatible alias
GpServer = BrokerGpServer


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    BrokerGpServer().run(port)
    _ = input("Press enter to exit...")
