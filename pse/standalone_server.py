import logging
import sys
import threading

from pse.gp import Gp
from pse.gp_server import GpServerBase

logger = logging.getLogger(__name__)


class StandaloneGpServer(GpServerBase):
    def __init__(self):
        self._opt_thread = None
        super().__init__()

    def _start_service(self, data):
        data.pop("client", None)
        gpo = Gp(**data)
        self.gpo = gpo
        self.task_dict.update({'cancelled': False, 'progress': '0%', 'paused': False})
        self._opt_thread = threading.Thread(
            target=self._run_optimization, args=(gpo,), daemon=True
        )
        self._opt_thread.start()
        return "PSE started"

    def _run_optimization(self, gpo):
        try:
            if not gpo.gp_hardware_intitialzation():
                return
            if gpo.optimizer == 'grid':
                gpo.gridsearch_optimization_loop()
            else:
                gpo.gpcam_optimization_loop()
        except Exception:
            logger.exception("Standalone optimization loop failed.")
        finally:
            gpo.gp_hardware_shutdown()
            self.gpo = None
            self.task_dict['cancelled'] = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    StandaloneGpServer().run(port)
    _ = input("Press enter to exit...")
