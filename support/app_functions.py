from multiprocessing import Process
import os
import requests
import time
from support import gp_server


def communicate(endpoint, port, data):
    """
    Communicate with GP server.
    :param endpoint: endpoint
    :param port: port
    :param data: data as a dictionary
    :return: server response in json format
    """
    print('\n')
    print('Submitting data to endpoint {}.'.format(endpoint))
    url = 'http://localhost:' + str(port) + endpoint
    headers = {'Content-Type': 'application/json'}
    data = data.json()
    response = requests.post(url, headers=headers, data=data)
    print(response, response.text)
    return response.json()


def run_pse(pse_pars, pse_dir, acq_func="variance", optimizer='gpcam', gpcam_iterations=50, parallel_measurements=1):
    """
    Initializes and runs the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag.
    :param pse_pars: (dict) PSE parameters, those that are being optimized and those that are not.
    :param pse_dir: (str) Path to the directory where the results are saved.
    :param acq_func: (str) gpCAM acquisition function
    :param optimizer: (str) PSE optimization algorithm (gpcam or grid)
    :param gpcam_iterations: (int) number of Gaussian Process iterations
    :param parallel_measurements: (int) number of parallel measurements per iteration
    :return: (Bool) success flag.
    """

    # check if previous port file exists and delete it if it does
    fp = os.path.join(pse_dir, 'service_port.txt')
    if os.path.isfile(fp):
        os.remove(fp)

    flask_process = Process(target=gp_server.start_server, args=pse_dir)
    flask_process.start()

    # check 10 times if new service port is available
    success = False
    for i in range(10):
        if os.path.isfile(fp):
            with open(fp, "r") as f:
                port = f.read().strip()
                port = int(port)

            kwdir = {
                'exp_par': pse_pars,
                'storage_path': pse_dir,
                'acq_func': acq_func,
                'optimizer': optimizer,
                'gpcam_iterations': gpcam_iterations,
                'parallel_measurements': parallel_measurements,
                'resume': True
            }
            communicate('/start_pse', port, kwdir)

        else:
            time.sleep(2)

    return success

