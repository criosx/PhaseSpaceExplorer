import os
import requests
import subprocess
import time
from pse import gp_server


def communicate_post(endpoint, port, data):
    """
    Communicate with GP server.
    :param endpoint: endpoint
    :param port: port
    :param data: data as a dict or json
    :return: server response in json format
    """
    print('\n')
    print('Submitting data to endpoint {}.'.format(endpoint))
    url = 'http://127.0.0.1:' + str(port) + endpoint
    print('Connecting to server with the following url: {}'.format(url))
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=data)
    print(response, response.text)
    return response


def communicate_get(endpoint, port):
    """
    Communicate with GP server via GET.
    :param endpoint: endpoint
    :param port: port
    :return: server response in json format
    """
    print('\n')
    print('Submitting data to endpoint {}.'.format(endpoint))
    url = 'http://127.0.0.1:' + str(port) + endpoint
    print('Connecting to server with the following url: {}'.format(url))
    response = requests.get(url)
    print(response, response.text)
    return response


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

    server_path = gp_server.__file__
    subprocess.Popen(['python', server_path, pse_dir])
    # flask_process = Process(target=gp_server.start_server, args=pse_dir)
    # flask_process.start()



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

            start = time.time()
            timeout = 10
            while True:
                try:
                    response = communicate_get('/', port)
                    if response.status_code in {200, 404, 405}:  # server is alive
                        break
                except requests.exceptions.ConnectionError:
                    if time.time() - start > timeout:
                        raise TimeoutError(f"PSE server did not start in time.")
                    time.sleep(0.5)  # try again soon

            communicate_post('/start_pse', port, kwdir)
            print('Phase Space Exploration started.')
            break

        else:
            time.sleep(2)

    return success

