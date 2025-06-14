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


def run_pse(**kwargs):
    """
    Initializes and runs the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag.
    :param kwargs: (dict) argurments to be passed through to gp.__init__()
    :return: (Bool) success flag.
    """

    # check if previous port file exists and delete it if it does
    pse_dir = kwargs['storage_path']
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

            communicate_post('/start_pse', port, kwargs)
            success = True
            break

        else:
            time.sleep(2)

    return success, port


