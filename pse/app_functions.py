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


def pause_pse(port):
    """
    Pauses the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag.
    :param port: port number for GP server
    :return: (Bool) success flag.
    """
    try:
        communicate_get('/pause_pse', port)
    except requests.exceptions.ConnectionError:
        return False

    return True


def resume_pse(port, **kwargs):
    """
    Resumes the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag.
    :param port: port number for GP server
    :param kwargs: (dict) argurments to be passed through to gp.__init__()
    :return: (Bool) success flag.
    """
    try:
        communicate_post('/resume_pse', port, kwargs)
    except requests.exceptions.ConnectionError:
        return False

    return True


def run_pse(port, **kwargs):
    """
    Initializes and runs the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag. Initializes the instrumentation
    :param port: port number for GP server
    :param kwargs: (dict) argurments to be passed through to gp.__init__()
    :return: (Bool) success flag.
    """
    '''
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
    '''

    try:
        communicate_post('/start_pse', port, kwargs)
    except requests.exceptions.ConnectionError:
        return False

    return True


def stop_pse(port):
    """
    Stops the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag. Shuts down the instrumentation.
    :param port: port number for GP server
    :return: (Bool) success flag.
    """
    try:
        communicate_get('/stop_pse', port)
    except requests.exceptions.ConnectionError:
        return False

    return True


