import copy
import json
import numpy as np
import os
import pandas
from pathlib import Path
import pickle
import requests
import shutil
import streamlit as st
import uuid

def adjust_PSE_status():
    """
    Aligns the Streamlit knowledge of the server status in st.session_state['jobs_status'] with the reality obtained
    from a server get_status call.
    :return: (bool) whether to rerun the global Streamlit script
    """
    if st.session_state['gp_server_port'] is None:
        return False

    port = st.session_state['gp_server_port']
    status = communicate_get('/get_status', port).text
    jstatus = st.session_state['jobs_status']

    if 'failure' in status:
        st.session_state['jobs_status'] = jstatus
        return False

    rerun_flag = False

    if jstatus == 'pending PSE startup' or jstatus == 'pending PSE resume':
        if status == 'running':
            st.session_state['jobs_status'] = 'running'
        elif status == 'idle':
            # Wait in case startup was just initialized and check for status again. Testing this case is needed, if exit
            # condition is already met at startup (sufficient iterations measured). In this case, the status will not
            # change to 'running'.
            st.session_state['update_counter'] += 1
            if st.session_state['update_counter'] > 1:
                st.session_state['jobs_status'] = 'idle'
                # reset optimization start/pause toggles
                st.session_state['rpse_key'] = str(uuid.uuid4())
                st.session_state['ppse_key'] = str(uuid.uuid4())
                st.session_state['update_counter'] = 0
                rerun_flag = True

    elif jstatus == 'pending PSE pause':
        if status == 'idle':
            st.session_state['jobs_status'] = 'paused'

    elif jstatus == 'pending PSE shutdown':
        if status == 'idle':
            st.session_state['jobs_status'] = 'idle'

    # catches reruns of Streamlit scripts while the server continues in the background
    elif jstatus == 'idle' and status == 'running':
        st.session_state['jobs_status'] = 'running'

    elif jstatus == 'running' and status == 'idle':
        st.session_state['jobs_status'] = 'idle'
        # reset optimization start/pause toggles
        st.session_state['rpse_key'] = str(uuid.uuid4())
        st.session_state['ppse_key'] = str(uuid.uuid4())
        rerun_flag = True

    return rerun_flag


def clear_project_data():
    def _rmdir(directory_path):
        # Check if the directory exists
        if not os.path.exists(directory_path):
            print(f"The directory '{directory_path}' does not exist.")
            return

        for entry in os.scandir(directory_path):
            try:
                if entry.is_file():
                    os.remove(entry.path)  # Remove the file
                elif entry.is_dir():
                    shutil.rmtree(entry.path)  # Remove the subdirectory and its contents
            except Exception as e:
                print(f"Failed to delete {entry.path}: {e}")

    project_dir = st.session_state['pse_dir']
    if project_dir is None:
        return
    result_dir = os.path.join(project_dir, 'results')
    plots_dir = os.path.join(project_dir, 'plots')
    _rmdir(result_dir)
    _rmdir(plots_dir)


def communicate_post(endpoint, port, data):
    """
    Communicate with GP server.
    :param endpoint: endpoint
    :param port: port
    :param data: data as a dict or JSON
    :return: server response in JSON format
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
    :return: server response in JSON format
    """
    print('\n')
    print('Submitting data to endpoint {}.'.format(endpoint))
    url = 'http://127.0.0.1:' + str(port) + endpoint
    print('Connecting to server with the following url: {}'.format(url))
    response = requests.get(url)
    print(response, response.text)
    return response


def load_session_state(folder):
    file_path = os.path.join(folder, 'pse_parameters.pkl')
    if os.path.exists(file_path):
        print('trying to load pse_paramters.pkl')
        with open(file_path, 'rb') as f:
            st.session_state['opt_pars'] = pickle.load(f)
        st.session_state['opt_pars_original'] = st.session_state['opt_pars']
        print('loaded pse_paramters.pkl')
        print(st.session_state['opt_pars_original'])
    else:
        # create new session
        project_dir = Path(st.session_state['pse_dir']).expanduser().resolve()
        result_dir = project_dir / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = project_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # TODO implement parameter data frame initializion
        df_opt_pars = {'name': ['lipid1', 'lipid2', 'lipid3', 'lipid concentration'],
                       'type': ['compound', 'compound', 'compound', 'parameter'], 'value': [1.0, 1.0, 1.0, 5.0],
                       'lower_opt': 0.0, 'upper_opt': 1.0, 'optimize': False, 'step_opt': 0.01}
        st.session_state['opt_pars_original'] = df_opt_pars
        st.session_state['opt_pars'] = df_opt_pars

        save_session_state(project_dir)


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


"""
def run_measurement(kwargs):
    success = streamlit_components.run_pse(**kwargs)
    if success:
        st.session_state['job_status'] = 'idle'
    else:
        st.session_state['job_status'] = 'failed measurement'
"""


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


def save_session_state(folder):
    # save only pse parameters so far
    file_path = os.path.join(folder, 'pse_parameters.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(st.session_state['opt_pars'], f)
    # also save a json record
    path_name = os.path.join(folder, 'pse_parameters.json')
    with open(path_name, 'w') as file:
        json.dump(st.session_state['opt_pars'], file)


def start_stop_optimization(kwargs=None):
    """
    Implementatio of the start/stop logic of the PSE exploration.
    :param kwargs: (dict) argurments to be passed through ultimately to gp.go_pse(). Base parameters are:
        :opt_acq: (str) the acquisition function of the gp optimizer (None if in grid mode)
        :client: (str) the optimization client (ROADMAP, test function, i.e.)
        :opt_optimizer: (str) optimizer ('grid' or 'gpcam')
        :init_iter: (int | None) initial (burn in) iterations for gpcam
        :gp_iter: (int | None) number of iterations for gpcam
        :parallel_meas: (int) number of parallel measurements to be executed
        :gp_discrete_points: (np array-like | None) optional discrete evaluation points
        :storage_path: (str | Path-like) path to PSE storage folder
        :exp_par: (Pandas dataframe converted to JSON) experimental parameters for the PSE exploration
        :resume: (bool) whether to resume the PSE exploration (default: True)
        :project_name: (str) project name
    :return: no return value
    """
    # validate inputs
    save_exists = os.path.isfile(os.path.join(st.session_state['pse_dir'], 'evaluation_points.json'))
    if save_exists:
        reuse_points = st.checkbox('Reuse saved evaluation points', value=True)
        if reuse_points:
            kwargs['gp_discrete_points'] = 'default file'

    if 'exp_par' in kwargs:
        if isinstance(kwargs['exp_par'], pandas.DataFrame):
            print('Actually, I was here.')
            kwargs['exp_par'] = kwargs['exp_par'].to_dict(orient='records')

    col_opt_5, col_opt_6 = st.columns([1, 1])
    port = st.session_state['gp_server_port']
    jstatus = st.session_state['jobs_status']

    # find presets in case of first run
    if jstatus == 'running':
        rpse_first = True
        ppse_first = False
    elif jstatus == 'paused':
        rpse_first = True
        ppse_first = True
    else:
        rpse_first = False
        ppse_first = False

    # force rerendering of toggle widgets
    rpse_key = st.session_state['rpse_key']
    ppse_key = st.session_state['ppse_key']

    rpse = col_opt_5.toggle('Run PSE', value=rpse_first, key=rpse_key)
    ppse = col_opt_6.toggle('Pause PSE', disabled=(not rpse), value=ppse_first, key=ppse_key)

    if jstatus == 'running':
        if not rpse:
            if stop_pse(port):
                jstatus = 'pending PSE shutdown'
            else:
                jstatus = 'failure - PSE shutdown'
        elif ppse:
            if pause_pse(port):
                jstatus = 'pending PSE pause'
            else:
                jstatus = 'failure - PSE pause'
    elif jstatus == 'idle':
        if rpse and not ppse:
            st.session_state['gp_iterations'] = kwargs['gpcam_iterations']
            if run_pse(port, **kwargs):
                jstatus = 'pending PSE startup'
            else:
                jstatus = 'failure - PSE startup'
    elif jstatus == 'paused':
        if not rpse:
            if stop_pse(port):
                jstatus = 'pending PSE shutdown'
            else:
                jstatus = 'failure - PSE shutdown'
        elif not ppse:
            st.session_state['gp_iterations'] = kwargs['gpcam_iterations']
            if resume_pse(port, **kwargs):
                jstatus = 'pending PSE resume'
            else:
                jstatus = 'failure - PSE resume'

    st.session_state['jobs_status'] = jstatus


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


# --------------  components ---------------------
@st.fragment
def check_session_state():
    """
    Checks the state of session variables at the beginning of the script and initializes them if necessary.
    :return: no return value
    """

    if 'jobs_status' not in st.session_state:
        # valid job status values: pending, idle, running, failure, (down)
        st.session_state['jobs_status'] = 'idle'


        # Jobs status values for PSE
        # down - no answer from server
        # idle - server up, instruments not initialized
        # instruments initialized - PSE ready to go
        # running - PSE running
        # paused - PSE paused

        # pending PSE startup -
        # pending PSE shutdown -
        # pending PSE pause -
        # pending PSE resume -

        load_session_state(st.session_state['pse_dir'])

    if 'widget_key' not in st.session_state:
        st.session_state['widget_key'] = str(uuid.uuid4())
    if 'gp_iterations' not in st.session_state:
        st.session_state['gp_iterations'] = 50
    if 'measurement_process' not in st.session_state:
        st.session_state['measurement_process'] = None


@st.fragment
def clear_project_data_dialog():
    st.info("Project directory: {}".format(st.session_state['pse_dir']))
    if st.button('Clear Project Data', disabled=(st.session_state['jobs_status'] == 'running'),
                 width='stretch'):
        clear_project_data()


@st.fragment(run_every=60)
def monitor():
    # list jobs status
    st.info('Server port: {}'.format(st.session_state['gp_server_port']))
    if adjust_PSE_status():
        st.rerun()
    st.info('Job status: {}'.format(st.session_state['jobs_status']))

    # List to store paths to .png files
    png_files = []
    if st.session_state['pse_dir'] is not None:

        # List current iterations to be worked on
        ci_path = os.path.join(st.session_state['pse_dir'], 'results', 'current_iterations.pkl')
        if os.path.exists(ci_path):
            with open(ci_path, 'rb') as file:
                df_ci = pandas.DataFrame(pickle.load(file))
            st.text("Current measurements in progress:")
            st.dataframe(df_ci, hide_index=True)

        res_path_gpcam = os.path.join(st.session_state['pse_dir'], 'results', 'gpCAMstream.pkl')
        res_path_grid = os.path.join(st.session_state['pse_dir'], 'results', 'pse_grid_results.pkl')
        if os.path.exists(res_path_gpcam):
            with open(res_path_gpcam, 'rb') as file:
                df_res_gpcam = pandas.DataFrame(pickle.load(file))
            st.text("Finished measurements:")
            st.dataframe(df_res_gpcam, hide_index=False, width='stretch')

            if st.session_state['jobs_status'] == 'running':
                if df_res_gpcam.shape[0] >= st.session_state['gp_iterations']:
                    st.session_state['jobs_status'] = 'idle'
        elif os.path.exists(res_path_grid):
            with open(res_path_grid, 'rb') as file:
                res_grid = pickle.load(file)
            index_combinations = np.array(list(np.ndindex(res_grid.shape)))
            values = res_grid.flatten()

            opt_pars = pandas.DataFrame(st.session_state['opt_pars'])
            opt_pars = opt_pars[opt_pars['optimize']]
            name_pars = opt_pars['name'].tolist()

            if opt_pars.empty:
                st.info('No optimized parameters.')
                return

            # List of exploration steps and axes
            steplist = []
            axes = []
            for row in opt_pars.itertuples():
                steps = int((row.upper_opt - row.lower_opt) / row.step_opt) + 1
                steplist.append(steps)
                axis = []
                for i in range(steps):
                    axis.append(row.lower_opt + i * row.step_opt)
                axes.append(axis)
            axes = np.array(axes)

            index_combinations_mapped = np.stack(
                [axes[j][index_combinations[:, j]] for j in range(index_combinations.shape[1])],
                axis=-1
            )

            index_combinations = list(index_combinations_mapped)

            df_res_grid = pandas.DataFrame(index_combinations, columns=[name_pars[i] for i in range(res_grid.ndim)])
            df_res_grid["result"] = values
            st.text("Measurement Results:")
            st.dataframe(df_res_grid, hide_index=False, width='stretch')

        else:
            st.text("No results to show.")

        figure_path = os.path.join(st.session_state['pse_dir'], 'plots')
        # Iterate over all entries in the directory
        for entry in os.listdir(figure_path):
            # Construct full path
            full_path = os.path.join(figure_path, entry)
            # Check if it's a file and ends with .png
            if os.path.isfile(full_path) and entry.lower().endswith('.png'):
                png_files.append(full_path)

    for file in png_files:
        try:
            st.image(file, width='stretch')
        except FileNotFoundError:
            pass

    if st.button('Update job monitor'):
        pass


@st.fragment
def parameter_input():
    df_opt_pars = copy.deepcopy(st.session_state['opt_pars_original'])
    parameters_edited = st.data_editor(
        df_opt_pars,
        key=st.session_state['widget_key'],
        disabled=["_index"],
        num_rows='dynamic',
        column_order=["name", "type", "value", "optimize", "lower_opt", "upper_opt",
                      "step_opt"],
        column_config={
            'name': 'name',
            'type': st.column_config.SelectboxColumn(
                "type",
                help="Variable type",
                options=['compound', 'parameter']
            ),
            'lower_opt': 'lower opt',
            'upper_opt': 'upper',
            'optimize': 'optimize',
            'step_opt': 'step'
        }
    )
    st.session_state['opt_pars'] = parameters_edited
    save_session_state(st.session_state['pse_dir'])

def run_control(configuration, gp_discrete_points=None, kwargs=None):
    """
    Implements the run/stop, pause/unpause section of the Streamlit GUI. If the PSE should be carried out over a set of
    discrete evaluation points, they should be provided.
    :param kwargs: additional keyword arguments to be passed on to the gp object (for subclassing)
    :param configuration: the configuration module used for the particular application
    :param gp_discrete_points: a set of discrete evaluation points
    :return: no return value
    """

    col_opt_rc3, col_opt_rc4 = st.columns([1, 1])

    # optimizer GPCam vs. grid
    opts = ['gpcam', 'grid', ]
    idx = opts.index(st.session_state.cfg.optimizer)
    opt_optimizer = col_opt_rc3.selectbox(
        label="optimizer",
        options=opts,
        index=idx,
        disabled=(st.session_state.jobs_status != 'idle')
    )
    st.session_state.cfg.optimizer = opt_optimizer

    if opt_optimizer == 'gpcam':
        gp_iter = col_opt_rc3.number_input('GP iterations', min_value=20, value=st.session_state.cfg.gp_iterations,
                                         format='%i', step=100)
        init_iter = col_opt_rc3.number_input('Initial Measurments', min_value=1,
                                           value=st.session_state.cfg.initial_iterations, format='%i', step=1)
        opts = ['variance', 'ucb', 'lcb', 'maximum', 'minimum', 'gradient', 'total correlation', 'expected improvement',
                'probability of improvement', 'relative information entropy', 'relative information entropy set',
                'target probability']
        idx = opts.index(st.session_state.cfg.acquisition_function)
        opt_acq = col_opt_rc3.selectbox("GP acquisition function", opts, index=idx)
        st.session_state.cfg.gp_iterations = gp_iter
        st.session_state.cfg.initial_iterations = init_iter
        st.session_state.cfg.acquisition_function = opt_acq
    else:
        opt_acq = None
        init_iter = None
        gp_iter = None

    opts = ['ROADMAP', 'Test Ackley Function']
    idx = opts.index(st.session_state.cfg.client)
    client = col_opt_rc4.selectbox("client", opts, index=idx)
    st.session_state.cfg.client = client
    parallel_meas = col_opt_rc4.number_input('Parallel measurements', min_value=1,
                                           value=st.session_state.cfg.parallel_measurements, step=1, format='%i')
    st.session_state.cfg.parallel_measurements = parallel_meas

    configuration.save_persistent_cfg(st.session_state.cfg)



    kwargs2 = {
        'storage_path': str(st.session_state['pse_dir']),
        'acq_func': opt_acq,
        'client': client,
        'optimizer': opt_optimizer,
        'gp_discrete_points': gp_discrete_points,
        'gpcam_init_dataset_size': init_iter,
        'gpcam_iterations': gp_iter,
        'parallel_measurements': parallel_meas,
        'resume': True,
        'project_name': st.session_state.cfg.experiment
    }
    if kwargs is None:
        kwargs = kwargs2
    else:
        # function arguments in kwargs have priority over kwargs2
        kwargs2.update(kwargs)
        kwargs = kwargs2

    start_stop_optimization(kwargs)


