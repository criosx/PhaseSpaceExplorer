from datetime import datetime
import numpy as np
import os
import pandas
from pathlib import Path
import pickle
import requests
import shutil
import streamlit as st
import time
import uuid

from roadmap_datamanager.gui.streamlit_components import file_browser_button

import pse.configuration


def _copy_directory_contents(src_dir: Path, dst_dir: Path):
    """Copy all files and folders inside src_dir into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_item in src_dir.iterdir():
        dst_item = dst_dir / src_item.name
        if src_item.is_dir():
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

def adjust_PSE_status():
    """
    Aligns the Streamlit knowledge of the server status in st.session_state['pse_jobs_status'] with the reality obtained
    from a server get_status call.
    :return: (bool) whether to rerun the global Streamlit script
    """
    if st.session_state['gp_server_port'] is None:
        return False

    port = st.session_state['gp_server_port']
    status = communicate_get('/get_status', port).text
    jstatus = st.session_state['pse_jobs_status']

    if 'failure' in status:
        st.session_state['pse_jobs_status'] = jstatus
        return False

    rerun_flag = False

    if jstatus == 'pending PSE startup' or jstatus == 'pending PSE resume':
        if status == 'running':
            st.session_state['pse_jobs_status'] = 'running'
        elif status == 'idle':
            # Wait in case startup was just initialized and check for status again. Testing this case is needed, if exit
            # condition is already met at startup (sufficient iterations measured). In this case, the status will not
            # change to 'running'.
            st.session_state['update_counter'] += 1
            if st.session_state['update_counter'] > 1:
                st.session_state['pse_jobs_status'] = 'idle'
                # reset optimization start/pause toggles
                st.session_state['rpse_key'] = str(uuid.uuid4())
                st.session_state['ppse_key'] = str(uuid.uuid4())
                st.session_state['update_counter'] = 0
                rerun_flag = True

    elif jstatus == 'pending PSE pause':
        if status == 'idle':
            st.session_state['pse_jobs_status'] = 'paused'

    elif jstatus == 'pending PSE shutdown':
        if status == 'idle':
            st.session_state['pse_jobs_status'] = 'idle'

    # catches reruns of Streamlit scripts while the server continues in the background
    elif jstatus == 'idle' and status == 'running':
        st.session_state['pse_jobs_status'] = 'running'

    elif jstatus == 'running' and status == 'idle':
        st.session_state['pse_jobs_status'] = 'idle'
        # reset optimization start/pause toggles
        st.session_state['rpse_key'] = str(uuid.uuid4())
        st.session_state['ppse_key'] = str(uuid.uuid4())
        rerun_flag = True

    return rerun_flag


def clear_project_data(everything=False):
    """
    Deletes contents from the PSE directory. If everything is True, the entire directory content is deleted. Otherwise,
    only the results and plots directories are removed.

    :param everything: (bool) whether to delete the entire directory contents or only the results and plots directories.
                        Defaults to False.
    :return: No return value.
    """
    def _rmdir(directory_path):
        # Check if the directory exists
        if not directory_path.is_dir():
            print(f"The directory '{str(directory_path)}' is not a directory.")
            return

        for entry in directory_path.iterdir():
            try:
                if entry.is_file():
                    os.remove(entry)  # Remove the file
                elif entry.is_dir():
                    shutil.rmtree(entry)  # Remove the subdirectory and its contents
            except Exception as e:
                print(f"Failed to delete {entry}: {e}")

    project_dir = st.session_state['pse_dir']
    if project_dir is None:
        return
    project_dir = Path(project_dir).expanduser().resolve()
    if everything:
        _rmdir(project_dir)
    else:
        result_dir = project_dir / 'results'
        plots_dir = project_dir / 'plots'
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


def start_stop_optimization(kwargs=None):
    """
    Implementatio of the start/stop logic of the PSE exploration.
    :param kwargs: (dict) argurments to be passed through ultimately to gp.go_pse(). Base parameters are:
        :opt_acq: (str) the acquisition function of the gp optimizer (None if in grid mode)
        :client: (str) the optimization client (ROADMAP, test function)
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
            kwargs['exp_par'] = kwargs['exp_par'].to_dict(orient='records')
    else:
        st.error('No experimental optimization parameter provided. This is a script error and should not happen.')
        st.stop()

    col_opt_5, col_opt_6 = st.columns([1, 1])
    port = st.session_state['gp_server_port']
    jstatus = st.session_state['pse_jobs_status']

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

    st.session_state['pse_jobs_status'] = jstatus


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
def start_of_script_business():
    """
    Checks the state of session variables at the beginning of the script and initializes them if necessary.
    :return: no return value
    """
    if not st.session_state["data_folders_ready"]:
        st.info("Files and Folders not set up. Please visit the File System tab.")
        st.stop()

    if 'pse_jobs_status' not in st.session_state:
        # valid job status values: pending, idle, running, failure, (down)
        st.session_state['pse_jobs_status'] = 'idle'

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

        # we are running the script the first time, and have reloaded the configuration
        st.session_state['configuration_reloaded'] = True


def clear_project_data_dialog(everything=False):
    col_pse_cpdd_1, col_pse_cpdd_2 = st.columns([3, 1])
    with col_pse_cpdd_1:
        st.info("Project directory: {}".format(st.session_state['pse_dir']))
    with col_pse_cpdd_2:
        file_browser_button(st.session_state['pse_dir'])
        if st.button('Clear Project Data', disabled=(st.session_state['pse_jobs_status'] == 'running'),
                     width='stretch'):
            clear_project_data(everything=everything)
            st.rerun()

@st.fragment(run_every=60)
def monitor():
    # list jobs status
    st.info('Server port: {}'.format(st.session_state['gp_server_port']))
    if adjust_PSE_status():
        st.rerun()
    st.info('Job status: {}'.format(st.session_state['pse_jobs_status']))

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

            if st.session_state['pse_jobs_status'] == 'running':
                if df_res_gpcam.shape[0] >= st.session_state['gp_iterations']:
                    st.session_state['pse_jobs_status'] = 'idle'
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

        figure_path = Path(st.session_state['pse_dir']).expanduser().resolve() / 'plots'
        if figure_path.is_dir():
            png_files.extend(
                file for file in figure_path.iterdir()
                if file.is_file() and file.suffix.lower() == '.png'
            )

    for file in png_files:
        try:
            st.image(file, width='stretch')
        except FileNotFoundError:
            pass

    if st.button('Update job monitor'):
        pass

def pse_directory(identifier:str='PSE', st_directory_identifier:str='pse_dir'):
    """
    Implements a working directory archival and restoration dialog.

    :param identifier:              (str) The leading string of the name of any archive directory such as 'PSE' or 'SANS
                                    Optimization Directory'. The archive will then be created with a name that starts
                                    with that string and adds ' archive' plus any user input.
    :param st_directory_identifier: (str) The st.session_state key under which the path to the optimizatin directory is
                                    stored.
    :return:                        no return value
    """
    clear_project_data_dialog(everything=True)

    pse_dir = Path(st.session_state[st_directory_identifier]).expanduser().resolve()
    archive_root = pse_dir.parent
    cfg: pse.configuration.DataManagerConfig = st.session_state['cfg']   # configuration

    col_opt_a1, col_opt_a2 = st.columns([3, 1])
    if (pse_dir / 'results').is_dir():
        archive_name = col_opt_a1.text_input(
            "Name of archive directory:",
            value= identifier + " archive " + datetime.now().strftime("%Y_%m_%d"),
        )
        if archive_name.startswith(identifier + ' archive'):
            archive_dir = archive_root / archive_name
        else:
            archive_dir = archive_root / (identifier + archive_name)

        if archive_dir.is_dir():
            col_opt_a1.info('Archive exists.')
        if col_opt_a2.button("Create archive of optimization directory", disabled=archive_dir.is_dir()):
                shutil.copytree(str(Path(pse_dir)), archive_dir)
                # save subset of configuration data class members to the archive dir
                cfg.save_subset(path=archive_dir / 'config.json', groups=("pse",))
                col_opt_a2.success('Optimization directory archived.')
    else:
        col_opt_a1.info("The optimization directory does not contain results.")

        archives = sorted(
            p for p in archive_root.iterdir()
            if p.is_dir()
            and p != pse_dir
            and p.name.startswith(identifier + " archive")
        )

        if archives:
            col_opt_a3, col_opt_a4 = st.columns([3, 1])
            archive_names = [p.name for p in archives]

            archive_name = col_opt_a3.selectbox(
                "Archive to restore",
                archive_names,
                index=None,
                placeholder="Choose an archive...",
                key="restore_archive_name",
            )

            if archive_name:
                if col_opt_a4.button("Restore archive"):
                    archive_dir = archive_root / archive_name
                    _copy_directory_contents(archive_dir, pse_dir)
                    # reload PSE-related config entries
                    cfg.load_subset(path=archive_dir / 'config.json', groups=("pse",))
                    st.session_state['configuration_reloaded'] = True
                    col_opt_a4.success("Archive restored.")
                    time.sleep(1)
                    st.rerun()

@st.fragment
def parameter_input():
    if st.session_state.configuration_reloaded:
        st.session_state['pse_input_widget_key'] = uuid.uuid4()
        if st.session_state.cfg.pse_opt_pars:
            df_opt_pars = pandas.DataFrame(st.session_state.cfg.pse_opt_pars)
            st.session_state['opt_pars_original'] = df_opt_pars
    if 'opt_pars_original' not in st.session_state:
        # TODO implement parameter data frame initializion
        df_opt_pars = {'name': ['lipid1', 'lipid2', 'lipid3', 'lipid concentration'],
                       'type': ['compound', 'compound', 'compound', 'parameter'], 'value': [1.0, 1.0, 1.0, 5.0],
                       'lower_opt': 0.0, 'upper_opt': 1.0, 'optimize': False, 'step_opt': 0.01}
        st.session_state['opt_pars_original'] = pandas.DataFrame(df_opt_pars)
    df_opt_pars_original = st.session_state['opt_pars_original']
    parameters_edited = st.data_editor(
        df_opt_pars_original,
        key=st.session_state['pse_input_widget_key'],
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
    st.session_state.cfg.pse_opt_pars = parameters_edited.to_dict(orient='records')

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
        disabled=(st.session_state.pse_jobs_status != 'idle')
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

def end_of_script_business():
    cfg: pse.configuration.DataManagerConfig = st.session_state.cfg
    cfg.save_subset(st.session_state['pse_dir'] / 'config.json', groups=('pse',))
    cfg.save()
    st.session_state['configuration_reloaded'] = False
