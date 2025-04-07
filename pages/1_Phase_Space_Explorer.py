import copy
import json
import numpy as np
import os
import pandas
import pickle
import shutil
import streamlit as st
import threading
import uuid

import sys
from support import app_functions
sys.path.append(st.session_state['app_functions_dir'])

if 'jobs_status' not in st.session_state:
    st.session_state['jobs_status'] = 'idle'
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(uuid.uuid4())
if 'gp_iterations' not in st.session_state:
    st.session_state['gp_iterations'] = 50
if 'measurement_process' not in st.session_state:
    st.session_state['measurement_process'] = None


# ------------ Functionality -----------
def activate_project(project_name):
    print('activating ...')
    project_dir = os.path.join(st.session_state['streamlit_dir'], project_name, 'phase_space')
    st.session_state['user_qcmd_opt_dir'] = project_dir
    st.session_state['active_project'] = project_name
    print('trying to load from {}'.format(project_dir))
    load_session_state(project_dir)


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

    project_dir = st.session_state['user_qcmd_opt_dir']
    if project_dir is None:
        return
    result_dir = os.path.join(project_dir, 'results')
    plots_dir = os.path.join(project_dir, 'plots')
    _rmdir(result_dir)
    _rmdir(plots_dir)


def create_new_project(project_name):
    project_dir1 = str(os.path.join(st.session_state['streamlit_dir'], project_name))
    os.mkdir(project_dir1)
    project_dir = os.path.join(project_dir1, 'phase_space')
    os.mkdir(project_dir)
    result_dir = os.path.join(project_dir, 'results')
    os.mkdir(result_dir)
    plots_dir = os.path.join(project_dir, 'plots')
    os.mkdir(plots_dir)

    # TODO implement parameter data frame initializion
    df_opt_pars = {'name': ['lipid1', 'lipid2', 'lipid3', 'lipid concentration'],
                   'type': ['compound', 'compound', 'compound', 'parameter'], 'value': [1.0, 1.0, 1.0, 5.0],
                   'lower_opt': 0.0, 'upper_opt': 1.0, 'optimize': False, 'step_opt': 0.01}
    st.session_state['opt_pars_original'] = df_opt_pars
    st.session_state['opt_pars'] = df_opt_pars
    st.session_state['active_project'] = project_name
    st.session_state['user_qcmd_opt_dir'] = project_dir

    save_session_state(project_dir)


def load_session_state(folder):
    file_path = os.path.join(folder, 'pse_parameters.pkl')
    if os.path.exists(file_path):
        print('trying to load pse_paramters.pkl')
        with open(file_path, 'rb') as f:
            st.session_state['opt_pars'] = pickle.load(f)
        st.session_state['opt_pars_original'] = st.session_state['opt_pars']
        print('loaded pse_paramters.pkl')
        print(st.session_state['opt_pars_original'])


@st.fragment(run_every=60)
def monitor():
    st.info('Job status: {}'.format(st.session_state['jobs_status']))

    # List to store paths to .png files
    png_files = []
    if st.session_state['user_qcmd_opt_dir'] is not None:

        # List current iterations to be worked on
        ci_path = os.path.join(st.session_state['user_qcmd_opt_dir'], 'results', 'current_iterations.pkl')
        if os.path.exists(ci_path):
            with open(ci_path, 'rb') as file:
                df_ci = pandas.DataFrame(pickle.load(file))
            st.text("Current measurements in progress:")
            st.dataframe(df_ci, hide_index=True)

        res_path_gpcam = os.path.join(st.session_state['user_qcmd_opt_dir'], 'results', 'gpCAMstream.pkl')
        res_path_grid = os.path.join(st.session_state['user_qcmd_opt_dir'], 'results', 'pse_grid_results.pkl')
        if os.path.exists(res_path_gpcam):
            with open(res_path_gpcam, 'rb') as file:
                df_res_gpcam = pandas.DataFrame(pickle.load(file))
            st.text("Finished measurements:")
            st.dataframe(df_res_gpcam, hide_index=False, use_container_width=True)

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
            st.dataframe(df_res_grid, hide_index=False, use_container_width=True)

        else:
            st.text("No results to show.")

        figure_path = os.path.join(st.session_state['user_qcmd_opt_dir'], 'plots')
        # Iterate over all entries in the directory
        for entry in os.listdir(figure_path):
            # Construct full path
            full_path = os.path.join(figure_path, entry)
            # Check if it's a file and ends with .png
            if os.path.isfile(full_path) and entry.lower().endswith('.png'):
                png_files.append(full_path)

    for file in png_files:
        st.image(file, use_container_width=True)

    if st.button('Update job monitor'):
        pass


def run_measurment(kwargs):
    success = app_functions.run_pse(**kwargs)

    if success:
        st.session_state['job_status'] = 'idle'
    else:
        st.session_state['job_status'] = 'failed measurement'


def save_session_state(folder):
    # save only pse parameters so far
    file_path = os.path.join(folder, 'pse_parameters.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(st.session_state['opt_pars'], f)
    # also save a json record
    path_name = os.path.join(folder, 'pse_parameters.json')
    with open(path_name, 'w') as file:
        json.dump(st.session_state['opt_pars'], file)


@st.fragment
def parameter_input():
    if st.session_state['active_project'] is not None:
        df_opt_pars = copy.deepcopy(st.session_state['opt_pars_original'])
        parameters_edited = st.data_editor(
            df_opt_pars,
            key=st.session_state['widget_key'],
            num_rows='dynamic',
            disabled=["_index"],
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
        save_session_state(st.session_state['user_qcmd_opt_dir'])


# ------------  GUI -------------------
st.write("""
# Job Monitor
""")

with (st.expander('Monitor')):
    monitor()

st.write("""
# Setup New Phase Space Exploration
""")

with st.expander('Setup'):

    st.write("## File System")
    project_list = [entry for entry in os.listdir(st.session_state['streamlit_dir'])
                    if os.path.isdir(os.path.join(st.session_state['streamlit_dir'], entry))]

    project = st.selectbox('Choose Existing Project', project_list, placeholder="Select Project ...")
    if project is not None:
        project_dir = os.path.join(st.session_state['streamlit_dir'], project, 'phase_space')
        if project_dir != st.session_state['user_qcmd_opt_dir']:
            activate_project(project)
            st.session_state['widget_key'] = str(uuid.uuid4())

    new_project = st.text_input("... or create a new project directory.", placeholder="Project name ...")
    if new_project != '' and new_project not in project_list:
        create_new_project(new_project)
        st.session_state['widget_key'] = str(uuid.uuid4())

    if st.session_state['user_qcmd_opt_dir'] is not None:
        st.info("Project directory: {}".format(st.session_state['user_qcmd_opt_dir']))
        st.info("Active project: {}".format(st.session_state['active_project']))

        if st.button('Clear Project Data', disabled=(st.session_state['jobs_status'] == 'running'),
                     use_container_width=True):
            clear_project_data()

        if st.session_state['jobs_status'] == 'running':
            if st.button('Set status to idle', use_container_width=True):
                st.session_state['jobs_status'] = 'idle'

    else:
        st.info("No active project")

    st.write("""
    ## Parameters
    ### Model Fit
    """)

    # TODO setup dataframe for exploration parameter selection
    parameter_input()

st.write("""
# Run or Continue Optimization
""")

res_path_gpcam = os.path.join(st.session_state['user_qcmd_opt_dir'], 'results', 'gpCAMstream.pkl')
res_path_grid = os.path.join(st.session_state['user_qcmd_opt_dir'], 'results', 'pse_grid_results.pkl')
if os.path.exists(res_path_gpcam):
    present_optimizer = 'gpcam'
elif os.path.exists(res_path_grid):
    present_optimizer = 'grid'
else:
    present_optimizer = None

col_opt_3, col_opt_4 = st.columns([1, 1])
if present_optimizer is None:
    opt_optimizer = col_opt_3.selectbox("optimizer", ['gpcam', 'grid', ])
else:
    col_opt_3.text("optimizer: {}".format(present_optimizer))
    opt_optimizer = present_optimizer

opt_acq = 'variance'
gp_iter = 50
if opt_optimizer == 'gpcam':
    gp_iter = col_opt_3.number_input('GP iterations', min_value=20, value=1000, format='%i', step=100)
    opt_acq = col_opt_3.selectbox("GP acquisition function", ['variance', 'ucb', 'relative information entropy',
                                                              'probability of improvement'])

parallel_meas = col_opt_4.number_input('Parallel measurements', min_value=1, value=1, step=1, format='%i')

col_opt_5, col_opt_6 = st.columns([1, 1])

if col_opt_5.button('Start or Resume Optimization', disabled=(st.session_state['jobs_status'] == 'running'),
                    use_container_width=True):
    if st.session_state['jobs_status'] == 'idle':
        st.session_state['gp_iterations'] = gp_iter
        st.session_state['jobs_status'] = 'running'
        kwargs = {'pse_pars': pandas.DataFrame(st.session_state['opt_pars']),
                  'pse_dir': st.session_state['user_qcmd_opt_dir'],
                  'acq_func': opt_acq,
                  'optimizer': opt_optimizer,
                  'gpcam_iterations': gp_iter,
                  'parallel_measurements': parallel_meas,
                  'project_name': st.session_state['active_project']
                  }
        thread = threading.Thread(target=app_functions.run_pse, kwargs=kwargs)
        st.session_state['job_status'] = 'running'
        thread.start()
        st.rerun()


