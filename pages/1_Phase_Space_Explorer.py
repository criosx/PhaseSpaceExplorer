import copy
import os
import pandas
import pickle
import shutil
import streamlit as st
import time
import tkinter as tk
from tkinter import filedialog
import uuid

import sys
from support import app_functions
sys.path.append(st.session_state['app_functions_dir'])

if 'jobs_status' not in st.session_state:
    st.session_state['jobs_status'] = 'idle'
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(uuid.uuid4())


# ------------ Functionality -----------
def activate_project(project_name):
    print('activating ...')
    project_dir = os.path.join(st.session_state['streamlit_dir'], project_name, 'phase_space')
    st.session_state['user_qcmd_opt_dir'] = project_dir
    st.session_state['active_project'] = project_name
    print('trying to load from {}'.format(project_dir))
    load_session_state(project_dir)


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


def save_session_state(folder):
    # save only pse parameters so far
    file_path = os.path.join(folder, 'pse_parameters.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(st.session_state['opt_pars'], f)


@st.fragment
def parameter_input():
    if st.session_state['active_project'] is not None:
        df_opt_pars = copy.deepcopy(st.session_state['opt_pars_original'])
        parameters_edited = st.data_editor(
            df_opt_pars,
            key=st.session_state['widget_key'],
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

        res_path = os.path.join(st.session_state['user_qcmd_opt_dir'], 'results', 'gpCAMstream.pkl')
        if os.path.exists(res_path):
            with open(res_path, 'rb') as file:
                df_res_gpcam = pandas.DataFrame(pickle.load(file))
            st.text("Finished measurements:")
            st.dataframe(df_res_gpcam, hide_index=False, use_container_width=True)

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
            print('trying to activate')
            activate_project(project)
            st.session_state['widget_key'] = str(uuid.uuid4())

    new_project = st.text_input("... or create a new project directory.", placeholder="Project name ...")
    if new_project != '' and new_project not in project_list:
        create_new_project(new_project)
        st.session_state['widget_key'] = str(uuid.uuid4())

    if st.session_state['user_qcmd_opt_dir'] is not None:
        st.info("Project directory: {}".format(st.session_state['user_qcmd_opt_dir']))
        st.info("Active project: {}".format(st.session_state['active_project']))
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

col_opt_3, col_opt_4 = st.columns([1, 1])

opt_optimizer = col_opt_3.selectbox("optimizer", ['gpcam', 'grid', ])
opt_acq = 'variance'
gp_iter = 50
if opt_optimizer == 'gpcam':
    gp_iter = col_opt_3.number_input('GP iterations', min_value=20, value=1000, format='%i', step=100)
    opt_acq = col_opt_3.selectbox("GP acquisition function", ['shannon_ig_vec', 'ucb', 'variance', 'maximum'])

col_opt_5, col_opt_6 = st.columns([1, 1])
st.info('Job status: {}'.format(st.session_state['jobs_status']))
if col_opt_5.button('Start or Resume Optimization', disabled=(st.session_state['jobs_status'] == 'running'),
                    use_container_width=True):
    st.session_state['jobs_status'] = 'running'
    app_functions.run_pse(pse_pars=pandas.DataFrame(st.session_state['opt_pars']),
                          pse_dir=st.session_state['user_qcmd_opt_dir'],
                          acq_func=opt_acq, optimizer=opt_optimizer, gpcam_iterations=gp_iter)

