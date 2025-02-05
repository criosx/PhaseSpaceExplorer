import glob
import os
import pandas

import shutil
import streamlit as st
import time

import sys
from support import app_functions
sys.path.append(st.session_state['app_functions_dir'])

user_qcmd_opt_dir = st.session_state['user_qcmd_opt_dir']
user_qcmd_file_dir = st.session_state['user_qcmd_file_dir']
user_qcmd_temp_dir = os.path.join(st.session_state['streamlit_dir'], 'temp')

# ------------ Functionality -----------

# ------------  GUI -------------------
file_path = user_qcmd_file_dir
file_list = os.listdir(file_path)
file_list = sorted(element for element in file_list if element[0] != '.')

st.write("""
# Job Monitor
""")

with st.expander('Monitor'):
    jobs_df = app_functions.monitor_jobs(user_qcmd_opt_dir)
    st.dataframe(jobs_df)
    if 'running' in jobs_df['Job Status']:
        jobs_status = 'running'
    else:
        jobs_status = 'finished'

st.write("""
# Setup New Phase Space Exploration
""")

with st.expander('Setup'):

    st.write("""
    ## Parameters
    ### Model Fit
    """)

    # TODO setup dataframe for exploration parameter selection
    data = {
        'name': ['lipid1', 'lipid2', 'lipid3', 'lipid concentration'],
        'type': ['compound', 'compound', 'compound', 'parameter'],
        'value': [1.0, 1.0, 1.0, 5.0]
    }
    df_pars = pandas.DataFrame(data)

    df_pars['lower_opt'] = 0.0
    df_pars['upper_opt'] = 1.0
    df_pars['optimize'] = False
    df_pars['step_opt'] = 1.0
    parameters_edited = st.data_editor(
        df_pars,
        key='opt_pars',
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
    st.session_state['opt_parameters'] = parameters_edited

st.write("""
# Run or Continue Optimization
""")

col_opt_3, col_opt_4 = st.columns([1, 1])

opt_optimizer = col_opt_3.selectbox("optimizer", ['gaussian process regression (GP)', 'grid search', ])
opt_acq = 'variance'
gp_iter = 50
if opt_optimizer == 'gaussian process regression (GP)':
    gp_iter = col_opt_3.number_input('GP iterations', min_value=20, value=1000, format='%i', step=100)
    opt_acq = col_opt_3.selectbox("GP acquisition function", ['shannon_ig_vec', 'ucb', 'variance', 'maximum'])

col_opt_5, col_opt_6 = st.columns([1, 1])
if col_opt_5.button('Start Optimization', disabled=(jobs_status == 'running'), use_container_width=True):
    app_functions.run_pse(pse_pars=df_pars, pse_dir=st.session_state['user_qcmd_opt_dir'], acq_func=opt_acq,
                          optimizer=opt_optimizer, gpcam_iterations=gp_iter)
if col_opt_5.button('Resume Optimization', disabled=(jobs_status == 'running'), use_container_width=True):
    app_functions.run_pse(pse_pars=df_pars, pse_dir=st.session_state['user_qcmd_opt_dir'], acq_func=opt_acq,
                          optimizer=opt_optimizer, gpcam_iterations=gp_iter)
