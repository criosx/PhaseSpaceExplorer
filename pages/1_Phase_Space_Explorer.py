import glob
import os
import pandas
import plotly.graph_objects as go
from scattertools.support import api_sasview
import shutil
import streamlit as st
import time

import sys
import imports.app_functions as app_functions
sys.path.append(st.session_state['app_functions_dir'])


user_qcmd_opt_dir = st.session_state['user_qcmd_opt_dir']
user_qcmd_file_dir = st.session_state['user_qcmd_file_dir']
user_qcmd_temp_dir = os.path.join(st.session_state['streamlit_dir'], 'temp')


# ------------ Functionality -----------
def adjust_consecutive_configurations(reference_config):
    if len(st.session_state['df_opt_config_updated']) > 1:
        for j in range(1, len(st.session_state['df_opt_config_updated'])):
            change_flag = False
            # shorthand handles
            df0 = st.session_state['df_opt_config_updated'][0]
            dfj = st.session_state['df_opt_config_updated'][j]
            # modify successive configurations if any setting is shared with first configuration
            for par in df0.index.values:
                # delete a shared setting from consecutive data frames
                if df0.loc[df0.index == par, 'shared'].iat[0]:
                    df_temp = dfj.loc[dfj.index != par]
                    if len(df_temp.index) != len(dfj.index):
                        dfj = df_temp
                        change_flag = True
                # add unshared settings back
                elif par not in dfj.index.values:
                    if par in st.session_state['df_opt_config_default'][j].index.values:
                        dfa = st.session_state['df_opt_config_default'][j]
                        dfj = pandas.concat([dfj, dfa.loc[dfa.index == par]])
                        dfj = dfj.sort_index()
                        # dfj.reset_index(inplace=True)
                        change_flag = True
            if change_flag:
                st.session_state['df_opt_config_associated'][j] = dfj.copy(deep=True)
                # change key of data editor associated with that particular data frame
                st.session_state['df_opt_config_key'] = str(time.time())


def summarize_optimization_parameter_settings():
    li_summary = []
    dfb = st.session_state['opt_background']

    # add model parameters
    for index, row in st.session_state['opt_parameters'].iterrows():
        parname = index
        # ["value", "lowerlimit", "upperlimit", "relative", "lower_opt", "upper_opt", "optimize"]
        if row['type'] == 'information':
            partype = 'd'
        else:
            partype = 'i'
        if not row['relative']:
            partype = 'f' + partype

        if index in dfb['source'].values:
            dataset = str(dfb.loc[dfb['source'] == index, 'dataset'].at[0])
            parconfig = '*'
        elif index in dfb['sink'].values:
            dataset = 'b'+str(dfb.loc[dfb['sink'] == index, 'dataset'].at[0])
            parconfig = '*'
        else:
            dataset = '-'
            parconfig = '-'

        if row['optimize']:
            lower_opt = str(row['lower_opt'])
            upper_opt = str(row['upper_opt'])
            step_opt = str(row['step_opt'])
        else:
            lower_opt = ''
            upper_opt = ''
            step_opt = ''
        li_summary.append([partype, dataset, parconfig, parname, row['value'], row['lowerlimit'], row['upperlimit'],
                           lower_opt, upper_opt, step_opt])

    # add configuration settings
    df0 = st.session_state['df_opt_config_updated'][0]
    num_config = len(st.session_state['df_opt_config_updated'])
    for i, config in enumerate(st.session_state['df_opt_config_updated']):
        for index, row in config.iterrows():
            parname = index
            if parname in df0.index.values and (num_config == 1 or df0.loc[index, 'shared']):
                parconfig = '*'
            else:
                parconfig = str(i)
            if row['optimize']:
                lfit = ufit = '0'
                lower_opt = str(row['lower_opt'])
                upper_opt = str(row['upper_opt'])
                step_opt = str(row['step_opt'])
            else:
                lfit = ufit = lower_opt = upper_opt = step_opt = ''

            li_summary.append(['n', '*', parconfig, parname, row['value'], lfit, ufit, lower_opt, upper_opt, step_opt])

    df_summary = pandas.DataFrame(li_summary, columns=['type', 'dataset', 'config.', 'parameter', 'value', 'l_fit',
                                                       'u_fit', 'l_opt', 'u_opt', 'step_opt'])

    return df_summary


def update_df_config(config_list_select):

    # function argument homogeineization
    if config_list_select is not None:
        if not isinstance(config_list, list):
            config_list_select = [config_list_select]
    else:
        config_list_select = []

    # only run update if config_lisit_select has changed
    if 'opt_config_list_select' in st.session_state and \
            st.session_state['opt_config_list_select'] == config_list_select:
        return
    else:
        st.session_state['opt_config_list_select'] = config_list_select

    df_config = []
    for config_name in config_list_select:
        df_config.append(pandas.read_json(os.path.join(config_path, config_name), orient='record'))

    for i, config in enumerate(df_config):
        if len(df_config) > 1:
            df_config[i]['shared'] = False
        df_config[i]['lower_opt'] = 0.0
        df_config[i]['upper_opt'] = 1.0
        df_config[i]['step_opt'] = 0.0
        df_config[i]['optimize'] = False
        df_config[i] = config.sort_values('setting')
        df_config[i].set_index('setting', inplace=True)

    dfcopy1 = []
    dfcopy2 = []
    for i, element in enumerate(df_config):
        dfcopy1.append(element.copy(deep=True))
        dfcopy2.append(element.copy(deep=True))
    st.session_state['df_opt_config_default'] = dfcopy1
    st.session_state['df_opt_config_updated'] = dfcopy2
    st.session_state['df_opt_config_associated'] = []
    for _ in range(len(df_config)):
        st.session_state['df_opt_config_associated'].append(None)

    st.session_state['df_opt_config_key'] = []
    for _ in range(len(df_config)):
        st.session_state['df_opt_config_key'].append(str(time.time()))


# ------------  GUI -------------------
file_path = user_qcmd_file_dir
file_list = os.listdir(file_path)
file_list = sorted(element for element in file_list if element[0] != '.')

st.write("""
# Job Monitor
""")

with st.expander('Monitor'):
    jobtime, status = app_functions.monitor_jobs(user_qcmd_opt_dir)
    if status == 'idle':
        st.info('No active job.')
    if status == 'running':
        st.info('Job in progress. Last update at ', time.localtime(jobtime))
    if status == 'finished':
        st.info('Job finished at ', time.localtime(jobtime))

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
if opt_optimizer == 'gaussian process regression (GP)':
    gp_iter = col_opt_3.number_input('GP iterations', min_value=20, value=1000, format='%i', step=100)
    opt_acq = col_opt_3.selectbox("GP acquisition function", ['shannon_ig_vec', 'ucb', 'variance', 'maximum'])

col_opt_5, col_opt_6 = st.columns([1, 1])
if col_opt_5.button('Start Optimization', disabled=(status != 'idle'), use_container_width=True):
    app_functions.run_pse()
if col_opt_5.button('Resume Optimization', disabled=(status != 'finished'), use_container_width=True):
    app_functions.resume_pse()
