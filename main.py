import os
import pandas
from pathlib import Path
import shutil
import streamlit as st

# check if all working directories exist
app_dir = os.path.join(os.path.expanduser('~'), 'app_data')
if not os.path.isdir(app_dir):
    os.mkdir(app_dir)

streamlit_dir = os.path.join(app_dir, 'streamlit_QCMD_phasespace')
if not os.path.isdir(streamlit_dir):
    os.mkdir(streamlit_dir)

app_functions_dir = os.path.join(str(Path(__file__).parent), 'support')

# save paths to persistent session state
st.session_state['streamlit_dir'] = streamlit_dir
st.session_state['app_functions_dir'] = app_functions_dir
st.session_state['active_project'] = None
st.session_state['user_qcmd_opt_dir'] = None

df_folders = pandas.DataFrame({
    'App home': [st.session_state['streamlit_dir']],
})

df_folders = df_folders.T
df_folders.columns = ['folder']

st.write("""
# QCMD Phase Space Exploration App
Welcome to the QCMD Phase Space Exploration App.

## The App File System
Like a mobile App, the SANS App has a limited file system. All data are stored in the following folders:

""")

df_folders

st.divider()

"""
*Contact: frank.heinrich@nist.gov*
"""

