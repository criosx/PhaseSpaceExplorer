from contextlib import closing
import os
import pandas
from pathlib import Path
import socket
import streamlit as st
import subprocess

from pse import gp_server

# check if all working directories exist
app_dir = os.path.join(os.path.expanduser('~'), 'app_data')
if not os.path.isdir(app_dir):
    os.mkdir(app_dir)

streamlit_dir = os.path.join(app_dir, 'streamlit_QCMD_phasespace')
if not os.path.isdir(streamlit_dir):
    os.mkdir(streamlit_dir)

app_functions_dir = os.path.join(str(Path(__file__).parent), 'pse')

# save paths to persistent session state
st.session_state['streamlit_dir'] = streamlit_dir
st.session_state['app_functions_dir'] = app_functions_dir
st.session_state['active_project'] = None
st.session_state['user_qcmd_opt_dir'] = None

# get free server port
with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(('', 0))  # Bind to a free port provided by the host.
    port = s.getsockname()[1]  # Return the port number assigned.
st.session_state['gp_server_port'] = port
server_path = gp_server.__file__
subprocess.Popen(['python', server_path, str(port)])

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


