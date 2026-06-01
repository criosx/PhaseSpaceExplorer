from contextlib import closing
import os
from pathlib import Path
import socket
import streamlit as st
import subprocess
import sys
import uuid

from pse import configuration

st.write("""
# QCMD Phase Space Exploration
Welcome to the QCMD Phase Space Exploration App.

## The App File System
The Phasespace Explorer uses the ROADMAP datamanager for file storage. See the File System tab.
""")

st.divider()

# first initialization
if 'first_intialization' not in st.session_state:
    st.session_state['first_intialization'] = True

    st.session_state["data_folders_ready"] = False
    st.session_state['pse_dir'] = None
    st.session_state.user_root_dir = Path.home() / "app_data"

    st.session_state.cfg = configuration.load_persistent_cfg()
    # initialize some widgets
    # force rerendering of toggle widgets
    st.session_state['rpse_key'] = str(uuid.uuid4())
    st.session_state['ppse_key'] = str(uuid.uuid4())
    st.session_state['update_counter'] = 0
    app_functions_dir = os.path.join(str(Path(__file__).parent), 'pse')
    st.session_state['app_functions_dir'] = app_functions_dir

    # get free server port
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        port = s.getsockname()[1]  # Return the port number assigned.

    st.session_state['gp_server_port'] = port
    st.session_state['gp_server_process'] = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from pse.gp_server import GpServer; "
                "GpServer().run(int(sys.argv[1]))"
            ),
            str(port),
        ],
        stdout=None,
        stderr=None,
    )

