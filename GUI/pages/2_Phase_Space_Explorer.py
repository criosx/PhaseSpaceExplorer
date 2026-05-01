import streamlit as st

from pse import streamlit_components
from pse import configuration

if not st.session_state["data_folders_ready"]:
    st.info("Files and Folders not set up. Please visit the File System tab.")
    st.stop()

streamlit_components.check_session_state()

st.write("""
# Job Monitor
""")
with (st.expander('Monitor')):
    streamlit_components.monitor()

st.write("""
# Setup
""")
with st.expander('Setup'):
    streamlit_components.clear_project_data_dialog()
    st.write("""
    ## Parameters
    ### Model Fit
    """)
    streamlit_components.parameter_input()

if 'opt_pars' in st.session_state:
    if not any(st.session_state['opt_pars']['optimize']):
        st.warning("Please, select at least on parameter to optimize before starting PSE.")
        st.stop()
    else:
        kwargs = {'exp_par': st.session_state['opt_pars']}
else:
    kwargs = {}

st.write("""
# Run Control
""")
streamlit_components.run_control(configuration=configuration, kwargs=kwargs)