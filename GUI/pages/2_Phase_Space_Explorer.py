import streamlit as st

from pse import streamlit_components
from pse import configuration

streamlit_components.start_of_script_business()

st.write("""
# Job Monitor
""")
with (st.expander('Monitor')):
    streamlit_components.monitor()

st.write("""
# Setup
""")
with st.expander('Setup'):
    st.write("""
    ## PSE Directory
    """)
    streamlit_components.pse_directory()
    st.write("""
    ## Parameters
    ### Model Fit
    """)
    streamlit_components.parameter_input()
    # TODO: Revisit restart logic. Reset of configuration_loaded flag necessary here to make any changes to the input permanent. What about run_control?
    st.session_state.configuration_reloaded = False

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

streamlit_components.end_of_script_business()