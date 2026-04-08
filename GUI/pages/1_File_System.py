from pathlib import Path

import streamlit as st

from roadmap_datamanager.gui import streamlit_components as stc
from pse import configuration

cfg = st.session_state.cfg

# ----------------------- User dialog --------------------------------------
cfg= stc.UI_fragment_user(
    cfg=cfg,
    user_root_dir=st.session_state.user_root_dir,
    enable_user_selection=True
)
st.session_state.cfg = cfg
configuration.save_persistent_cfg(st.session_state.cfg)

dm_root = st.session_state.cfg.dm_root
if dm_root is None or not dm_root.is_dir():
    st.stop()

# ------------------ Project/Campaign/Experiment Diaolog -------------------
cfg, st.session_state.data_folders_ready, rerun = stc.UI_fragment_PCE(cfg)
st.session_state.cfg = cfg
configuration.save_persistent_cfg(st.session_state.cfg)
if rerun:
    st.rerun()
if not st.session_state.data_folders_ready:
    st.stop()

# -------------------- Storage Directory --------------------------------------
cfg, rerun = stc.UI_fragment_app_storage(
    cfg=cfg,
    storage_folders=['PSE'],
    gitignore_folders=['PSE'],
    special_action=None,
    special_action_arguments=None,
    special_action_label='',
    special_action_enabled=True
)
st.session_state.cfg = cfg
configuration.save_persistent_cfg(st.session_state.cfg)
cfg = st.session_state.cfg
exp_root = Path(cfg.dm_root).expanduser().resolve() / cfg.project / cfg.campaign / cfg.experiment
st.session_state["pse_dir"] = exp_root / 'PSE'
if rerun:
    st.rerun()

# --------------------- Datalad UI fragment --------------------------
cfg, dm = stc.UI_fragment_datalad(
    cfg=st.session_state.cfg
)
st.session_state.cfg = cfg
st.session_state.datamanager = dm
configuration.save_persistent_cfg(st.session_state.cfg)
if not st.session_state.cfg.use_datalad or dm is None:
    st.stop()

# ---------------------- GIN remote storage ----------------------------
cfg, rerun = stc.UI_fragment_GIN_actions(st.session_state.cfg, st.session_state.datamanager)
st.session_state.cfg = cfg
configuration.save_persistent_cfg(st.session_state.cfg)
if rerun:
    st.rerun()
