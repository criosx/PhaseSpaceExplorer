"""Stripped-down PSE monitor page for service mode.

Connects to the GP server running on DEFAULT_PORT (no subprocess management,
no File System tab required).  Intended for use in process-compose where the
GP server is a separate service.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

DEFAULT_PORT = 5025

st.set_page_config(page_title="PSE Monitor", layout="wide")
st.title("Phase Space Explorer — Monitor")

# ── Connection settings ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Connection")
    port = st.number_input(
        "GP server port",
        min_value=1024, max_value=65535,
        value=DEFAULT_PORT,
        step=1,
        format="%d",
    )
    st.caption(f"Default port: {DEFAULT_PORT}")

    st.divider()
    st.header("Browse Archives")
    known_path = st.session_state.get("_monitor_storage_path")
    if known_path:
        archive_root = Path(known_path).parent
        active = Path(known_path)
        subdirs = sorted(p for p in archive_root.iterdir() if p.is_dir() and p != active)
        if subdirs:
            selected = st.selectbox(
                "Select directory to view",
                options=[None] + subdirs,
                format_func=lambda p: "— live campaign —" if p is None else p.name,
            )
            st.session_state["_browse_override"] = selected
        else:
            st.caption("No sibling directories found.")
            st.session_state["_browse_override"] = None
    else:
        st.caption("Connect to server first to populate archive list.")
        st.session_state["_browse_override"] = None

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(endpoint: str) -> requests.Response | None:
    try:
        return requests.get(f"http://127.0.0.1:{port}{endpoint}", timeout=3)
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None


def _fetch_info() -> dict | None:
    resp = _get("/get_info")
    if resp is not None and resp.ok:
        return resp.json()
    return None


# ── Server status ─────────────────────────────────────────────────────────────

@st.fragment(run_every=30)
def server_status():
    info = _fetch_info()
    col1, col2, col3 = st.columns(3)
    if info is None:
        col1.metric("Server", "unreachable")
        col2.metric("Service", "—")
        col3.metric("Status", "—")
        st.error(f"Cannot reach GP server on port {port}. Is it running?")
        return

    col1.metric("Server", f"port {port}")
    col2.metric("Service", "active" if info["has_service"] else "idle")
    col3.metric("Status", info["status"] or "—")

    storage_path = info.get("storage_path")
    if storage_path:
        st.info(f"Storage path: `{storage_path}`")
        st.session_state["_monitor_storage_path"] = storage_path
    else:
        last = st.session_state.get("_monitor_storage_path")
        if last:
            st.caption(f"No active campaign. Showing last known path: `{last}`")
        else:
            st.warning("No campaign active — storage path unknown.")


# ── Results ───────────────────────────────────────────────────────────────────

@st.fragment(run_every=30)
def results_panel():
    override = st.session_state.get("_browse_override")
    if override is not None:
        pse_dir = Path(override)
        st.info(f"Browsing archive: `{pse_dir}`")
    else:
        storage_path = st.session_state.get("_monitor_storage_path")
        if not storage_path:
            st.info("Waiting for an active campaign to show results.")
            return
        pse_dir = Path(storage_path)

    # In-flight iterations
    ci_path = pse_dir / "results" / "current_iterations.pkl"
    if ci_path.exists():
        with open(ci_path, "rb") as fh:
            df_ci = pd.DataFrame(pickle.load(fh))
        if not df_ci.empty:
            st.subheader("In-progress measurements")
            st.dataframe(df_ci, hide_index=True, use_container_width=True)

    # Finished results — gpCAM
    res_gpcam = pse_dir / "results" / "gpCAMstream.pkl"
    res_grid = pse_dir / "results" / "pse_grid_results.pkl"

    if res_gpcam.exists():
        with open(res_gpcam, "rb") as fh:
            df_res = pd.DataFrame(pickle.load(fh))
        st.subheader(f"Completed measurements ({len(df_res)})")
        st.dataframe(df_res, hide_index=False, use_container_width=True)

    elif res_grid.exists():
        with open(res_grid, "rb") as fh:
            grid = pickle.load(fh)
        idx = np.array(list(np.ndindex(*grid.shape)))
        df_grid = pd.DataFrame(idx, columns=[f"dim_{i}" for i in range(grid.ndim)])
        df_grid["result"] = grid.flatten()
        st.subheader("Grid results")
        st.dataframe(df_grid, hide_index=False, use_container_width=True)

    else:
        st.info("No results yet.")

    # Plots
    plot_dir = pse_dir / "plots"
    if plot_dir.is_dir():
        pngs = sorted(plot_dir.glob("*.png"))
        if pngs:
            st.subheader("Plots")
            for png in pngs:
                try:
                    st.image(str(png), use_container_width=True)
                except FileNotFoundError:
                    pass


# ── Layout ────────────────────────────────────────────────────────────────────

server_status()
st.divider()
results_panel()

if st.button("Refresh now"):
    st.rerun()
