"""
Streamlit app for searching Phases of Play and Player Runs feeds.

Run with:
    streamlit run streamlit_phases_xml.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import COMPETITION_DIRS, LOGOS_DIR
from src.vod import get_vod_api_key
from src.ui import render_header, LOGO_LIGHT
from src.sidebar import sidebar_upload_mode, sidebar_local_mode
from src.tab_runs import analysis_runs_by_phase
from src.tab_phases import analysis_phase_analysis
from src.tab_blocks import analysis_block_analysis


def main():
    st.set_page_config(
        page_title="Phases of Play – Feed Analysis",
        layout="wide",
        page_icon=str(LOGO_LIGHT) if LOGO_LIGHT.exists() else "⚽",
    )
    render_header()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 Load Data")

        if not get_vod_api_key():
            st.text_input(
                "🔑 VOD API Key",
                type="password",
                key="_vod_k",
                help="Required for video playback. Set the VOD_API_KEY environment variable or Streamlit secret to avoid entering it here.",
            )

        _has_local_feeds = bool(COMPETITION_DIRS)
        if _has_local_feeds:
            load_mode = st.radio(
                "Data source",
                ["📤 Upload files", "📁 Local feeds directory"],
                horizontal=False,
                key="load_mode",
            )
        else:
            load_mode = "📤 Upload files"

        st.markdown("---")

        if load_mode == "📤 Upload files":
            sidebar_upload_mode()
        else:
            sidebar_local_mode()

    # ── Main content ──────────────────────────────────────────────────────
    if "phases_df" not in st.session_state or st.session_state["phases_df"] is None:
        st.info("👈 Upload your feed files (or select from local directory) in the sidebar and click **Load** to begin.")
        st.stop()

    phases_df: pd.DataFrame = st.session_state["phases_df"]
    runs_df: pd.DataFrame = st.session_state["runs_df"]
    match_info: dict = st.session_state["match_info"]
    squad_map: dict[str, str] = st.session_state.get("squad_map", {})
    jersey_map: dict[str, str] = st.session_state.get("jersey_map", {})

    tab_runs, tab_analysis, tab_block = st.tabs(
        ["Runs Search", "Phase Analysis", "Block Analysis"]
    )

    with tab_runs:
        if runs_df is None or runs_df.empty:
            st.info("No run data available for the loaded game(s).")
        else:
            analysis_runs_by_phase(phases_df, runs_df, match_info, squad_map, jersey_map)

    with tab_analysis:
        analysis_phase_analysis(phases_df, match_info, squad_map, jersey_map)

    with tab_block:
        analysis_block_analysis(phases_df, match_info)


if __name__ == "__main__":
    main()

