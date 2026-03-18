"""Runs Search tab — cached computation + UI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from src.vod import get_vod_api_key, get_vod_streaming
from src.pitch import render_runs_pitch_map, pitch_zone_selector
from src.utils import ms_to_mmss
from src.config import BRAND_AMBER


@st.cache_data(show_spinner="Searching runs…", max_entries=3)
def compute_runs_result(
    phases_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    contestant_map: dict,
    available_labels: tuple,
    selected_labels: tuple,
    includes_shots_choice: str,
    includes_goal_choice: str,
    selected_run_labels: tuple,
    run_type_choice: str,
    dlb_choice: str,
    dangerous_choice: str,
    followed_by_shot_choice: str,
    followed_by_goal_choice: str,
    speed_lo: float,
    speed_max_hi: float,
    et_lo: float,
    et_hi: float,
    run_coord_bounds: tuple | None,
    selected_team_name: str,
    selected_player_name: str,
    squad_map: tuple,
) -> pd.DataFrame:
    """Heavy phase-run interval join + all filters.  Cached."""
    squad_map_dict: dict = dict(squad_map)
    coord_bounds: dict | None = dict(run_coord_bounds) if run_coord_bounds is not None else None
    et_col = "expectedThreat_max"
    _et_same = (
        not (et_col in runs_df.columns and runs_df[et_col].notna().any())
        or runs_df[et_col].min(skipna=True) == runs_df[et_col].max(skipna=True)
    )

    _phase_label_filter = list(selected_labels) if selected_labels else list(available_labels)
    filtered_phases = phases_df[phases_df["phaseLabel"].isin(_phase_label_filter)]
    if includes_shots_choice != "Any" and "includesShots" in filtered_phases.columns:
        filtered_phases = filtered_phases[filtered_phases["includesShots"] == includes_shots_choice]
    if includes_goal_choice != "Any" and "includesGoal" in filtered_phases.columns:
        filtered_phases = filtered_phases[filtered_phases["includesGoal"] == includes_goal_choice]
    if filtered_phases.empty:
        return pd.DataFrame()

    # Build slim phase lookup — only the columns we need, one copy
    ph_cols_src = ["phase_id", "periodId", "possessionContestantId",
                   "startFrame", "endFrame", "phaseLabel", "startTime"]
    if "game_id" in filtered_phases.columns:
        ph_cols_src.insert(0, "game_id")
    for _c in ("includesShots", "includesGoal"):
        if _c in filtered_phases.columns:
            ph_cols_src.append(_c)
    ph_cols_src = [c for c in ph_cols_src if c in filtered_phases.columns]
    ph_slim = filtered_phases[ph_cols_src].copy()
    ph_slim["_gid"]    = ph_slim["game_id"].astype(str) if "game_id" in ph_slim.columns else ""
    ph_slim["_pid"]    = ph_slim["phase_id"].astype(str)
    ph_slim["_period"] = ph_slim["periodId"].astype(str)
    ph_slim["_team"]   = ph_slim["possessionContestantId"].astype(str)
    ph_slim["_psf"]    = ph_slim["startFrame"].astype(int)
    ph_slim["_pef"]    = ph_slim["endFrame"].astype(int)
    for _c in ("includesShots", "includesGoal"):
        if _c not in ph_slim.columns:
            ph_slim[_c] = ""
    ph_slim = ph_slim.rename(columns={"startTime": "phase_startTime"})
    ph_keep = ["_gid", "_pid", "_period", "_team", "_psf", "_pef",
               "phaseLabel", "includesShots", "includesGoal", "phase_startTime"]
    ph_slim = ph_slim[[c for c in ph_keep if c in ph_slim.columns]]

    # Build slim run lookup — select needed columns first, then copy once
    # NOTE: composite_run_id is NOT included here; it is captured via _crid
    # which is later renamed back. Including it would create a duplicate column.
    run_keep_cols = ["run_id", "game_id", "periodId", "playerId", "contestantId",
                     "masterLabel", "runType", "defensiveLineBroken",
                     "dangerous", "expectedThreat_max", "speed_max",
                     "runFollowedByTeamShot", "runFollowedByTeamGoal",
                     "startTime", "endTime", "startFrame", "endFrame",
                     "startX", "startY", "endX", "endY"]
    run_keep_cols = [c for c in run_keep_cols if c in runs_df.columns]
    ru_slim = runs_df[run_keep_cols].copy()
    ru_slim["_gid"]    = ru_slim["game_id"].astype(str) if "game_id" in ru_slim.columns else ""
    ru_slim["_period"] = ru_slim["periodId"].astype(str)
    ru_slim["_team"]   = ru_slim["contestantId"].astype(str)
    ru_slim["_rsf"]    = ru_slim["startFrame"]
    ru_slim["_ref"]    = ru_slim["endFrame"]
    ru_slim = ru_slim.dropna(subset=["_rsf", "_ref"])
    ru_slim["_rsf"]    = ru_slim["_rsf"].astype(int)
    ru_slim["_ref"]    = ru_slim["_ref"].astype(int)
    ru_slim["_crid"]   = (runs_df.loc[ru_slim.index, "composite_run_id"] if "composite_run_id" in runs_df.columns else ru_slim["run_id"]).astype(str)

    if "runType" in ru_slim.columns:
        _is_inp = ru_slim["runType"] == "inPossession"
        _is_oop = ru_slim["runType"] == "outOfPossession"
        ru_inp   = ru_slim[_is_inp]
        ru_oop   = ru_slim[_is_oop]
        ru_other = ru_slim[~(_is_inp | _is_oop)]
    else:
        ru_inp   = ru_slim
        ru_oop   = ru_slim.iloc[0:0]
        ru_other = ru_slim.iloc[0:0]

    merged_inp = ru_inp.merge(ph_slim, left_on=["_gid", "_period", "_team"], right_on=["_gid", "_period", "_team"], how="inner")

    if not ru_oop.empty:
        merged_oop = ru_oop.merge(ph_slim.rename(columns={"_team": "_ph_team"}), left_on=["_gid", "_period"], right_on=["_gid", "_period"], how="inner")
        merged_oop = merged_oop[merged_oop["_team"] != merged_oop["_ph_team"]]
        merged_oop = merged_oop.drop(columns=["_ph_team"], errors="ignore")
    else:
        merged_oop = pd.DataFrame()

    if not ru_other.empty:
        merged_other = ru_other.merge(ph_slim, left_on=["_gid", "_period", "_team"], right_on=["_gid", "_period", "_team"], how="inner")
    else:
        merged_other = pd.DataFrame()

    # Only concat non-empty frames — concatenating empty DataFrames with category
    # dtype columns causes pandas to produce corrupted dtypes (multi-dimensional
    # groupers) that break groupby even when the column being grouped is object.
    frames_to_concat = [f for f in [merged_inp, merged_oop, merged_other] if not f.empty]
    if not frames_to_concat:
        return pd.DataFrame()
    merged = pd.concat(frames_to_concat, ignore_index=True)
    if not merged.empty:
        overlap = (merged["_rsf"] < merged["_pef"]) & (merged["_ref"] > merged["_psf"])
        merged = merged[overlap]

    result_df = merged.rename(columns={"_crid": "composite_run_id", "_pid": "phase_id", "_rsf": "run_startFrame", "_ref": "run_endFrame"}).drop(columns=["_gid", "_period", "_team", "_psf", "_pef"], errors="ignore")

    for _id_col in ("run_id", "phase_id"):
        if _id_col in result_df.columns:
            result_df[_id_col] = pd.to_numeric(result_df[_id_col], errors="coerce").astype("Int64")

    if result_df.empty:
        return result_df

    if contestant_map and "contestantId" in result_df.columns:
        result_df["team_name"] = result_df["contestantId"].map(contestant_map)

    has_et = et_col in result_df.columns and result_df[et_col].notna().any()
    if selected_run_labels:
        result_df = result_df[result_df["masterLabel"].isin(selected_run_labels)]
    if run_type_choice != "Any":
        result_df = result_df[result_df["runType"] == run_type_choice]
    if dlb_choice == "Yes":
        result_df = result_df[result_df["defensiveLineBroken"] == 1.0]
    elif dlb_choice == "No":
        result_df = result_df[result_df["defensiveLineBroken"] == 0.0]
    if dangerous_choice == "Yes":
        result_df = result_df[result_df["dangerous"] == 1.0]
    elif dangerous_choice == "No":
        result_df = result_df[result_df["dangerous"] == 0.0]
    if followed_by_shot_choice == "Yes" and "runFollowedByTeamShot" in result_df.columns:
        result_df = result_df[result_df["runFollowedByTeamShot"] == 1.0]
    elif followed_by_shot_choice == "No" and "runFollowedByTeamShot" in result_df.columns:
        result_df = result_df[result_df["runFollowedByTeamShot"] == 0.0]
    if followed_by_goal_choice == "Yes" and "runFollowedByTeamGoal" in result_df.columns:
        result_df = result_df[result_df["runFollowedByTeamGoal"] == 1.0]
    elif followed_by_goal_choice == "No" and "runFollowedByTeamGoal" in result_df.columns:
        result_df = result_df[result_df["runFollowedByTeamGoal"] == 0.0]
    if speed_max_hi is not None and "speed_max" in result_df.columns:
        result_df = result_df[result_df["speed_max"].isna() | ((result_df["speed_max"] >= speed_lo) & (result_df["speed_max"] <= speed_max_hi))]
    if has_et and not _et_same:
        result_df = result_df[result_df[et_col].isna() | ((result_df[et_col] >= et_lo) & (result_df[et_col] <= et_hi))]

    if coord_bounds:
        for coord, b_min, b_max in [
            ("startX", coord_bounds["start_x_min"], coord_bounds["start_x_max"]),
            ("startY", coord_bounds["start_y_min"], coord_bounds["start_y_max"]),
            ("endX",   coord_bounds.get("end_x_min", 0),   coord_bounds.get("end_x_max", 100)),
            ("endY",   coord_bounds.get("end_y_min", 0),   coord_bounds.get("end_y_max", 100)),
        ]:
            if coord in result_df.columns and (b_min != 0 or b_max != 100):
                result_df = result_df[result_df[coord].notna() & (result_df[coord] >= b_min) & (result_df[coord] <= b_max)]

    if selected_team_name != "All teams":
        _cid_by_name = {contestant_map.get(str(c), str(c)): str(c) for c in (result_df["contestantId"].dropna().unique() if "contestantId" in result_df.columns else [])}
        selected_cid = _cid_by_name.get(selected_team_name, "")
        result_df = result_df[result_df["contestantId"] == selected_cid]
        if selected_player_name != "All players":
            if squad_map_dict:
                matching_pids = {pid for pid in result_df["playerId"].dropna().unique() if squad_map_dict.get(str(pid), str(pid)) == selected_player_name}
            else:
                matching_pids = {selected_player_name}
            result_df = result_df[result_df["playerId"].isin(matching_pids)]

    if not result_df.empty:
        # Category columns with unused levels (left over from merge / filter)
        # cause pandas groupby to raise "Grouper not 1-dimensional".
        # Convert every category column back to plain object before grouping.
        _cat_cols = [c for c, dt in result_df.dtypes.items() if dt.name == "category"]
        if _cat_cols:
            result_df = result_df.copy()
            for _cc in _cat_cols:
                result_df[_cc] = result_df[_cc].astype(object)

        phase_cols_set = {"phase_id", "phaseLabel", "phase_startTime", "includesShots", "includesGoal"}
        def _join_unique(x):
            return ", ".join(sorted({str(v) for v in x if pd.notna(v) and str(v) != "nan"}))
        agg_rules = {c: (_join_unique if c in phase_cols_set else "first") for c in result_df.columns if c != "composite_run_id"}
        result_df = result_df.groupby("composite_run_id", as_index=False).agg(agg_rules)
        if contestant_map and "contestantId" in result_df.columns:
            result_df["team_name"] = result_df["contestantId"].map(contestant_map)

    if squad_map_dict and "playerId" in result_df.columns:
        result_df["player_name"] = result_df["playerId"].map(lambda x: squad_map_dict.get(str(x), str(x)) if pd.notna(x) else "")

    return result_df


@st.fragment
def analysis_runs_by_phase(phases_df: pd.DataFrame, runs_df: pd.DataFrame, match_info: dict, squad_map: dict | None = None, jersey_map: dict | None = None):
    """List all runs that occur during phases matching phaseSummary and run-level criteria."""
    if squad_map is None:
        squad_map = {}
    if jersey_map is None:
        jersey_map = {}
    st.subheader("🏃 Runs Search")
    st.markdown("Use the **Phase Criteria** expander to choose which phases to search within, then open **Run Criteria** to refine by individual run properties.")

    if "startFrame" not in phases_df.columns or "endFrame" not in phases_df.columns:
        st.error("Phase feed is missing startFrame / endFrame – cannot link to runs.")
        return
    if "startFrame" not in runs_df.columns or "endFrame" not in runs_df.columns:
        st.error("Run feed is missing startFrame / endFrame – cannot link to phases.")
        return

    available_labels = sorted(phases_df["phaseLabel"].unique())
    _run_cmap = match_info.get("contestant_map", {})
    _all_run_cids = sorted(runs_df["contestantId"].dropna().unique()) if "contestantId" in runs_df.columns else []
    _run_cid_by_name = {_run_cmap.get(str(c), str(c)): str(c) for c in _all_run_cids}
    _available_run_team_names = sorted(_run_cid_by_name.keys())
    _all_run_master_labels = sorted(runs_df["masterLabel"].dropna().unique()) if "masterLabel" in runs_df.columns else []
    et_col = "expectedThreat_max"
    _has_et_src = et_col in runs_df.columns and runs_df[et_col].notna().any()
    if _has_et_src:
        _et_min_src = float(runs_df[et_col].min(skipna=True))
        _et_max_src = float(runs_df[et_col].max(skipna=True))
        _et_same = _et_min_src == _et_max_src
    else:
        _et_min_src = _et_max_src = 0.0
        _et_same = True

    with st.expander("🔍 Filters", expanded=True):
        ftab_run, ftab_phase, ftab_coords, ftab_outcomes, ftab_team = st.tabs(
            ["🏃 Run Criteria", "📋 Phase Criteria", "📍 Run Coordinates", "🎯 Attacking Outcomes", "👥 Team / Player"]
        )

        with ftab_phase:
            ph_col1, ph_col2 = st.columns(2)
            with ph_col1:
                _, btn_col1, btn_col2 = st.columns([4, 1, 1])
                with btn_col1:
                    if st.button("Select all", key="rp_select_all"):
                        st.session_state["run_phase_labels"] = available_labels
                with btn_col2:
                    if st.button("Clear", key="rp_clear"):
                        st.session_state["run_phase_labels"] = []
                selected_labels = st.multiselect("Phase labels", available_labels, default=available_labels, key="run_phase_labels")
                includes_shots_choice = st.radio("**Includes shots**", ["Any", "True", "False"], horizontal=True, key="rp_includes_shots")
            with ph_col2:
                includes_goal_choice = st.radio("**Includes goal**", ["Any", "True", "False"], horizontal=True, key="rp_includes_goal")

        with ftab_run:
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                selected_run_labels = st.multiselect("Main label", _all_run_master_labels, default=[], key="run_master_labels")
            if _has_et_src and not _et_same:
                with r1c2:
                    _et_sl = st.slider("Expected threat", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.001, format="%.3f", key="run_et_range")
                    et_lo, et_hi = float(_et_sl[0]), float(_et_sl[1])
            else:
                et_lo = _et_min_src
                et_hi = _et_max_src
            with r1c3:
                _has_speed = "speed_max" in runs_df.columns and runs_df["speed_max"].notna().any()
                if _has_speed:
                    _speed_data_max = float(runs_df["speed_max"].max(skipna=True))
                    _speed_sl_max = round(_speed_data_max + 0.5, 1)
                    _speed_sl = st.slider("Max speed (m/s)", min_value=0.0, max_value=_speed_sl_max, value=(0.0, _speed_sl_max), step=0.1, format="%.1f", key="run_speed_range")
                    speed_lo, speed_max_hi = float(_speed_sl[0]), float(_speed_sl[1])
                else:
                    speed_lo = 0.0
                    speed_max_hi = None
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                run_type_choice = st.radio("**Run type**", ["Any", "inPossession", "outOfPossession"], horizontal=True, key="run_type_filter")
            with r2c2:
                dlb_choice = st.radio("**Defensive line broken**", ["Any", "Yes", "No"], horizontal=True, key="run_dlb")
            with r2c3:
                dangerous_choice = st.radio("**Dangerous**", ["Any", "Yes", "No"], horizontal=True, key="run_dangerous")

        with ftab_coords:
            run_coord_bounds = pitch_zone_selector(key_prefix="run_coords", has_start="startX" in runs_df.columns, has_end="endX" in runs_df.columns)

        with ftab_team:
            tf_col1, tf_col2 = st.columns(2)
            with tf_col1:
                selected_team_name = st.selectbox("Filter by team", options=["All teams"] + _available_run_team_names, key="run_team_filter")
            with tf_col2:
                selected_player_name = "All players"
                if selected_team_name != "All teams":
                    _selected_cid_pre = _run_cid_by_name.get(selected_team_name, "")
                    _team_run_pids = sorted(runs_df.loc[runs_df["contestantId"] == _selected_cid_pre, "playerId"].dropna().unique()) if "contestantId" in runs_df.columns else []
                    if squad_map:
                        _player_options_pre = sorted({squad_map.get(str(p), str(p)) for p in _team_run_pids})
                    else:
                        _player_options_pre = sorted(str(p) for p in _team_run_pids)
                    selected_player_name = st.selectbox("Filter by player", options=["All players"] + _player_options_pre, key="run_player_filter")

        with ftab_outcomes:
            st.caption("These filters apply to **inPossession** runs only.")
            ao_c1, ao_c2 = st.columns(2)
            with ao_c1:
                followed_by_shot_choice = st.radio("**Followed by Team Shot**", ["Any", "Yes", "No"], horizontal=True, key="run_followed_shot")
            with ao_c2:
                followed_by_goal_choice = st.radio("**Followed by Team Goal**", ["Any", "Yes", "No"], horizontal=True, key="run_followed_goal")

    if st.button("▶ Generate Outputs", type="primary", key="runs_generate"):
        st.session_state["runs_committed"] = {
            "selected_labels": selected_labels, "includes_shots": includes_shots_choice,
            "includes_goal": includes_goal_choice, "selected_run_labels": selected_run_labels,
            "run_type_choice": run_type_choice, "dlb_choice": dlb_choice,
            "dangerous_choice": dangerous_choice, "followed_by_shot": followed_by_shot_choice,
            "followed_by_goal": followed_by_goal_choice, "speed_lo": speed_lo,
            "speed_max_hi": speed_max_hi, "et_lo": et_lo, "et_hi": et_hi,
            "run_coord_bounds": run_coord_bounds, "selected_team_name": selected_team_name,
            "selected_player_name": selected_player_name,
        }

    committed = st.session_state.get("runs_committed")
    if not committed:
        st.info("Set your filters above and click **▶ Generate Outputs** to run the analysis.")
        return

    _coord_bounds_hashable = tuple(sorted(committed["run_coord_bounds"].items())) if committed["run_coord_bounds"] else None
    result_df = compute_runs_result(
        phases_df=phases_df, runs_df=runs_df, contestant_map=_run_cmap,
        available_labels=tuple(available_labels), selected_labels=tuple(committed["selected_labels"]),
        includes_shots_choice=committed["includes_shots"], includes_goal_choice=committed["includes_goal"],
        selected_run_labels=tuple(committed["selected_run_labels"]),
        run_type_choice=committed["run_type_choice"], dlb_choice=committed["dlb_choice"],
        dangerous_choice=committed["dangerous_choice"],
        followed_by_shot_choice=committed.get("followed_by_shot", "Any"),
        followed_by_goal_choice=committed.get("followed_by_goal", "Any"),
        speed_lo=float(committed.get("speed_lo", 0.0)), speed_max_hi=committed.get("speed_max_hi", None),
        et_lo=float(committed["et_lo"]), et_hi=float(committed["et_hi"]),
        run_coord_bounds=_coord_bounds_hashable,
        selected_team_name=committed["selected_team_name"],
        selected_player_name=committed["selected_player_name"],
        squad_map=tuple(sorted(squad_map.items())),
    )

    if result_df.empty:
        if len(result_df.columns) == 0:
            st.warning("No phases match the selected phase criteria.")
        else:
            st.warning("No runs found overlapping with the selected phase criteria.")
        return

    unique_phases_count = result_df["phase_id"].nunique() if "phase_id" in result_df.columns else 0
    st.markdown("---")
    st.markdown(f"**{len(result_df)}** unique run(s) found across **{unique_phases_count}** matching phase(s).")

    c_view, c_group = st.columns(2)
    with c_view:
        view_mode = st.radio("**View mode**", ["Individual runs", "Aggregated", "Pitch map"], horizontal=True, key="run_view_mode")
    with c_group:
        group_by = st.radio("**Aggregate by**", ["Team", "Player"], horizontal=True, key="run_group_by", disabled=(view_mode != "Aggregated"))

    if view_mode == "Individual runs":
        display_cols = ["game_id", "run_id", "phase_id", "periodId", "startTime", "endTime", "masterLabel", "phaseLabel"]
        if "team_name" in result_df.columns:
            display_cols.append("team_name")
        if "player_name" in result_df.columns:
            display_cols.append("player_name")
        elif "playerId" in result_df.columns:
            display_cols.append("playerId")
        display_cols = [c for c in display_cols if c in result_df.columns]

        if jersey_map and "playerId" in result_df.columns:
            result_df = result_df.copy()
            result_df["jersey_number"] = result_df["playerId"].map(lambda x: jersey_map.get(str(x), "") if pd.notna(x) else "")
            display_cols.append("jersey_number")

        ind_display_df = result_df[display_cols].reset_index(drop=True).copy()
        for _tc in ("startTime", "endTime"):
            if _tc in ind_display_df.columns:
                ind_display_df[_tc] = ind_display_df[_tc].apply(ms_to_mmss)

        st.caption("👆 Click a row to select it, then press **▶ Play Video** below.")
        run_table_sel = st.dataframe(ind_display_df, use_container_width=True, height=min(600, max(200, len(result_df) * 38)), selection_mode="single-row", on_select="rerun", key="run_ind_table")
        _sel_run_rows = run_table_sel.selection.get("rows", []) if run_table_sel and run_table_sel.selection else []
        run_selected_idx = int(_sel_run_rows[0]) if _sel_run_rows else None

        st.markdown("##### 🎬 Video Playback")
        if run_selected_idx is None:
            st.info("Click a row in the table above to select a run, then press ▶ Play Video.")
        else:
            _sel_run = result_df.iloc[run_selected_idx]
            st.success(f"Selected: run **{_sel_run.get('run_id', '')}** — **{_sel_run.get('team_name', _sel_run.get('contestantId', ''))}** — {_sel_run.get('player_name', _sel_run.get('playerId', ''))} — Period {_sel_run.get('periodId', '')} — {_sel_run.get('masterLabel', '')}")

        vid_c1, vid_c2 = st.columns(2)
        with vid_c1:
            run_before_buf = st.number_input("Pre-buffer (s)", min_value=0, max_value=10, value=5, step=1, key="run_vid_pre")
        with vid_c2:
            run_after_buf = st.number_input("Post-buffer (s)", min_value=0, max_value=10, value=5, step=1, key="run_vid_post")

        if st.button("▶ Play Video", key="run_play_video", type="primary", disabled=run_selected_idx is None):
            _row = result_df.iloc[run_selected_idx]
            _game_id = str(_row.get("game_id", ""))
            _period_id = int(_row.get("periodId", 1))
            _start_ms = _row.get("startTime")
            _end_ms   = _row.get("endTime")
            if pd.notna(_start_ms) and pd.notna(_end_ms):
                _time_in  = int(float(_start_ms) / 1000)
                _time_out = int(float(_end_ms) / 1000)
            elif pd.notna(_start_ms):
                _time_in  = int(float(_start_ms) / 1000)
                _time_out = _time_in
            else:
                _sf = _row.get("run_startFrame") or _row.get("startFrame", 0)
                _ef = _row.get("run_endFrame")   or _row.get("endFrame", _sf)
                _time_in  = int(float(_sf) / 25)
                _time_out = int(float(_ef) / 25)

            if not _game_id:
                st.error("No game_id available for this run.")
            else:
                _api_key = get_vod_api_key()
                if not _api_key:
                    st.error("VOD API key is not configured. Set the `VOD_API_KEY` environment variable or enter it in the sidebar.")
                else:
                    with st.spinner("Fetching video clip…"):
                        try:
                            _url = get_vod_streaming(game_uuid=_game_id, period=_period_id, time_in=_time_in, time_out=_time_out, before_time=int(run_before_buf), after_time=int(run_after_buf), api_key=_api_key)
                            components.html(f'<iframe src="{_url}" width="950" height="600" frameborder="0" allowfullscreen></iframe>', height=620)
                        except requests.exceptions.HTTPError as exc:
                            st.error(f"VOD API request failed: {exc.response.status_code} {exc.response.reason}")
                        except requests.exceptions.RequestException:
                            st.error("Network error fetching video clip. Please try again.")
                        except (ValueError, KeyError, IndexError) as exc:
                            st.error(f"Could not retrieve video: {exc}")
                        except Exception:
                            st.error("Unexpected error fetching video. Please try again.")

    elif view_mode == "Pitch map":
        render_runs_pitch_map(result_df, match_info, squad_map)

    else:
        if group_by == "Team":
            group_cols = ["team_name"] if "team_name" in result_df.columns else ["contestantId"]
            label_col = group_cols[0]; id_col = None
        else:
            group_cols = ["playerId", "team_name"] if "team_name" in result_df.columns else ["playerId"]
            label_col = "playerId"; id_col = "playerId"

        agg_df = result_df.groupby(group_cols, dropna=False).agg(total_runs=("run_id", "count")).sort_values("total_runs", ascending=False).reset_index()
        agg_df = agg_df[agg_df[label_col].notna() & (agg_df[label_col].astype(str).str.strip() != "") & (agg_df[label_col].astype(str) != "None")]

        if id_col and squad_map:
            agg_df["_display_label"] = agg_df[id_col].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else str(x))
            display_label_col = "_display_label"
        else:
            display_label_col = label_col

        max_n = len(agg_df)
        top_n = st.slider("Top N to display", min_value=1, max_value=max(max_n, 2), value=min(10, max_n), step=1, key="run_agg_top_n") if max_n > 1 else max_n
        display_agg = agg_df.head(top_n).copy()
        agg_display = st.radio("**Display as**", ["Table", "Bar chart"], horizontal=True, key="run_agg_display")

        if agg_display == "Table":
            show_cols = [display_label_col, "total_runs"]
            if "team_name" in display_agg.columns and group_by == "Player":
                show_cols = [display_label_col, "team_name", "total_runs"]
            show_cols = [c for c in show_cols if c in display_agg.columns]
            rename_map = {display_label_col: "Player" if group_by == "Player" else "Team", "team_name": "Team", "total_runs": "Total Runs"}
            st.dataframe(display_agg[show_cols].rename(columns=rename_map).reset_index(drop=True), use_container_width=True, height=min(600, max(120, len(display_agg) * 38 + 40)))
        else:
            plot_df_agg = display_agg.iloc[::-1].reset_index(drop=True)
            fig = px.bar(
                plot_df_agg, x="total_runs", y=display_label_col, orientation="h",
                text="total_runs",
                labels={"total_runs": "Total Runs", display_label_col: group_by},
                height=max(300, len(plot_df_agg) * 36 + 80),
                color_discrete_sequence=[BRAND_AMBER],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                margin={"l": 10, "r": 40, "t": 30, "b": 10},
                yaxis={"automargin": True, "color": "#f0f0f0", "gridcolor": "#222"},
                xaxis={"color": "#f0f0f0", "gridcolor": "#222"},
                plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
                font={"color": "#f0f0f0", "family": "Barlow"},
                hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": BRAND_AMBER,
                            "font": {"size": 12, "color": "#f0f0f0", "family": "Barlow"}},
            )
            st.plotly_chart(fig, use_container_width=True, key="runs_agg_bar")

