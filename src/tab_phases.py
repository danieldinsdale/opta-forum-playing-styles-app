"""Phase Analysis tab — filters + phase list + aggregation."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from src.config import BRAND_AMBER, BRAND_RED
from src.vod import get_vod_api_key, get_vod_streaming
from src.pitch import opta_pitch_shapes, pitch_zone_selector
from src.utils import ms_to_mmss


@st.fragment
def analysis_phase_analysis(phases_df: pd.DataFrame, match_info: dict, squad_map: dict[str, str] | None = None, jersey_map: dict[str, str] | None = None):
    """Merged Phase Search + Phase Analysis: shared filters, two result views."""
    if squad_map is None:
        squad_map = {}
    if jersey_map is None:
        jersey_map = {}
    st.subheader("📊 Phase Analysis")
    st.markdown(
        "Open **Filters** to narrow phases by criteria, "
        "then switch between the **Phase List** and **Aggregation** tabs below."
    )

    available_labels = sorted(phases_df["phaseLabel"].unique())

    # Pre-compute team options from the full dataset
    pa_cmap = match_info.get("contestant_map", {}) if match_info else {}
    _all_cids = sorted(phases_df["possessionContestantId"].dropna().unique()) if "possessionContestantId" in phases_df.columns else []
    _pa_cid_by_name = {pa_cmap.get(str(c), str(c)): str(c) for c in _all_cids}
    _available_pa_team_names = sorted(_pa_cid_by_name.keys())

    with st.expander("🔍 Filters", expanded=True):
        pa_ftab_labels, pa_ftab_coords, pa_ftab_counts, pa_ftab_outcomes, pa_ftab_compact, pa_ftab_team = st.tabs(
            ["🏷️ Phase Labels", "📍 Coordinates", "⚡ Action Counts", "🎯 Attacking Outcomes", "📏 Compactness", "👥 Team / Player"]
        )

        with pa_ftab_labels:
            pa_label_mode = st.radio(
                "Selection mode",
                ["Any of (multi-select)", "Leads to (sequence)"],
                horizontal=True, key="pa_label_mode",
            )

            if pa_label_mode == "Any of (multi-select)":
                _, btn_all, btn_clr = st.columns([6, 1, 1])
                with btn_all:
                    if st.button("Select all", key="pa_select_all"):
                        st.session_state["pa_labels"] = available_labels
                with btn_clr:
                    if st.button("Clear", key="pa_clear"):
                        st.session_state["pa_labels"] = []
                pa_selected_labels = st.multiselect("Phase labels", available_labels, default=[], key="pa_labels")
                pa_seq_first = None
                pa_seq_leads_to = None
            else:
                seq_c1, seq_c2 = st.columns(2)
                with seq_c1:
                    pa_seq_first = st.selectbox("Phase label", options=["(select)"] + available_labels, key="pa_seq_first")
                with seq_c2:
                    pa_seq_leads_to = st.multiselect("Leads to", options=available_labels, default=[], key="pa_seq_leads_to")
                pa_selected_labels = []

        with pa_ftab_coords:
            if pa_label_mode == "Leads to (sequence)":
                st.markdown(
                    f'<small><span style="color:{BRAND_AMBER};">&#9632;</span> <b>Start zone</b> filters on the <b>start</b> of the 1st phase. &nbsp; '
                    f'<span style="color:{BRAND_RED};">&#9632;</span> <b>End zone</b> filters on the <b>end</b> of the 2nd (Leads to) phase.</small>',
                    unsafe_allow_html=True,
                )
            pa_coord_bounds = pitch_zone_selector(
                key_prefix="pa_coords",
                has_start="startX" in phases_df.columns,
                has_end="endX" in phases_df.columns,
            )

        with pa_ftab_counts:
            if pa_label_mode == "Leads to (sequence)":
                st.caption("In sequence mode, filters apply to the **sum** of counts across both phases.")
            count_cols_pa = [
                ("Number of Dangerous Runs",            "numberDangerousRuns",          "pa_dr"),
                ("Number of Line-Breaking Actions",     "numberLineBreakingActions",     "pa_lba"),
                ("Number of Passes",                    "numberPasses",                  "pa_np"),
                ("Number of High Pressure on Receiver", "numberHighPressureOnReceiver",  "pa_hpr"),
                ("Number of High Pressure on Touches",  "numberHighPressureOnTouches",   "pa_hpt"),
            ]
            pa_ac_range_inputs: dict[str, tuple[float, float]] = {}
            # Filter to only the columns that exist and have a range
            active_count_cols = [
                (lbl, col, key) for lbl, col, key in count_cols_pa
                if col in phases_df.columns and int(phases_df[col].min()) != int(phases_df[col].max())
            ]
            ac_cols = st.columns(3)
            for i, (lbl, col, key) in enumerate(active_count_cols):
                col_min = int(phases_df[col].min())
                col_max = int(phases_df[col].max())
                with ac_cols[i % 3]:
                    selected = st.slider(
                        lbl,
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        step=1,
                        key=f"{key}_slider",
                    )
                    pa_ac_range_inputs[col] = (float(selected[0]), float(selected[1]))

        with pa_ftab_outcomes:
            if pa_label_mode == "Leads to (sequence)":
                st.caption("In sequence mode **True** = at least one of the 2 phases satisfies the condition. **False** = neither phase satisfies it.")
            pa_bool_choices: dict[str, str] = {}
            ao_left, ao_right = st.columns(2)
            # Shots & Goal filters
            for idx, (col_key, lbl, widget_key) in enumerate([
                ("includesShots",   "Includes Shots",    "pa_shots_filter"),
                ("includesGoal",    "Includes Goal",     "pa_goal_filter"),
            ]):
                if col_key not in phases_df.columns:
                    continue
                container = ao_left if idx % 2 == 0 else ao_right
                with container:
                    choice = st.radio(f"**{lbl}**", ["Any", "True", "False"], horizontal=True, key=widget_key)
                    pa_bool_choices[col_key] = choice

            # Overload filter — with type sub-selection and pitch zone viz
            st.markdown("---")
            ovl_c1, ovl_c2 = st.columns(2)
            with ovl_c1:
                if "containsOverload" in phases_df.columns:
                    ovl_choice = st.radio("**Contains Overload**", ["Any", "True", "False"], horizontal=True, key="pa_overload_filter")
                    pa_bool_choices["containsOverload"] = ovl_choice
                else:
                    ovl_choice = "Any"

            pa_overload_types: list[str] = []
            with ovl_c2:
                if ovl_choice == "True" and "overloadTypes" in phases_df.columns:
                    pa_overload_types = st.multiselect(
                        "**Overload Type**",
                        options=["Wide", "Central"],
                        default=["Wide", "Central"],
                        key="pa_overload_type",
                    )

            # Pitch zone visualisation for selected overload types
            if ovl_choice == "True" and pa_overload_types:
                fig_ovl = go.Figure()
                for shape in opta_pitch_shapes():
                    fig_ovl.add_shape(**shape)
                if "Wide" in pa_overload_types:
                    # Bottom wide zone: y 0–19
                    fig_ovl.add_shape(
                        type="rect", x0=0, y0=0, x1=100, y1=19,
                        xref="x", yref="y",
                        fillcolor="rgba(250,165,26,0.20)", line={"color": BRAND_AMBER, "width": 2},
                    )
                    # Top wide zone: y 81–100
                    fig_ovl.add_shape(
                        type="rect", x0=0, y0=81, x1=100, y1=100,
                        xref="x", yref="y",
                        fillcolor="rgba(250,165,26,0.20)", line={"color": BRAND_AMBER, "width": 2},
                    )
                    fig_ovl.add_annotation(
                        x=50, y=9.5, text="Wide", showarrow=False,
                        font={"color": BRAND_AMBER, "size": 13, "family": "Barlow"},
                    )
                    fig_ovl.add_annotation(
                        x=50, y=90.5, text="Wide", showarrow=False,
                        font={"color": BRAND_AMBER, "size": 13, "family": "Barlow"},
                    )
                if "Central" in pa_overload_types:
                    fig_ovl.add_shape(
                        type="rect", x0=0, y0=19, x1=100, y1=81,
                        xref="x", yref="y",
                        fillcolor="rgba(229,32,47,0.15)", line={"color": BRAND_RED, "width": 2},
                    )
                    fig_ovl.add_annotation(
                        x=50, y=50, text="Central", showarrow=False,
                        font={"color": BRAND_RED, "size": 13, "family": "Barlow"},
                    )
                fig_ovl.update_layout(
                    height=220, plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
                    margin={"l": 2, "r": 2, "t": 2, "b": 2},
                    xaxis={"range": [-2, 102], "showgrid": False, "zeroline": False,
                           "showticklabels": False, "scaleanchor": "y", "scaleratio": 105 / 68},
                    yaxis={"range": [-2, 102], "showgrid": False, "zeroline": False,
                           "showticklabels": False},
                    showlegend=False,
                )
                st.plotly_chart(fig_ovl, use_container_width=True, key="pa_overload_pitch")

        with pa_ftab_team:
            pa_tf_col1, pa_tf_col2 = st.columns(2)
            with pa_tf_col1:
                pa_selected_team_name = st.selectbox(
                    "Filter by team",
                    options=["All teams"] + _available_pa_team_names,
                    key="pa_team_filter",
                )
            with pa_tf_col2:
                if "initiatorPlayerId" in phases_df.columns:
                    # Scope initiator options to the selected team (or all teams)
                    if pa_selected_team_name != "All teams":
                        _pa_init_cid = _pa_cid_by_name.get(pa_selected_team_name, "")
                        _pa_init_rows = phases_df[phases_df["possessionContestantId"] == _pa_init_cid]
                    else:
                        _pa_init_rows = phases_df
                    _pa_init_ids = sorted([str(x) for x in _pa_init_rows["initiatorPlayerId"].dropna().unique() if str(x).strip()])
                    if squad_map:
                        _pa_init_options = [squad_map.get(x, x) for x in _pa_init_ids]
                        _pa_init_name_to_id = {squad_map.get(x, x): x for x in _pa_init_ids}
                        _selected_init_names = st.multiselect("Initiating Player", _pa_init_options, default=[], key="pa_initiators")
                        pa_selected_initiators = [_pa_init_name_to_id[n] for n in _selected_init_names if n in _pa_init_name_to_id]
                    else:
                        pa_selected_initiators = st.multiselect("Initiating Player (ID)", _pa_init_ids, default=[], key="pa_initiators")
                else:
                    pa_selected_initiators = []

        with pa_ftab_compact:
            # ── 1. Defensive Compactness Label ────────────────────────────
            _mcdc_col = "mostCommonDefensiveCompactness"
            pa_compact_labels: list[str] = []
            if _mcdc_col in phases_df.columns:
                _mcdc_opts = sorted(phases_df[_mcdc_col].dropna().astype(str).unique())
                if _mcdc_opts:
                    st.markdown("**Most Common Defensive Compactness**")
                    pa_compact_labels = st.multiselect(
                        "Selection (empty = all)",
                        options=_mcdc_opts,
                        default=[],
                        key="pa_compact_labels",
                    )

            st.markdown("---")

            # ── 2. Attacking Team compactness sliders ────────────────────
            st.markdown("**Attacking Team Compactness**")
            _atk_compact_cols = [
                ("averageAttackingTeamHorizontalWidth",    "Horizontal Width (m)",        "pa_atk_width",  68.0),
                ("averageAttackingTeamVerticalLength",     "Vertical Length (m)",         "pa_atk_length", 105.0),
                ("averageAttackingTeamHeightLastDefender", "Height of Last Defender (m)", "pa_atk_height", 105.0),
            ]
            pa_atk_compact_ranges: dict[str, tuple[float, float]] = {}
            atk_cols = st.columns(3)
            for idx, (col, lbl, key, _max) in enumerate(_atk_compact_cols):
                if col in phases_df.columns and phases_df[col].notna().any():
                    with atk_cols[idx]:
                        _sel = st.slider(lbl, min_value=0.0, max_value=_max,
                                         value=(0.0, _max), step=0.1,
                                         format="%.1f", key=f"{key}_slider")
                        pa_atk_compact_ranges[col] = (float(_sel[0]), float(_sel[1]))
                        if _sel[1] >= _max:
                            st.caption(f"Max: >{_max:.0f}")

            st.markdown("---")

            # ── 3. Defending Team compactness sliders ────────────────────
            st.markdown("**Defending Team Compactness**")
            _def_compact_cols = [
                ("averageDefendingTeamHorizontalWidth",    "Horizontal Width (m)",        "pa_def_width",  68.0),
                ("averageDefendingTeamVerticalLength",     "Vertical Length (m)",         "pa_def_length", 105.0),
                ("averageDefendingTeamHeightLastDefender", "Height of Last Defender (m)", "pa_def_height", 105.0),
            ]
            pa_def_compact_ranges: dict[str, tuple[float, float]] = {}
            def_cols = st.columns(3)
            for idx, (col, lbl, key, _max) in enumerate(_def_compact_cols):
                if col in phases_df.columns and phases_df[col].notna().any():
                    with def_cols[idx]:
                        _sel = st.slider(lbl, min_value=0.0, max_value=_max,
                                         value=(0.0, _max), step=0.1,
                                         format="%.1f", key=f"{key}_slider")
                        pa_def_compact_ranges[col] = (float(_sel[0]), float(_sel[1]))
                        if _sel[1] >= _max:
                            st.caption(f"Max: >{_max:.0f}")

    # ── Generate Outputs button ───────────────────────────────────────────
    if st.button("▶ Generate Outputs", type="primary", key="pa_generate"):
        st.session_state["pa_committed"] = {
            "label_mode":           pa_label_mode,
            "selected_labels":      pa_selected_labels,
            "seq_first":            pa_seq_first,
            "seq_leads_to":         pa_seq_leads_to,
            "selected_initiators":  pa_selected_initiators,
            "coord_bounds":         pa_coord_bounds,
            "ac_range_inputs":      pa_ac_range_inputs,
            "bool_choices":         pa_bool_choices,
            "overload_types":       pa_overload_types,
            "selected_team_name":   pa_selected_team_name,
            "compact_labels":       pa_compact_labels,
            "atk_compact_ranges":   pa_atk_compact_ranges,
            "def_compact_ranges":   pa_def_compact_ranges,
        }

    pa_committed = st.session_state.get("pa_committed")
    if not pa_committed:
        st.info("Set your filters above and click **▶ Generate Outputs** to run the analysis.")
        return

    # Unpack committed filter values
    pa_label_mode          = pa_committed["label_mode"]
    pa_selected_labels     = pa_committed["selected_labels"]
    pa_seq_first           = pa_committed["seq_first"]
    pa_seq_leads_to        = pa_committed["seq_leads_to"]
    pa_selected_initiators = pa_committed["selected_initiators"]
    pa_coord_bounds        = pa_committed["coord_bounds"]
    pa_ac_range_inputs     = pa_committed["ac_range_inputs"]
    pa_bool_choices        = pa_committed["bool_choices"]
    pa_overload_types      = pa_committed.get("overload_types", [])
    pa_selected_team_name  = pa_committed["selected_team_name"]
    pa_compact_labels      = pa_committed.get("compact_labels", [])
    pa_atk_compact_ranges  = pa_committed.get("atk_compact_ranges", {})
    pa_def_compact_ranges  = pa_committed.get("def_compact_ranges", {})

    # ── Apply filters ─────────────────────────────────────────────────────
    pa_sequence_mode = pa_label_mode == "Leads to (sequence)"

    if not pa_sequence_mode:
        filtered = phases_df
        if pa_selected_labels:
            filtered = filtered[filtered["phaseLabel"].isin(pa_selected_labels)]
        if pa_selected_initiators:
            filtered = filtered[filtered["initiatorPlayerId"].astype(str).isin(pa_selected_initiators)]
        if pa_coord_bounds:
            for coord, bound_min, bound_max in [
                ("startX", pa_coord_bounds["start_x_min"], pa_coord_bounds["start_x_max"]),
                ("startY", pa_coord_bounds["start_y_min"], pa_coord_bounds["start_y_max"]),
                ("endX",   pa_coord_bounds.get("end_x_min", 0),   pa_coord_bounds.get("end_x_max", 100)),
                ("endY",   pa_coord_bounds.get("end_y_min", 0),   pa_coord_bounds.get("end_y_max", 100)),
            ]:
                if coord in filtered.columns and (bound_min != 0 or bound_max != 100):
                    filtered = filtered[(filtered[coord] >= bound_min) & (filtered[coord] <= bound_max)]
        for col, (v_min, v_max) in pa_ac_range_inputs.items():
            if col in filtered.columns:
                filtered = filtered[(filtered[col] >= v_min) & (filtered[col] <= v_max)]
        for col_key, choice in pa_bool_choices.items():
            if choice != "Any" and col_key in filtered.columns:
                if filtered[col_key].dtype == object:
                    filtered = filtered[filtered[col_key] == choice]
                else:
                    filtered = filtered[filtered[col_key] == (choice == "True")]
        # Overload type sub-filter (only when containsOverload == True)
        if (pa_bool_choices.get("containsOverload") == "True"
                and pa_overload_types
                and "overloadTypes" in filtered.columns):
            def _has_ovl_type(val: str) -> bool:
                if not val:
                    return False
                types_in_phase = set(val.split(","))
                return bool(types_in_phase & set(pa_overload_types))
            filtered = filtered[filtered["overloadTypes"].apply(_has_ovl_type)]

    else:
        if pa_seq_first == "(select)" or not pa_seq_leads_to:
            st.info("Select a Phase label and at least one Leads to label to apply the sequence filter.")
            st.stop()

        group_keys = ["game_id", "periodId", "possessionContestantId"] if "game_id" in phases_df.columns else ["periodId", "possessionContestantId"]
        seq_df = phases_df.sort_values(group_keys + ["startTime"]).reset_index(drop=True)
        pair_records: list[dict] = []
        for _gkey, grp in seq_df.groupby(group_keys, sort=False):
            grp = grp.reset_index(drop=True)
            for i in range(len(grp) - 1):
                if grp.loc[i, "phaseLabel"] == pa_seq_first and grp.loc[i + 1, "phaseLabel"] in pa_seq_leads_to:
                    pair_records.append({"p1": grp.loc[i].to_dict(), "p2": grp.loc[i + 1].to_dict()})

        leads_to_str = ", ".join(f"**{l}**" for l in pa_seq_leads_to)
        if not pair_records:
            st.warning(f"No **{pa_seq_first}** phases immediately followed by {leads_to_str} for the same team in the same period and game.")
            st.stop()

        if pa_selected_initiators:
            pair_records = [pr for pr in pair_records if str(pr["p1"].get("initiatorPlayerId", "")) in pa_selected_initiators]
            if not pair_records:
                st.warning("No phase pairs match the selected initiating player(s).")
                st.stop()

        if pa_coord_bounds:
            def _pa_coord_ok(p1: dict, p2: dict) -> bool:
                for coord, bound_min, bound_max, src in [
                    ("startX", pa_coord_bounds["start_x_min"], pa_coord_bounds["start_x_max"], p1),
                    ("startY", pa_coord_bounds["start_y_min"], pa_coord_bounds["start_y_max"], p1),
                    ("endX",   pa_coord_bounds.get("end_x_min", 0),   pa_coord_bounds.get("end_x_max", 100),   p2),
                    ("endY",   pa_coord_bounds.get("end_y_min", 0),   pa_coord_bounds.get("end_y_max", 100),   p2),
                ]:
                    if bound_min != 0 or bound_max != 100:
                        v = src.get(coord)
                        if v is None or not (bound_min <= v <= bound_max):
                            return False
                return True
            pair_records = [pr for pr in pair_records if _pa_coord_ok(pr["p1"], pr["p2"])]

        for col, (v_min, v_max) in pa_ac_range_inputs.items():
            def _pa_count_ok(pr, col=col, v_min=v_min, v_max=v_max):
                return v_min <= float(pr["p1"].get(col) or 0) + float(pr["p2"].get(col) or 0) <= v_max
            pair_records = [pr for pr in pair_records if _pa_count_ok(pr)]

        for col_key, choice in pa_bool_choices.items():
            if choice == "Any":
                continue
            def _pa_bool_ok(pr, col_key=col_key, choice=choice):
                either_true = str(pr["p1"].get(col_key, "False")).strip() == "True" or str(pr["p2"].get(col_key, "False")).strip() == "True"
                return either_true if choice == "True" else not either_true
            pair_records = [pr for pr in pair_records if _pa_bool_ok(pr)]

        # Overload type sub-filter in sequence mode
        if (pa_bool_choices.get("containsOverload") == "True"
                and pa_overload_types):
            def _seq_ovl_type_ok(pr):
                for pkey in ("p1", "p2"):
                    val = pr[pkey].get("overloadTypes", "")
                    if val:
                        types_in_phase = set(val.split(","))
                        if types_in_phase & set(pa_overload_types):
                            return True
                return False
            pair_records = [pr for pr in pair_records if _seq_ovl_type_ok(pr)]

        if not pair_records:
            st.warning("No phase pairs match all the selected criteria.")
            st.stop()

        if "game_id" in phases_df.columns:
            p1_keys_df = pd.DataFrame([
                {"game_id": str(pr["p1"].get("game_id", "")), "phase_id": str(pr["p1"]["phase_id"])}
                for pr in pair_records
            ]).drop_duplicates()
            candidate = phases_df[phases_df["phaseLabel"] == pa_seq_first].copy()
            candidate["_game_id_str"] = candidate["game_id"].astype(str)
            candidate["_phase_id_str"] = candidate["phase_id"].astype(str)
            filtered = candidate.merge(
                p1_keys_df.rename(columns={"game_id": "_game_id_str", "phase_id": "_phase_id_str"}),
                on=["_game_id_str", "_phase_id_str"], how="inner"
            ).drop(columns=["_game_id_str", "_phase_id_str"])
        else:
            phase1_ids = {str(pr["p1"]["phase_id"]) for pr in pair_records}
            filtered = phases_df[(phases_df["phaseLabel"] == pa_seq_first) & (phases_df["phase_id"].astype(str).isin(phase1_ids))].copy()
        st.caption(f"Showing **{pa_seq_first}** phases immediately followed by {leads_to_str} for the same team in the same period and game — after all filters.")

    # ── Apply Team / Player filter ────────────────────────────────────────────
    if pa_selected_team_name != "All teams":
        pa_selected_cid = _pa_cid_by_name.get(pa_selected_team_name, "")
        filtered = filtered[filtered["possessionContestantId"] == pa_selected_cid]

    # ── Apply Compactness filters ─────────────────────────────────────────────
    _mcdc_col = "mostCommonDefensiveCompactness"
    if pa_compact_labels and _mcdc_col in filtered.columns:
        filtered = filtered[filtered[_mcdc_col].astype(str).isin(pa_compact_labels)]

    # Sentinel max values per column — when selected max equals these, no upper bound is applied
    _compact_col_max = {
        "averageAttackingTeamHorizontalWidth":    68.0,
        "averageDefendingTeamHorizontalWidth":    68.0,
        "averageAttackingTeamVerticalLength":     105.0,
        "averageDefendingTeamVerticalLength":     105.0,
        "averageAttackingTeamHeightLastDefender": 105.0,
        "averageDefendingTeamHeightLastDefender": 105.0,
    }

    for compact_ranges in (pa_atk_compact_ranges, pa_def_compact_ranges):
        for col, (v_min, v_max) in compact_ranges.items():
            if col not in filtered.columns:
                continue
            _sentinel = _compact_col_max.get(col, 100.0)
            _at_default_min = (v_min == 0.0)
            _at_default_max = (v_max >= _sentinel)
            if _at_default_min and _at_default_max:
                continue  # slider untouched — no filter
            col_vals = filtered[col].astype(float)
            mask = filtered[col].notna()
            if not _at_default_min:
                mask = mask & (col_vals >= v_min)
            if not _at_default_max:
                mask = mask & (col_vals <= v_max)
            filtered = filtered[mask]

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**{len(filtered)}** phase(s) match the selected criteria.")
    if filtered.empty:
        return

    tab_list, tab_agg = st.tabs(["📋 Phase List", "📊 Aggregation"])

    with tab_list:
        result_cols = []
        if "game_id" in filtered.columns:
            result_cols.append("game_id")
        result_cols.append("phase_id")
        if "team_name" in filtered.columns:
            result_cols.append("team_name")
        result_cols.extend(["phaseLabel", "periodId", "startTime", "endTime"])
        # Show initiator player name and jersey if available
        if "initiatorPlayerId" in filtered.columns and filtered["initiatorPlayerId"].notna().any():
            if squad_map:
                filtered = filtered.copy()
                filtered["initiator_name"] = filtered["initiatorPlayerId"].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else "")
                result_cols.append("initiator_name")
            else:
                result_cols.append("initiatorPlayerId")
            # Jersey number for initiating player from squad jersey_map
            if jersey_map:
                filtered = filtered if "initiator_name" in filtered.columns else filtered.copy()
                filtered["initiator_jersey"] = filtered["initiatorPlayerId"].map(
                    lambda x: jersey_map.get(str(x), "") if pd.notna(x) else ""
                )
                result_cols.append("initiator_jersey")
        result_cols = [c for c in result_cols if c in filtered.columns]

        display_df = filtered[result_cols].reset_index(drop=True).copy()

        # Format startTime / endTime as mm:ss for readability
        for _tc in ("startTime", "endTime"):
            if _tc in display_df.columns:
                display_df[_tc] = display_df[_tc].apply(ms_to_mmss)

        st.caption("👆 Click a row to select it, then press **▶ Play Video** below.")
        table_selection = st.dataframe(
            display_df,
            use_container_width=True,
            height=min(600, max(200, len(display_df) * 38)),
            selection_mode="single-row",
            on_select="rerun",
            key="pa_phase_table",
        )

        # Determine selected row – prefer click selection, fall back to 0
        _selected_rows = table_selection.selection.get("rows", []) if table_selection and table_selection.selection else []
        selected_row_idx = int(_selected_rows[0]) if _selected_rows else None

        # ── Video playback ────────────────────────────────────────────────
        st.markdown("##### 🎬 Video Playback")

        if selected_row_idx is None:
            st.info("Click a row in the table above to select a phase, then press ▶ Play Video.")
        else:
            _sel_row = display_df.iloc[selected_row_idx]
            _time_str = f"Starting at {_sel_row.get('startTime', '')} in period {_sel_row.get('periodId', '')}"
            st.success(f"Selected: row {selected_row_idx} — **{_sel_row.get('team_name', '')}** — {_time_str}")
        vid_c1, vid_c2 = st.columns(2)
        with vid_c1:
            before_buffer = st.number_input("Pre-buffer (s)", min_value=0, max_value=10, value=5, step=1, key="pa_vid_pre")
        with vid_c2:
            after_buffer = st.number_input("Post-buffer (s)", min_value=0, max_value=10, value=5, step=1, key="pa_vid_post")

        if st.button("▶ Play Video", key="pa_play_video", type="primary", disabled=selected_row_idx is None):
            row = filtered.iloc[selected_row_idx]
            game_id = str(row.get("game_id", ""))
            period_id = int(row.get("periodId", 1))
            start_ms = float(row.get("startTime", 0))
            end_ms = float(row.get("endTime", 0))
            time_in = int(start_ms / 1000)
            time_out = int(end_ms / 1000)

            if not game_id:
                st.error("No game_id available for this phase.")
            else:
                api_key = get_vod_api_key()
                if not api_key:
                    st.error("VOD API key is not configured. Set the `VOD_API_KEY` environment variable or enter it in the sidebar.")
                else:
                    with st.spinner("Fetching video clip…"):
                        try:
                            url = get_vod_streaming(
                                game_uuid=game_id,
                                period=period_id,
                                time_in=time_in,
                                time_out=time_out,
                                before_time=int(before_buffer),
                                after_time=int(after_buffer),
                                api_key=api_key,
                            )
                            components.html(
                                f'<iframe src="{url}" width="950" height="600" '
                                f'frameborder="0" allowfullscreen></iframe>',
                                height=620,
                            )
                        except requests.exceptions.HTTPError as exc:
                            # Only show status code/reason
                            st.error(f"VOD API request failed: {exc.response.status_code} {exc.response.reason}")
                        except requests.exceptions.RequestException:
                            # Network-level errors (timeout, connection refused, etc.)
                            st.error("Network error fetching video clip. Please try again.")
                        except (ValueError, KeyError, IndexError) as exc:
                            st.error(f"Could not retrieve video: {exc}")
                        except Exception:
                            st.error("Unexpected error fetching video. Please try again.")

    with tab_agg:
        team_col = "team_name" if "team_name" in filtered.columns else "possessionContestantId"
        has_initiator = (
            "initiatorPlayerId" in filtered.columns
            and filtered["initiatorPlayerId"].notna().any()
        )
        has_first_touch = (
            "firstTouchPlayerId" in filtered.columns
            and filtered["firstTouchPlayerId"].notna().any()
        )

        # Pre-build name columns for player aggregation (single copy)
        _need_copy = (squad_map and has_initiator) or (squad_map and has_first_touch)
        if _need_copy:
            filtered = filtered.copy()
        if squad_map and has_initiator:
            filtered["_initiator_name"] = filtered["initiatorPlayerId"].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else "")
        if squad_map and has_first_touch:
            filtered["_first_touch_name"] = filtered["firstTouchPlayerId"].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else "")

        agg_c1, agg_c2, agg_c3 = st.columns(3)
        with agg_c1:
            agg_by_type = st.selectbox("Aggregate by", ["Team", "Player"], key="pa_agg_by_type")
        with agg_c2:
            if agg_by_type == "Player":
                player_sub_options = []
                if has_initiator:
                    player_sub_options.append("Initiator")
                if has_first_touch:
                    player_sub_options.append("First touch")
                if not player_sub_options:
                    st.warning("No player ID columns available in this dataset.")
                    player_sub = None
                else:
                    player_sub = st.selectbox("Player type", player_sub_options, key="pa_player_sub")
                team_metric = None
            else:
                team_metric = st.selectbox(
                    "Metric",
                    ["Total count", "Total time (seconds)", "Percentage time"],
                    key="pa_team_sub",
                )
                player_sub = None
        with agg_c3:
            display_as = st.selectbox("Display as", ["Table", "Bar chart"], key="pa_display_as")

        # Resolve grouping column (raw ID/team col for backend), display label, and name col for display
        if agg_by_type == "Team":
            group_col = team_col
            group_by_label = "Team"
            display_name_col = None  # teams shown as-is
        elif player_sub == "Initiator":
            group_col = "initiatorPlayerId"   # ← always the raw ID for groupby
            group_by_label = "Initiating Player"
            display_name_col = "_initiator_name" if "_initiator_name" in filtered.columns else None
        elif player_sub == "First touch":
            group_col = "firstTouchPlayerId"  # ← always the raw ID for groupby
            group_by_label = "First Touch Player"
            display_name_col = "_first_touch_name" if "_first_touch_name" in filtered.columns else None
        else:
            group_col = team_col
            group_by_label = "Team"
            display_name_col = None

        # Build aggregation
        has_duration = "phaseDuration" in filtered.columns and filtered["phaseDuration"].notna().any()

        if agg_by_type == "Team" and has_duration:
            agg_df = (
                filtered.groupby(group_col, dropna=False)
                .agg(total_phases=("phase_id", "count"), total_duration_ms=("phaseDuration", "sum"))
                .reset_index()
            )
            agg_df["total_time_s"] = (agg_df["total_duration_ms"] / 1000.0).round(1)
            # Percentage of each team's own total phase time (across ALL phases, not just filtered)
            team_total_ms = (
                phases_df.groupby(team_col, dropna=False)["phaseDuration"]
                .sum()
                .rename("team_total_ms")
                .reset_index()
            )
            agg_df = agg_df.merge(team_total_ms, on=group_col, how="left")
            agg_df["pct_time"] = (agg_df["total_duration_ms"] / agg_df["team_total_ms"] * 100).round(1)
            agg_df = agg_df.drop(columns=["total_duration_ms", "team_total_ms"])
            # Sort by selected metric
            sort_col = {"Total count": "total_phases", "Total time (seconds)": "total_time_s", "Percentage time": "pct_time"}.get(team_metric, "total_phases")
            agg_df = agg_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        else:
            agg_df = (
                filtered.groupby(group_col, dropna=False)
                .agg(total_phases=("phase_id", "count"))
                .sort_values("total_phases", ascending=False)
                .reset_index()
            )
            sort_col = "total_phases"

        agg_df = agg_df[agg_df[group_col].notna() & (agg_df[group_col].astype(str).str.strip() != "") & (agg_df[group_col].astype(str) != "None")]

        # Map raw player IDs to names for display — keep group_col (raw IDs) for all backend logic
        if display_name_col is not None and squad_map:
            agg_df["_agg_display"] = agg_df[group_col].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else str(x))
        else:
            agg_df["_agg_display"] = agg_df[group_col]
        display_col = "_agg_display"  # use this for all table/chart y-axis labels

        # Column config for nicer display
        col_labels = {
            display_col: group_by_label,
            "total_phases": "Total Count",
            "total_time_s": "Total Time (s)",
            "pct_time": "% of Team Time",
        }

        # Determine which metric column drives the bar chart
        bar_metric_col = sort_col
        bar_metric_label = col_labels.get(bar_metric_col, bar_metric_col)

        # Stacked mode: multi-select with >1 label chosen (not sequence mode)
        use_stacked = (
            not pa_sequence_mode
            and len(pa_selected_labels) > 1
        )

        if display_as == "Table":
            max_rows = max(len(agg_df), 1)
            num_rows = st.number_input(
                "Number of rows to show",
                min_value=1,
                max_value=max_rows,
                value=min(20, max_rows),
                step=1,
                key="pa_table_rows",
            )
            # Show display name column + the chosen metric (raw ID column stays hidden)
            display_cols = [c for c in [display_col, bar_metric_col] if c in agg_df.columns]
            display_agg = agg_df[display_cols].head(int(num_rows)).reset_index(drop=True)
            display_agg.columns = [col_labels.get(c, c) for c in display_agg.columns]
            st.dataframe(
                display_agg,
                use_container_width=True,
                height=min(600, max(120, int(num_rows) * 38 + 40)),
            )
        else:
            max_n = max(len(agg_df), 1)
            top_n = st.number_input(
                "Number of players / teams to show in chart",
                min_value=1,
                max_value=max_n,
                value=min(10, max_n),
                step=1,
                key="pa_bar_top_n",
            )
            chart_title = st.text_input(
                "Chart title (optional)",
                value="",
                placeholder="e.g. Phases by Initiating Player",
                key="pa_bar_title",
            )

            # Top-N raw IDs (for filtering filtered df), and their display names (for chart labels)
            top_rows = agg_df.head(int(top_n))
            top_ids = top_rows[group_col].tolist()          # raw IDs for backend filtering
            top_display = top_rows[display_col].tolist()    # names for chart y-axis

            if use_stacked:
                # Build a per-(group, phaseLabel) breakdown of the chosen metric — group by raw ID
                if bar_metric_col == "pct_time" and has_duration:
                    team_total_ms_map = (
                        phases_df.groupby(team_col, dropna=False)["phaseDuration"]
                        .sum()
                        .to_dict()
                    )
                    stacked_raw = (
                        filtered[filtered[group_col].isin(top_ids)]
                        .groupby([group_col, "phaseLabel"], dropna=False)
                        .agg(_dur_ms=("phaseDuration", "sum"))
                        .reset_index()
                    )
                    stacked_raw["_val"] = stacked_raw.apply(
                        lambda r: round(r["_dur_ms"] / team_total_ms_map.get(r[group_col], 1) * 100, 1),
                        axis=1,
                    )
                    stacked_raw = stacked_raw.drop(columns=["_dur_ms"])
                elif bar_metric_col == "total_time_s" and has_duration:
                    stacked_raw = (
                        filtered[filtered[group_col].isin(top_ids)]
                        .groupby([group_col, "phaseLabel"], dropna=False)
                        .agg(_val=("phaseDuration", "sum"))
                        .reset_index()
                    )
                    stacked_raw["_val"] = (stacked_raw["_val"] / 1000.0).round(1)
                else:  # total_phases (count)
                    stacked_raw = (
                        filtered[filtered[group_col].isin(top_ids)]
                        .groupby([group_col, "phaseLabel"], dropna=False)
                        .agg(_val=("phase_id", "count"))
                        .reset_index()
                    )
                stacked_raw = stacked_raw[
                    stacked_raw[group_col].notna() &
                    (stacked_raw[group_col].astype(str).str.strip() != "") &
                    (stacked_raw[group_col].astype(str) != "None")
                ]
                # Map raw IDs → display names for chart labels
                id_to_display = dict(zip(top_ids, top_display))
                stacked_raw[display_col] = stacked_raw[group_col].map(lambda x: id_to_display.get(x, str(x)))
                stacked_raw[display_col] = pd.Categorical(stacked_raw[display_col], categories=top_display, ordered=True)
                stacked_raw = stacked_raw.sort_values([display_col, "phaseLabel"])

                fig = px.bar(
                    stacked_raw,
                    x="_val",
                    y=display_col,
                    color="phaseLabel",
                    orientation="h",
                    labels={"_val": bar_metric_label, display_col: group_by_label, "phaseLabel": "Phase Label"},
                    title=chart_title if chart_title else None,
                    height=max(300, len(top_display) * 36 + 80),
                    category_orders={display_col: list(reversed(top_display))},
                )
                fig.update_layout(
                    barmode="stack",
                    margin={"l": 10, "r": 50, "t": 50 if chart_title else 30, "b": 10},
                    yaxis={"automargin": True, "color": "#f0f0f0", "gridcolor": "#222"},
                    xaxis={"color": "#f0f0f0", "gridcolor": "#222"},
                    plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
                    font={"color": "#f0f0f0", "family": "Barlow"},
                    hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": BRAND_AMBER,
                                "font": {"size": 12, "color": "#f0f0f0", "family": "Barlow"}},
                )
            else:
                plot_df = agg_df.head(int(top_n)).iloc[::-1].reset_index(drop=True)
                fig = px.bar(
                    plot_df,
                    x=bar_metric_col,
                    y=display_col,
                    orientation="h",
                    text=bar_metric_col,
                    labels={bar_metric_col: bar_metric_label, display_col: group_by_label},
                    title=chart_title if chart_title else None,
                    height=max(300, len(plot_df) * 36 + 80),
                    color_discrete_sequence=[BRAND_AMBER],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    margin={"l": 10, "r": 50, "t": 50 if chart_title else 30, "b": 10},
                    yaxis={"automargin": True, "color": "#f0f0f0", "gridcolor": "#222"},
                    xaxis={"color": "#f0f0f0", "gridcolor": "#222"},
                    plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
                    font={"color": "#f0f0f0", "family": "Barlow"},
                    hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": BRAND_AMBER,
                                "font": {"size": 12, "color": "#f0f0f0", "family": "Barlow"}},
                )
            st.plotly_chart(fig, use_container_width=True, key="pa_agg_bar")



