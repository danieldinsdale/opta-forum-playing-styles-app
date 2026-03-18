"""Team Compactness tab — time-weighted width & length per team."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import BRAND_AMBER, BRAND_ORANGE, BRAND_RED


# Columns we need from the phase data (parsed from phaseSummary stats)
_ATK_WIDTH   = "averageAttackingTeamHorizontalWidth"
_ATK_LENGTH  = "averageAttackingTeamVerticalLength"
_ATK_HEIGHT  = "averageAttackingTeamHeightLastDefender"
_DEF_WIDTH   = "averageDefendingTeamHorizontalWidth"
_DEF_LENGTH  = "averageDefendingTeamVerticalLength"
_DEF_HEIGHT  = "averageDefendingTeamHeightLastDefender"
_DEF_AREA    = "averageDefensiveAreaCoverage"
_DURATION    = "phaseDuration"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _add_opponent_column(
    df: pd.DataFrame,
    phases_df: pd.DataFrame,
    contestant_map: dict[str, str],
) -> tuple[pd.DataFrame, str]:
    """Add ``opponent_id`` / ``opponent_name`` and return (df, display_col)."""
    all_cids = sorted(
        phases_df["possessionContestantId"].dropna().unique()
        if "possessionContestantId" in phases_df.columns
        else []
    )

    if len(all_cids) == 2:
        cid_a, cid_b = all_cids
        df["opponent_id"] = df["possessionContestantId"].map(
            {cid_a: cid_b, cid_b: cid_a}
        )
    else:
        game_key = "game_id" if "game_id" in df.columns else None
        if game_key:
            _uniq = phases_df.drop_duplicates(
                subset=[game_key, "possessionContestantId"]
            )
            _pairs = (
                _uniq.groupby(game_key)["possessionContestantId"]
                .apply(list)
                .reset_index(name="_all_teams")
            )
            df = df.merge(_pairs, on=game_key, how="left")
            df["opponent_id"] = df.apply(
                lambda r: next(
                    (
                        t
                        for t in (r.get("_all_teams") or [])
                        if t != r["possessionContestantId"]
                    ),
                    None,
                ),
                axis=1,
            )
            df = df.drop(columns=["_all_teams"])
        else:
            df["opponent_id"] = None

    if contestant_map:
        df["opponent_name"] = df["opponent_id"].map(contestant_map)
    opp_col = (
        "opponent_name" if "opponent_name" in df.columns else "opponent_id"
    )
    return df, opp_col


def _weighted_avg(
    group: pd.DataFrame, value_col: str, weight_col: str
) -> float:
    """Duration-weighted average; returns NaN when no valid data."""
    mask = (
        group[value_col].notna()
        & group[weight_col].notna()
        & (group[weight_col] > 0)
    )
    g = group.loc[mask]
    if g.empty:
        return float("nan")
    total_w = g[weight_col].sum()
    if total_w == 0:
        return float("nan")
    return (g[value_col] * g[weight_col]).sum() / total_w


def _compute_compactness(
    df: pd.DataFrame,
    team_col: str,
    width_col: str,
    length_col: str,
    area_col: str | None = None,
    height_col: str | None = None,
) -> pd.DataFrame:
    """Time-weighted Width, Length, Height of Last Defender, and optionally Area Coverage per team."""
    rows: list[dict] = []
    for team, grp in df.groupby(team_col, dropna=False):
        if pd.isna(team) or str(team).strip() in ("", "None"):
            continue
        w_avg = _weighted_avg(grp, width_col, _DURATION)
        l_avg = _weighted_avg(grp, length_col, _DURATION)
        total_time = grp[_DURATION].sum() / 1000.0
        n_phases = len(grp)
        row_dict: dict = {
            "Team": str(team),
            "Horizontal Width (m)": round(w_avg, 2) if pd.notna(w_avg) else None,
            "Vertical Length (m)": round(l_avg, 2) if pd.notna(l_avg) else None,
            "Phases": n_phases,
            "Total Time (s)": round(total_time, 1),
        }
        if height_col and height_col in df.columns:
            h_avg = _weighted_avg(grp, height_col, _DURATION)
            row_dict["Height of Last Defender (m)"] = round(h_avg, 2) if pd.notna(h_avg) else None
        if area_col and area_col in df.columns:
            a_avg = _weighted_avg(grp, area_col, _DURATION)
            row_dict["Area Coverage (m²)"] = round(a_avg, 2) if pd.notna(a_avg) else None
        rows.append(row_dict)
    return pd.DataFrame(rows)


def _apply_phase_label_filter(
    phases_df: pd.DataFrame,
    label_mode: str,
    selected_labels: list[str],
    seq_first: str | None,
    seq_leads_to: list[str],
) -> pd.DataFrame:
    """Return the subset of *phases_df* matching the chosen phase-label criteria.

    In **multi-select** mode only phases whose label is in *selected_labels* are
    kept (empty selection = all phases).

    In **leads-to (sequence)** mode, we find consecutive phase pairs where the
    1st phase matches *seq_first* and the 2nd matches one of *seq_leads_to*
    (same team, same period, same game).  Only the **1st phase** of each pair is
    returned so the compactness calculation uses only that phase.

    Uses the same string-based merge approach as the Phase Analysis tab.
    """
    if label_mode == "Any of (multi-select)":
        if selected_labels:
            return phases_df[phases_df["phaseLabel"].isin(selected_labels)]
        return phases_df

    # ── Leads-to (sequence) mode ──────────────────────────────────────────
    if not seq_first or seq_first == "(select)" or not seq_leads_to:
        return phases_df.iloc[0:0]  # empty — caller should warn

    group_keys = (
        ["game_id", "periodId", "possessionContestantId"]
        if "game_id" in phases_df.columns
        else ["periodId", "possessionContestantId"]
    )
    seq_df = phases_df.sort_values(group_keys + ["startTime"]).reset_index(
        drop=True
    )

    # Collect the 1st phase of each matching pair as a dict.
    # Cast labels to str so comparisons work regardless of category dtype.
    seq_first_str = str(seq_first)
    seq_leads_to_set = {str(l) for l in seq_leads_to}

    pair_records: list[dict] = []
    for _gkey, grp in seq_df.groupby(group_keys, sort=False):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - 1):
            if (
                str(grp.loc[i, "phaseLabel"]) == seq_first_str
                and str(grp.loc[i + 1, "phaseLabel"]) in seq_leads_to_set
            ):
                pair_records.append(grp.loc[i].to_dict())

    if not pair_records:
        return phases_df.iloc[0:0]

    # Match back to original phases_df using string-based merge on
    # (game_id, phase_id) — the same approach used in tab_phases.py.
    if "game_id" in phases_df.columns:
        p1_keys_df = pd.DataFrame([
            {"game_id": str(pr.get("game_id", "")), "phase_id": str(pr["phase_id"])}
            for pr in pair_records
        ]).drop_duplicates()
        candidate = phases_df[
            phases_df["phaseLabel"].astype(str) == seq_first_str
        ].copy()
        candidate["_game_id_str"] = candidate["game_id"].astype(str)
        candidate["_phase_id_str"] = candidate["phase_id"].astype(str)
        result = candidate.merge(
            p1_keys_df.rename(
                columns={"game_id": "_game_id_str", "phase_id": "_phase_id_str"}
            ),
            on=["_game_id_str", "_phase_id_str"],
            how="inner",
        ).drop(columns=["_game_id_str", "_phase_id_str"])
    else:
        phase1_ids = {str(pr["phase_id"]) for pr in pair_records}
        result = phases_df[
            (phases_df["phaseLabel"].astype(str) == seq_first_str)
            & (phases_df["phase_id"].astype(str).isin(phase1_ids))
        ].copy()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Chart
# ──────────────────────────────────────────────────────────────────────────────


def _render_dimensions_chart(
    result_df: pd.DataFrame, title: str, chart_key: str
) -> None:
    """Grouped bar chart — Width, Length, and Height of Last Defender per team."""
    if result_df.empty:
        st.info("No data to display.")
        return

    teams = result_df["Team"].tolist()
    has_height = (
        "Height of Last Defender (m)" in result_df.columns
        and result_df["Height of Last Defender (m)"].notna().any()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=teams, y=result_df["Horizontal Width (m)"],
        name="Horizontal Width (m)", marker_color=BRAND_AMBER,
        text=result_df["Horizontal Width (m)"], textposition="outside",
        texttemplate="%{text:.1f}",
    ))
    fig.add_trace(go.Bar(
        x=teams, y=result_df["Vertical Length (m)"],
        name="Vertical Length (m)", marker_color=BRAND_ORANGE,
        text=result_df["Vertical Length (m)"], textposition="outside",
        texttemplate="%{text:.1f}",
    ))
    if has_height:
        fig.add_trace(go.Bar(
            x=teams, y=result_df["Height of Last Defender (m)"],
            name="Height of Last Defender (m)", marker_color=BRAND_RED,
            text=result_df["Height of Last Defender (m)"], textposition="outside",
            texttemplate="%{text:.1f}",
        ))
    fig.update_layout(
        barmode="group",
        title={"text": title, "font": {"size": 15, "color": "#f0f0f0"}, "x": 0.5},
        plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        font={"color": "#f0f0f0"}, legend={"bgcolor": "rgba(0,0,0,0.5)"},
        xaxis={"title": "Team", "color": "#f0f0f0", "gridcolor": "#222"},
        yaxis={"title": "Metres", "color": "#f0f0f0", "gridcolor": "#222"},
        margin={"l": 10, "r": 20, "t": 50, "b": 10}, height=420,
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _render_area_chart(
    result_df: pd.DataFrame, title: str, chart_key: str
) -> None:
    """Bar chart — Defensive Area Coverage per team."""
    if result_df.empty or "Area Coverage (m²)" not in result_df.columns:
        st.info("No area coverage data to display.")
        return

    teams = result_df["Team"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=teams, y=result_df["Area Coverage (m²)"],
        name="Area Coverage (m²)", marker_color=BRAND_RED,
        text=result_df["Area Coverage (m²)"], textposition="outside",
        texttemplate="%{text:.0f}",
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": "#f0f0f0"}, "x": 0.5},
        plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        font={"color": "#f0f0f0"}, legend={"bgcolor": "rgba(0,0,0,0.5)"},
        xaxis={"title": "Team", "color": "#f0f0f0", "gridcolor": "#222"},
        yaxis={"title": "Area (m²)", "color": "#f0f0f0", "gridcolor": "#222"},
        margin={"l": 10, "r": 20, "t": 50, "b": 10}, height=420,
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────────


@st.fragment
def analysis_team_compactness(
    phases_df: pd.DataFrame, match_info: dict
) -> None:
    """Show time-weighted team compactness (width & length) in/out of possession."""
    st.subheader("📏 Team Compactness")

    # ── Check required columns ────────────────────────────────────────────
    required = [_ATK_WIDTH, _ATK_LENGTH, _DEF_WIDTH, _DEF_LENGTH, _DURATION]
    missing = [
        c
        for c in required
        if c not in phases_df.columns or phases_df[c].isna().all()
    ]
    if missing:
        st.warning(
            f"Phase data is missing required compactness columns: "
            f"{', '.join(missing)}. "
            "Make sure the phase feed contains team shape statistics."
        )
        return

    contestant_map: dict[str, str] = match_info.get("contestant_map", {})
    available_labels = sorted(phases_df["phaseLabel"].unique())

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        tc_ftab_possession, tc_ftab_labels = st.tabs(
            ["👁️ Possession State", "🏷️ Phase Labels"]
        )

        with tc_ftab_possession:
            view = st.radio(
                "**Possession state**",
                ["In Possession", "Out of Possession"],
                horizontal=True,
                key="tc_view",
            )

        with tc_ftab_labels:
            tc_label_mode = st.radio(
                "Selection mode",
                ["Any of (multi-select)", "Leads to (sequence)"],
                horizontal=True,
                key="tc_label_mode",
            )

            if tc_label_mode == "Any of (multi-select)":
                _, btn_all, btn_clr = st.columns([6, 1, 1])
                with btn_all:
                    if st.button("Select all", key="tc_select_all"):
                        st.session_state["tc_labels"] = available_labels
                with btn_clr:
                    if st.button("Clear", key="tc_clear"):
                        st.session_state["tc_labels"] = []
                tc_selected_labels: list[str] = st.multiselect(
                    "Phase labels",
                    available_labels,
                    default=[],
                    key="tc_labels",
                )
                tc_seq_first = None
                tc_seq_leads_to: list[str] = []
            else:
                seq_c1, seq_c2 = st.columns(2)
                with seq_c1:
                    tc_seq_first = st.selectbox(
                        "Phase label",
                        options=["(select)"] + available_labels,
                        key="tc_seq_first",
                    )
                with seq_c2:
                    tc_seq_leads_to = st.multiselect(
                        "Leads to",
                        options=available_labels,
                        default=[],
                        key="tc_seq_leads_to",
                    )
                tc_selected_labels = []

    # ── Generate Outputs button ────────────────────────────────────────────
    if st.button("▶ Generate Outputs", type="primary", key="tc_generate"):
        st.session_state["tc_committed"] = {
            "view": view,
            "label_mode": tc_label_mode,
            "selected_labels": tc_selected_labels,
            "seq_first": tc_seq_first,
            "seq_leads_to": tc_seq_leads_to,
        }

    tc_committed = st.session_state.get("tc_committed")
    if not tc_committed:
        st.info("Set your filters above and click **▶ Generate Outputs** to run the analysis.")
        return

    # Use committed values
    view = tc_committed["view"]
    tc_label_mode = tc_committed["label_mode"]
    tc_selected_labels = tc_committed["selected_labels"]
    tc_seq_first = tc_committed["seq_first"]
    tc_seq_leads_to = tc_committed["seq_leads_to"]

    # ── Apply phase-label filter ──────────────────────────────────────────
    is_sequence = tc_label_mode == "Leads to (sequence)"

    if is_sequence and (not tc_seq_first or tc_seq_first == "(select)" or not tc_seq_leads_to):
        st.info(
            "Select a Phase label and at least one **Leads to** label, "
            "then click **▶ Generate Outputs** to calculate."
        )
        return

    work_df = _apply_phase_label_filter(
        phases_df,
        tc_label_mode,
        tc_selected_labels,
        tc_seq_first,
        tc_seq_leads_to,
    )

    if work_df.empty:
        if is_sequence:
            leads_to_str = ", ".join(f"**{l}**" for l in tc_seq_leads_to)
            st.warning(
                f"No **{tc_seq_first}** phases immediately followed by "
                f"{leads_to_str} for the same team in the same period/game."
            )
        else:
            st.warning("No phases match the selected labels.")
        return

    # ── Ensure team columns ───────────────────────────────────────────────
    work_df = work_df.copy()  # own the DataFrame before mutating
    if "team_name" not in work_df.columns and contestant_map:
        work_df["team_name"] = work_df["possessionContestantId"].map(
            contestant_map
        )
    team_col = (
        "team_name"
        if "team_name" in work_df.columns
        else "possessionContestantId"
    )

    # Add opponent column (needed for Out of Possession view)
    work_df, opp_col = _add_opponent_column(work_df, phases_df, contestant_map)

    # ── Caption ───────────────────────────────────────────────────────────
    if is_sequence:
        leads_to_str = ", ".join(f"**{l}**" for l in tc_seq_leads_to)
        st.caption(
            f"Showing compactness for **{tc_seq_first}** phases immediately "
            f"followed by {leads_to_str} — calculation uses the 1st phase only."
        )

    n_label = len(work_df)
    label_summary = (
        f"**{n_label}** phase(s) included in the calculation"
        + (f" (labels: {', '.join(tc_selected_labels)})" if tc_selected_labels else "")
        + "."
    )
    st.markdown(label_summary)

    # ── Compute & render ──────────────────────────────────────────────────
    if view == "In Possession":
        st.markdown("#### In Possession Compactness")
        st.caption(
            "Time-weighted average of the **attacking** team's horizontal "
            "width, vertical length, and height of last defender while they are in possession."
        )
        result = _compute_compactness(
            work_df, team_col, _ATK_WIDTH, _ATK_LENGTH, height_col=_ATK_HEIGHT
        )
        _render_dimensions_chart(result, "In Possession", "tc_ip_chart")
    else:
        st.markdown("#### Out of Possession Compactness")
        result = _compute_compactness(
            work_df, opp_col, _DEF_WIDTH, _DEF_LENGTH,
            area_col=_DEF_AREA, height_col=_DEF_HEIGHT
        )
        _has_area_data = (
            _DEF_AREA in work_df.columns
            and work_df[_DEF_AREA].notna().any()
            and "Area Coverage (m²)" in result.columns
            and result["Area Coverage (m²)"].notna().any()
        )
        _oop_metric_options = ["Horizontal Width & Vertical Length (m)"]
        if _has_area_data:
            _oop_metric_options.append("Area Coverage (m²)")
        oop_metric = st.radio(
            "**Metric**",
            _oop_metric_options,
            horizontal=True,
            key="tc_oop_metric",
        )
        if oop_metric == "Area Coverage (m²)":
            st.caption(
                "Time-weighted average **defensive area coverage** (m²) while "
                "the opponent is in possession. "
                "Attributed to the defending (out-of-possession) team."
            )
            _render_area_chart(result, "Out of Possession — Defensive Area Coverage", "tc_oop_chart")
        else:
            st.caption(
                "Time-weighted average of the **defending** team's horizontal "
                "width, vertical length, and height of last defender while the opponent is in possession. "
                "Attributed to the defending (out-of-possession) team."
            )
            _render_dimensions_chart(result, "Out of Possession", "tc_oop_chart")

    # ── Summary table ─────────────────────────────────────────────────────
    if not result.empty:
        st.dataframe(
            result,
            use_container_width=True,
            height=min(400, max(120, len(result) * 38 + 40)),
        )

