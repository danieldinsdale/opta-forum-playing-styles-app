"""Block Analysis tab — block faced / deployed proportions."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import BRAND_AMBER, BRAND_ORANGE, BRAND_RED


BLOCK_LABELS = [
    "Build Up against Low Block",
    "Build Up against Medium Block",
    "Build Up against High Block",
]
# Short display names (strip "Build Up against " prefix)
BLOCK_SHORT = {
    "Build Up against Low Block":    "Low Block",
    "Build Up against Medium Block": "Medium Block",
    "Build Up against High Block":   "High Block",
}
# Colours for each block type
BLOCK_COLOURS = {
    "Build Up against Low Block":    BRAND_AMBER,
    "Build Up against Medium Block": BRAND_ORANGE,
    "Build Up against High Block":   BRAND_RED,
}


def block_chart(
    agg_df: pd.DataFrame,
    team_col: str,
    chart_type: str,
    value_label: str,
    chart_key: str,
) -> None:
    """Render a single block-proportion bar chart from a pre-aggregated DataFrame."""
    plot_df = agg_df.copy()
    plot_df["block_short"] = plot_df["phaseLabel"].map(BLOCK_SHORT)
    team_totals = agg_df.groupby(team_col)["value"].sum().rename("_team_total")
    plot_df = plot_df.merge(team_totals, on=team_col, how="left")
    plot_df["proportion"] = (plot_df["value"] / plot_df["_team_total"] * 100).round(1)

    colour_map = {BLOCK_SHORT[k]: v for k, v in BLOCK_COLOURS.items()}
    block_order = [BLOCK_SHORT[l] for l in BLOCK_LABELS]

    if "Stacked bar" in chart_type:
        y_col, y_label = "proportion", "% of Build Up"
        text_col, text_fmt, bar_mode = "proportion", "%{text:.1f}%", "stack"
    else:
        y_col, y_label = "value", value_label
        text_col, text_fmt, bar_mode = "value", "%{text}", "group"

    fig = px.bar(
        plot_df,
        x=team_col, y=y_col,
        color="block_short", barmode=bar_mode, text=text_col,
        color_discrete_map=colour_map,
        category_orders={"block_short": block_order},
        labels={team_col: "Team", y_col: y_label, "block_short": "Block Type"},
        height=420,
    )
    fig.update_traces(texttemplate=text_fmt, textposition="inside")
    fig.update_layout(
        plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        font={"color": "#f0f0f0"},
        legend={"title": {"text": "Block Type"}, "bgcolor": "rgba(0,0,0,0.5)"},
        xaxis={"title": "Team", "color": "#f0f0f0", "gridcolor": "#222"},
        yaxis={"title": y_label, "color": "#f0f0f0", "gridcolor": "#222"},
        margin={"l": 10, "r": 20, "t": 40, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


@st.fragment
def analysis_block_analysis(phases_df: pd.DataFrame, match_info: dict) -> None:
    """Show block-faced and block-deployed proportions for each team's Build Up phases."""
    st.subheader("🧱 Block Analysis")

    # ── Filter to Build Up phases only ───────────────────────────────────
    build_up_df = phases_df[phases_df["phaseLabel"].isin(BLOCK_LABELS)]

    if build_up_df.empty:
        st.warning("No Build Up phases found in the loaded data.")
        return

    contestant_map = match_info.get("contestant_map", {})
    all_contestant_ids = sorted(
        phases_df["possessionContestantId"].dropna().unique()
        if "possessionContestantId" in phases_df.columns else []
    )

    # Copy only when we need to mutate
    build_up_df = build_up_df.copy()

    # Add possessing team name
    if "team_name" not in build_up_df.columns and contestant_map:
        build_up_df["team_name"] = build_up_df["possessionContestantId"].map(contestant_map)
    team_col = "team_name" if "team_name" in build_up_df.columns else "possessionContestantId"

    # Add defending team: in a two-team match the opponent is the other contestant.
    # Works for multi-game data too — opponents are scoped per game + period.
    if len(all_contestant_ids) == 2:
        cid_a, cid_b = all_contestant_ids
        build_up_df["defending_id"] = build_up_df["possessionContestantId"].map(
            {cid_a: cid_b, cid_b: cid_a}
        )
    else:
        # For multi-game data, find the other team in the same game
        game_key = "game_id" if "game_id" in build_up_df.columns else None
        if game_key:
            # Vectorized opponent lookup via merge instead of row-by-row .apply()
            _uniq = phases_df.drop_duplicates(subset=[game_key, "possessionContestantId"])
            _game_team_pairs = (
                _uniq.groupby(game_key)["possessionContestantId"]
                .apply(list)
                .reset_index(name="_all_teams")
            )
            build_up_df = build_up_df.merge(_game_team_pairs, on=game_key, how="left")
            build_up_df["defending_id"] = build_up_df.apply(
                lambda r: next(
                    (t for t in (r.get("_all_teams") or []) if t != r["possessionContestantId"]),
                    None,
                ),
                axis=1,
            )
            build_up_df = build_up_df.drop(columns=["_all_teams"])
        else:
            build_up_df["defending_id"] = None

    if contestant_map:
        build_up_df["defending_name"] = build_up_df["defending_id"].map(contestant_map)
    defending_col = "defending_name" if "defending_name" in build_up_df.columns else "defending_id"

    has_duration = "phaseDuration" in build_up_df.columns and build_up_df["phaseDuration"].notna().any()

    # ── Game state (goal difference from the relevant team's perspective) ─
    has_game_state = "game_state" in build_up_df.columns and build_up_df["game_state"].notna().any()
    if has_game_state:
        # game_state on the phase is from the *possessing* team.
        # For "Block Deployed" the relevant team is the defender → negate.
        build_up_df["_gs_faced"]    = build_up_df["game_state"].astype(int)
        build_up_df["_gs_deployed"] = -build_up_df["game_state"].astype(int)

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
        with ctrl_c1:
            view_choice = st.radio(
                "**View**",
                ["Block Deployed", "Block Faced"],
                horizontal=True,
                key="ba_view",
            )
        with ctrl_c2:
            _metric_options = ["Time (seconds)", "Phase count"] if has_duration else ["Phase count"]
            metric_choice = st.radio(
                "**Metric**",
                _metric_options,
                horizontal=True,
                key="ba_metric",
            )
        with ctrl_c3:
            chart_type = st.radio(
                "**Chart type**",
                ["Stacked bar (proportion)", "Grouped bar (absolute)"],
                horizontal=True,
                key="ba_chart_type",
            )

        # ── Game State filter ─────────────────────────────────────────────
        gs_range_val = None
        if has_game_state:
            gs_col = "_gs_deployed" if view_choice == "Block Deployed" else "_gs_faced"
            gs_min = int(build_up_df[gs_col].min())
            gs_max = int(build_up_df[gs_col].max())
            if gs_min < gs_max:
                gs_range_val = st.slider(
                    "**Game State** (goal difference from selected team's perspective)",
                    min_value=gs_min,
                    max_value=gs_max,
                    value=(gs_min, gs_max),
                    step=1,
                    key="ba_game_state",
                )
            else:
                st.caption(f"Game State: all phases at **{gs_min}**")

    # ── Generate Outputs button ────────────────────────────────────────────
    if st.button("▶ Generate Outputs", type="primary", key="ba_generate"):
        st.session_state["ba_committed"] = {
            "view_choice": view_choice,
            "metric_choice": metric_choice,
            "chart_type": chart_type,
            "gs_range": gs_range_val,
        }

    ba_committed = st.session_state.get("ba_committed")
    if not ba_committed:
        st.info("Set your filters above and click **▶ Generate Outputs** to run the analysis.")
        return

    # Use committed values
    view_choice = ba_committed["view_choice"]
    metric_choice = ba_committed["metric_choice"]
    chart_type = ba_committed["chart_type"]
    gs_range_committed = ba_committed.get("gs_range")

    # Apply game state filter from committed values
    if gs_range_committed is not None and has_game_state:
        gs_col = "_gs_deployed" if view_choice == "Block Deployed" else "_gs_faced"
        build_up_df = build_up_df[
            (build_up_df[gs_col] >= gs_range_committed[0]) & (build_up_df[gs_col] <= gs_range_committed[1])
        ]
        if build_up_df.empty:
            st.warning("No Build Up phases match the selected game state range.")
            return

    use_duration = metric_choice == "Time (seconds)" and has_duration

    def _build_agg(group_col: str) -> tuple[pd.DataFrame, str]:
        """Aggregate build_up_df by group_col + phaseLabel."""
        if use_duration:
            df = (
                build_up_df
                .groupby([group_col, "phaseLabel"], dropna=False)
                .agg(value=("phaseDuration", "sum"))
                .reset_index()
            )
            df["value"] = (df["value"] / 1000.0).round(1)
            label = "Time (s)"
        else:
            df = (
                build_up_df
                .groupby([group_col, "phaseLabel"], dropna=False)
                .agg(value=("phase_id", "count"))
                .reset_index()
            )
            label = "Phase Count"
        df = df.rename(columns={group_col: "team"})
        df["team"] = df["team"].astype(str)
        df = df[df["team"].notna() & (df["team"] != "None") & (df["team"].str.strip() != "")]
        return df, label

    if view_choice == "Block Deployed":
        st.markdown("#### Block Deployed")
        st.caption("Proportion of time each team deployed each block type while the opponent was in Build Up.")
        deployed_df, value_label = _build_agg(defending_col)
        deployed_df = deployed_df.rename(columns={"team": defending_col})
        block_chart(deployed_df, defending_col, chart_type, value_label, "ba_deployed_chart")
    else:
        st.markdown("#### Block Faced")
        st.caption("Proportion of each team's own Build Up phases spent against each block type.")
        faced_df, value_label = _build_agg(team_col)
        faced_df = faced_df.rename(columns={"team": team_col})
        block_chart(faced_df, team_col, chart_type, value_label, "ba_faced_chart")

