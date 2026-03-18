"""Pitch drawing shapes, pitch-map rendering, and zone-selector widget."""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.config import BRAND_PURPLE, BRAND_AMBER, BRAND_ORANGE, BRAND_RED


@lru_cache(maxsize=1)
def opta_pitch_shapes() -> tuple:
    """Plotly shapes for an Opta 0-100 x 0-100 pitch (105m x 68m).
    Cached — trig computed once then reused on every subsequent call."""
    line = {"type": "line", "line": {"color": "#aaaaaa", "width": 1.5}, "xref": "x", "yref": "y"}
    rect = {"type": "rect", "line": {"color": "#aaaaaa", "width": 1.5}, "fillcolor": "rgba(0,0,0,0)", "xref": "x", "yref": "y"}
    circ = {"type": "circle", "line": {"color": "#aaaaaa", "width": 1.5}, "fillcolor": "rgba(0,0,0,0)", "xref": "x", "yref": "y"}

    def ln(x0, y0, x1, y1): return {**line, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    def bx(x0, y0, x1, y1): return {**rect, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    def cl(x0, y0, x1, y1): return {**circ, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    def ox(m): return m / 105 * 100
    def oy(m): return m / 68 * 100

    pa_d = ox(16.5); pa_y0 = oy((68 - 40.32) / 2); pa_y1 = oy((68 + 40.32) / 2)
    sb_d = ox(5.5);  sb_y0 = oy((68 - 18.32) / 2); sb_y1 = oy((68 + 18.32) / 2)
    gl_d = ox(2.44); gl_y0 = oy((68 - 7.32) / 2);  gl_y1 = oy((68 + 7.32) / 2)
    ps_x = ox(11.0)
    cc_rx = ox(9.15); cc_ry = oy(9.15); cc_cx = 50.0; cc_cy = 50.0
    arc_rx = ox(9.15); arc_ry = oy(9.15)

    _cos_lim = (pa_d - ps_x) / arc_rx
    _t_lim = math.acos(min(1.0, _cos_lim))
    _ARC_N = 40

    def _arc_segments(cx, sign):
        segs = []
        for i in range(_ARC_N):
            t0 = -_t_lim + (2 * _t_lim) * i / _ARC_N
            t1 = -_t_lim + (2 * _t_lim) * (i + 1) / _ARC_N
            segs.append(ln(
                cx + sign * arc_rx * math.cos(t0), 50.0 + arc_ry * math.sin(t0),
                cx + sign * arc_rx * math.cos(t1), 50.0 + arc_ry * math.sin(t1),
            ))
        return segs

    left_arc = _arc_segments(ps_x, +1.0)
    right_arc = _arc_segments(100.0 - ps_x, -1.0)

    return tuple([
        bx(0, 0, 100, 100), ln(50, 0, 50, 100),
        cl(cc_cx - cc_rx, cc_cy - cc_ry, cc_cx + cc_rx, cc_cy + cc_ry),
        ln(cc_cx - 0.3, cc_cy, cc_cx + 0.3, cc_cy),
        bx(0, pa_y0, pa_d, pa_y1), bx(0, sb_y0, sb_d, sb_y1),
        ln(ps_x - 0.3, 50, ps_x + 0.3, 50), *left_arc,
        bx(100 - pa_d, pa_y0, 100, pa_y1), bx(100 - sb_d, sb_y0, 100, sb_y1),
        ln(100 - ps_x - 0.3, 50, 100 - ps_x + 0.3, 50), *right_arc,
        bx(-gl_d, gl_y0, 0, gl_y1), bx(100, gl_y0, 100 + gl_d, gl_y1),
    ])


def render_runs_pitch_map(result_df: pd.DataFrame, match_info: dict, squad_map: dict[str, str] | None = None) -> None:
    """Interactive Plotly pitch map of run start locations with hover metadata."""
    if squad_map is None:
        squad_map = {}
    plot_df = result_df.dropna(subset=["startX", "startY"]).copy()
    if plot_df.empty:
        st.warning("No runs with coordinate data to plot.")
        return

    # Add player name column
    if "playerId" in plot_df.columns:
        plot_df["_player_name"] = plot_df["playerId"].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else "")

    has_teams = "team_name" in plot_df.columns and plot_df["team_name"].notna().any()

    show_arrows = st.checkbox("Show direction arrows", value=True, key="pm_arrows")

    palette = [BRAND_PURPLE, BRAND_ORANGE, BRAND_AMBER, "#222222", "#aaaaaa", "#555555"]
    num_teams = plot_df["team_name"].nunique() if has_teams else 0
    colour_by_team = num_teams > 1

    if colour_by_team:
        team_names = sorted(plot_df["team_name"].dropna().unique())
        team_colour = {name: palette[i % len(palette)] for i, name in enumerate(team_names)}
        plot_df["_colour"] = plot_df["team_name"].map(team_colour)
    else:
        team_colour = {}
        plot_df["_colour"] = palette[0]

    def _ms_to_mmss(val) -> str:
        try:
            total_s = int(float(val)) // 1000
            return f"{total_s // 60}:{total_s % 60:02d}"
        except (TypeError, ValueError):
            return str(val)

    plot_df["_start_mmss"] = plot_df["startTime"].apply(_ms_to_mmss) if "startTime" in plot_df.columns else ""
    if "_player_name" not in plot_df.columns:
        plot_df["_player_name"] = plot_df.get("playerId", "")

    fig = go.Figure()
    for shape in opta_pitch_shapes():
        fig.add_shape(**shape)

    hover_tmpl = (
        "<b>Run %{customdata[0]}</b><br>"
        "Player: %{customdata[1]}<br>"
        "Phase: %{customdata[2]}<br>"
        "Phase label: %{customdata[3]}<br>"
        "Period: %{customdata[4]}<br>"
        "Start time: %{customdata[5]}<br>"
        "Main label: %{customdata[6]}<br>"
        "<extra></extra>"
    )
    custom_cols = ["run_id", "_player_name", "phase_id", "phaseLabel", "periodId", "_start_mmss", "masterLabel"]
    for col in custom_cols:
        if col not in plot_df.columns:
            plot_df[col] = ""

    if show_arrows and "endX" in plot_df.columns and "endY" in plot_df.columns:
        arr_df = plot_df.dropna(subset=["endX", "endY"])
        if not arr_df.empty:
            arrow_colours = arr_df["_colour"].unique()
            for colour in arrow_colours:
                grp_a = arr_df[arr_df["_colour"] == colour]
                sx = grp_a["startX"].values
                ex = grp_a["endX"].values
                sy = grp_a["startY"].values
                ey = grp_a["endY"].values
                xs = np.empty(len(sx) * 3, dtype=object)
                xs[0::3] = sx; xs[1::3] = ex; xs[2::3] = None
                ys = np.empty(len(sy) * 3, dtype=object)
                ys[0::3] = sy; ys[1::3] = ey; ys[2::3] = None
                fig.add_trace(go.Scatter(
                    x=xs.tolist(), y=ys.tolist(), mode="lines",
                    line={"color": colour, "width": 1.5},
                    hoverinfo="skip", showlegend=False,
                ))
                mx = np.empty(len(sx) * 2, dtype=object)
                mx[0::2] = sx; mx[1::2] = ex
                my = np.empty(len(sy) * 2, dtype=object)
                my[0::2] = sy; my[1::2] = ey
                msizes = np.tile([0, 10], len(sx))
                msymbols = np.tile(["circle", "arrow"], len(sx))
                fig.add_trace(go.Scatter(
                    x=mx.tolist(), y=my.tolist(), mode="markers",
                    marker={"symbol": msymbols.tolist(), "color": colour,
                            "size": msizes.tolist(), "angleref": "previous",
                            "line": {"width": 0}},
                    hoverinfo="skip", showlegend=False,
                ))

    groups = plot_df.groupby("team_name") if colour_by_team else [("Runs", plot_df)]
    for label, grp in groups:
        colour = team_colour.get(str(label), palette[0]) if colour_by_team else palette[0]
        fig.add_trace(go.Scatter(
            x=grp["startX"], y=grp["startY"], mode="markers", name=str(label),
            marker={"color": colour, "size": 9, "line": {"color": "white", "width": 1}},
            customdata=grp[custom_cols].values, hovertemplate=hover_tmpl,
            hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": BRAND_AMBER, "font": {"size": 12, "family": "Barlow"}},
        ))

    fig.update_layout(
        height=700, plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        font={"color": "#f0f0f0", "family": "Barlow"},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        xaxis={"range": [-4, 104], "showgrid": False, "zeroline": False,
               "showticklabels": False, "scaleanchor": "y", "scaleratio": 105 / 68},
        yaxis={"range": [-4, 104], "showgrid": False, "zeroline": False, "showticklabels": False},
        showlegend=colour_by_team,
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(0,0,0,0.6)", "font": {"color": "#f0f0f0"}},
        title={"text": f"{len(plot_df)} run(s) plotted", "font": {"size": 13, "color": "#f0f0f0"}, "x": 0.5},
        hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": BRAND_AMBER,
                    "font": {"size": 12, "color": "#f0f0f0", "family": "Barlow"}, "align": "left", "namelength": 0},
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, key="runs_pitch_map")


def pitch_zone_selector(key_prefix: str, has_start: bool = True, has_end: bool = True) -> dict:
    """Render start/end zone sliders side-by-side, with the pitch preview below.

    Sliders and returned bounds all use the 0-100 Opta coordinate space.
    """
    bounds: dict = {}

    start_col, end_col = st.columns(2)

    with start_col:
        if has_start:
            st.markdown(
                f'<p style="font-weight:700;margin-bottom:4px;">'
                f'<span style="color:{BRAND_AMBER};">&#9632;</span> Start zone</p>',
                unsafe_allow_html=True,
            )
            if st.button("Reset start zone", key=f"{key_prefix}_sx_reset"):
                st.session_state[f"{key_prefix}_sx"] = (0, 100)
                st.session_state[f"{key_prefix}_sy"] = (0, 100)
            sx_range = st.slider("Start X", 0, 100, (0, 100), key=f"{key_prefix}_sx")
            sy_range = st.slider("Start Y", 0, 100, (0, 100), key=f"{key_prefix}_sy")
            bounds.update(
                start_x_min=float(sx_range[0]), start_x_max=float(sx_range[1]),
                start_y_min=float(sy_range[0]), start_y_max=float(sy_range[1]),
            )
        else:
            bounds.update(start_x_min=0.0, start_x_max=100.0, start_y_min=0.0, start_y_max=100.0)

    with end_col:
        if has_end:
            st.markdown(
                f'<p style="font-weight:700;margin-bottom:4px;">'
                f'<span style="color:{BRAND_RED};">&#9632;</span> End zone</p>',
                unsafe_allow_html=True,
            )
            if st.button("Reset end zone", key=f"{key_prefix}_ex_reset"):
                st.session_state[f"{key_prefix}_ex"] = (0, 100)
                st.session_state[f"{key_prefix}_ey"] = (0, 100)
            ex_range = st.slider("End X", 0, 100, (0, 100), key=f"{key_prefix}_ex")
            ey_range = st.slider("End Y", 0, 100, (0, 100), key=f"{key_prefix}_ey")
            bounds.update(
                end_x_min=float(ex_range[0]), end_x_max=float(ex_range[1]),
                end_y_min=float(ey_range[0]), end_y_max=float(ey_range[1]),
            )

    # Pitch preview: full width below the sliders
    fig = go.Figure()
    for shape in opta_pitch_shapes():
        fig.add_shape(**shape)
    if has_start:
        fig.add_shape(type="rect",
                      x0=sx_range[0], y0=sy_range[0], x1=sx_range[1], y1=sy_range[1],
                      xref="x", yref="y",
                      fillcolor="rgba(250,165,26,0.15)", line={"color": BRAND_AMBER, "width": 2})
    if has_end:
        fig.add_shape(type="rect",
                      x0=ex_range[0], y0=ey_range[0], x1=ex_range[1], y1=ey_range[1],
                      xref="x", yref="y",
                      fillcolor="rgba(229,32,47,0.15)", line={"color": BRAND_RED, "width": 2})
    fig.update_layout(
        height=260, plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        margin={"l": 2, "r": 2, "t": 2, "b": 2},
        xaxis={"range": [-2, 102], "showgrid": False, "zeroline": False,
               "showticklabels": False, "scaleanchor": "y", "scaleratio": 105 / 68},
        yaxis={"range": [-2, 102], "showgrid": False, "zeroline": False, "showticklabels": False},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key_prefix)

    return bounds

