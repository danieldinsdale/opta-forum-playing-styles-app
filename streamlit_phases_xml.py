"""
Streamlit app for searching Phases of Play and Player Runs feeds.

Loads JSON game files from the local feeds directory and lets the user
search for phases/runs matching various criteria.

Run with:
    streamlit run streamlit_phases_xml.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import tomli

# ──────────────────────────────────────────────────────────────────────────────
# Feeds directory configuration
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent
_FEEDS_BASE = _REPO_ROOT / "feeds"

# ──────────────────────────────────────────────────────────────────────────────
# Brand colours & fonts — loaded from .streamlit/config.toml so the palette
# and typefaces can be changed in one place without touching Python code.
# ──────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Read the full .streamlit/config.toml and return it as a dict."""
    _config_path = _REPO_ROOT / ".streamlit" / "config.toml"
    try:
        with open(_config_path, "rb") as _f:
            return tomli.load(_f)
    except Exception:
        return {}

_CFG = _load_config()

_BRAND = _CFG.get("brand", {})
_BRAND_PRIMARY = _BRAND.get("primary", "#222222")
_BRAND_PURPLE  = _BRAND.get("purple",  "#9E07AE")
_BRAND_AMBER   = _BRAND.get("amber",   "#FAA51A")
_BRAND_ORANGE  = _BRAND.get("orange",  "#F06424")
_BRAND_RED     = _BRAND.get("red",     "#E5202F")

_FONTS = _CFG.get("fonts", {})
_FONT_HEADLINE   = _FONTS.get("headline_font",    "Barlow Condensed")   # Black Compressed
_FONT_WIDE       = _FONTS.get("wide_font",         "Barlow")             # ExtraBold Wide
_FONT_TITLE      = _FONTS.get("title_font",        "Barlow")             # SemiBold Regular
_FONT_BODY       = _FONTS.get("body_font",         "Barlow")             # Semilight Regular
_FONT_GFX_URL    = _FONTS.get(
    "google_fonts_url",
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@900"
    "&family=Barlow:wght@300;400;600;800&display=swap",
)

_COMPETITION_DIRS: list[str] = (
    sorted(p.name for p in _FEEDS_BASE.iterdir() if p.is_dir())
    if _FEEDS_BASE.exists()
    else []
)


# ──────────────────────────────────────────────────────────────────────────────
# Game discovery
# ──────────────────────────────────────────────────────────────────────────────


def discover_available_games(competition_id: str | None = None) -> list[dict]:
    """Return games that have both phase and run JSON files.

    If *competition_id* is given only that sub-folder of ``feeds/`` is searched;
    otherwise every competition sub-folder is searched.

    Supports two file naming conventions:
    - ``<game_id>_phase.json`` / ``<game_id>_run.json``  (suffixed form)
    - ``<game_id>.json``                                  (plain form, same stem in both dirs)

    Expected layout::

        feeds/
            <competition_id>/
                remote/non_aggregated/phases/<game_id>_phase.json
                remote/non_aggregated/runs/<game_id>_run.json
    """
    results: list[dict] = []
    if not _FEEDS_BASE.exists():
        return results

    comp_dirs = [competition_id] if competition_id else _COMPETITION_DIRS
    for comp_id in comp_dirs:
        phases_dir = _FEEDS_BASE / comp_id / "remote" / "non_aggregated" / "phases"
        runs_dir   = _FEEDS_BASE / comp_id / "remote" / "non_aggregated" / "runs"
        if not phases_dir.exists() or not runs_dir.exists():
            continue

        # Build game_id → path maps, stripping _phase / _run suffixes when present
        def _game_map(directory: Path, suffix: str) -> dict[str, Path]:
            mapping: dict[str, Path] = {}
            for p in directory.glob("*.json"):
                stem = p.stem
                if stem.endswith(suffix):
                    gid = stem[: -len(suffix)]
                else:
                    gid = stem
                mapping[gid] = p
            return mapping

        phase_map = _game_map(phases_dir, "_phase")
        run_map   = _game_map(runs_dir,   "_run")

        for game_id in sorted(phase_map.keys() & run_map.keys()):
            results.append({
                "game_id":        game_id,
                "competition_id": comp_id,
                "phases_path":    str(phase_map[game_id]),
                "runs_path":      str(run_map[game_id]),
            })
    return results


def _peek_description(phases_path: str) -> str:
    """Fast-read the matchInfo.description from a phases JSON (only first 2 KB)."""
    try:
        with open(phases_path, encoding="utf-8") as f:
            head = f.read(2048)
        m = re.search(r'"description"\s*:\s*"([^"]+)"', head)
        if m:
            return m.group(1)
        with open(phases_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("matchInfo", {}).get("description", "")
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_squad_map(competition_id: str) -> dict[str, str]:
    """Load squad_lists.json for a competition and return a playerId → name mapping.

    Name preference: knownName > shortFirstName + " " + shortLastName.
    """
    squad_path = _FEEDS_BASE / competition_id / "squad_lists.json"
    if not squad_path.exists():
        return {}
    try:
        with open(squad_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    player_map: dict[str, str] = {}
    for squad_entry in data.get("squad", []):
        for person in squad_entry.get("person", []):
            pid = person.get("id", "")
            if not pid:
                continue
            known = person.get("knownName", "")
            if known:
                player_map[pid] = known
            else:
                first = person.get("shortFirstName", "")
                last  = person.get("shortLastName", "")
                player_map[pid] = f"{first} {last}".strip()
    return player_map


@st.cache_data(show_spinner=False)
def load_team_squad_map(competition_id: str) -> dict[str, dict]:
    """Return contestantId → {name, player_ids} from squad_lists.json.

    Only entries with type == 'player' are included in player_ids.
    Used to build cascading team → player filter dropdowns.
    """
    squad_path = _FEEDS_BASE / competition_id / "squad_lists.json"
    if not squad_path.exists():
        return {}
    try:
        with open(squad_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    team_map: dict[str, dict] = {}
    for squad_entry in data.get("squad", []):
        cid = squad_entry.get("contestantId", "")
        if not cid:
            continue
        team_map[cid] = {
            "name": squad_entry.get("contestantName", cid),
            "player_ids": [
                p["id"]
                for p in squad_entry.get("person", [])
                if p.get("id") and p.get("type") == "player"
            ],
        }
    return team_map


# ──────────────────────────────────────────────────────────────────────────────
# XML parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_phases_xml(file_bytes: bytes) -> tuple[dict, pd.DataFrame]:
    """Parse a remotePhasesOfPlay / inVenuePhasesOfPlay XML file.

    Returns
    -------
    match_info : dict
        Basic match metadata (id, description, contestants mapping).
    phases_df : pd.DataFrame
        One row per phase with columns for possession team, phase label,
        period, duration, and the full phaseSummary stats.
    """
    tree = ET.parse(BytesIO(file_bytes))
    root = tree.getroot()

    # ── Match info ────────────────────────────────────────────────────────
    match_info_el = root.find("matchInfo")
    match_id = match_info_el.get("id", "unknown") if match_info_el is not None else "unknown"
    description = ""
    if match_info_el is not None:
        desc_el = match_info_el.find("description")
        if desc_el is not None and desc_el.text:
            description = desc_el.text

    # Build contestant id → name mapping
    contestant_map: dict[str, str] = {}
    if match_info_el is not None:
        contestants_el = match_info_el.find("contestants")
        if contestants_el is not None:
            for c in contestants_el.findall("contestant"):
                contestant_map[c.get("id", "")] = c.get("name", c.get("officialName", ""))

    match_info = {
        "match_id": match_id,
        "description": description,
        "contestant_map": contestant_map,
    }

    # ── Phases ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    phase_by_phase = root.find(".//phaseByPhase")
    if phase_by_phase is None:
        return match_info, pd.DataFrame()

    for phase in phase_by_phase.findall("phase"):
        row: dict = {
            "phase_id": phase.get("id"),
            "possessionContestantId": phase.get("possessionContestantId"),
            "periodId": phase.get("periodId"),
            "startTime": int(phase.get("startTime", 0)),
            "endTime": int(phase.get("endTime", 0)),
            "initiatorPlayerId": phase.get("initiatorPlayerId"),
            "firstTouchPlayerId": phase.get("firstTouchPlayerId"),
            "firstTouchEventId": phase.get("firstTouchEventId"),
        }

        # Frame range (used to link runs ↔ phases)
        sf = phase.get("startFrame")
        if sf is not None:
            row["startFrame"] = int(sf)
        ef = phase.get("endFrame")
        if ef is not None:
            row["endFrame"] = int(ef)

        # Start / end X/Y – try phase-level attrs first, then child elements
        for _coord, _child_tag in (
            ("startX", "start"), ("startY", "start"),
            ("endX",   "end"),   ("endY",   "end"),
        ):
            v = phase.get(_coord)
            if v is not None:
                row[_coord] = float(v)
            else:
                child_el = phase.find(_child_tag)
                if child_el is not None:
                    v = child_el.get(_coord)
                    if v is not None:
                        row[_coord] = float(v)

        # Phase label
        label_el = phase.find("phaseLabel")
        row["phaseLabel"] = label_el.get("value") if label_el is not None else "Unknown"

        # Phase summary stats
        summary_el = phase.find("phaseSummary")
        if summary_el is not None:
            for stat in summary_el.findall("stat"):
                stat_type = stat.get("type", "")
                stat_value = stat.text
                if stat_value is not None:
                    # Try numeric conversion
                    try:
                        row[stat_type] = float(stat_value)
                    except ValueError:
                        row[stat_type] = stat_value

        # Overload summary — present when the phase contains an overload
        overload_el = phase.find("overloadSummary")
        row["containsOverload"] = overload_el is not None and len(list(overload_el)) > 0

        rows.append(row)

    phases_df = pd.DataFrame(rows)

    # Map contestant id to name
    if not phases_df.empty and contestant_map:
        phases_df["team_name"] = phases_df["possessionContestantId"].map(contestant_map)

    # Compute duration in seconds from phaseDuration (ms) or fallback
    if "phaseDuration" in phases_df.columns:
        phases_df["duration_seconds"] = phases_df["phaseDuration"] / 1000.0
    else:
        phases_df["duration_seconds"] = (phases_df["endTime"] - phases_df["startTime"]) / 1000.0

    return match_info, phases_df


# ──────────────────────────────────────────────────────────────────────────────
# Runs XML parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_runs_xml(file_bytes: bytes) -> pd.DataFrame:
    """Parse a remotePlayerRuns XML file using iterparse."""
    rows: list[dict] = []

    for _event, elem in ET.iterparse(BytesIO(file_bytes), events=("end",)):
        if elem.tag != "run":
            continue

        row: dict = {
            "run_id": elem.get("id"),
            "playerId": elem.get("playerId"),
            "contestantId": elem.get("contestantId"),
            "runType": elem.get("type"),
            "periodId": elem.get("periodId"),
        }

        sf = elem.get("startFrame")
        if sf is not None:
            row["startFrame"] = int(sf)
        ef = elem.get("endFrame")
        if ef is not None:
            row["endFrame"] = int(ef)

        for attr in ("startX", "startY", "endX", "endY"):
            v = elem.get(attr)
            if v is not None:
                row[attr] = float(v)

        labels_el = elem.find("labels")
        if labels_el is not None:
            for lbl in labels_el.findall("label"):
                if lbl.get("type") == "master":
                    row["masterLabel"] = lbl.get("value", "")
                    break

        quals_el = elem.find("qualifiers")
        if quals_el is not None:
            for q in quals_el.findall("qualifier"):
                qtype = q.get("type", "")
                if qtype == "defensiveLineBroken":
                    v = q.get("value")
                    if v is not None:
                        row["defensiveLineBroken"] = float(v)
                elif qtype == "dangerous":
                    v = q.get("value")
                    if v is not None:
                        row["dangerous"] = float(v)
                elif qtype == "expectedThreat":
                    v = q.get("max")
                    if v is not None:
                        row["expectedThreat_max"] = float(v)

        rows.append(row)
        elem.clear()

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_phases_json(data: dict) -> tuple[dict, pd.DataFrame]:
    """Parse a phases JSON dict (same schema as remotePhasesOfPlay JSON feed).

    Returns the same (match_info, phases_df) as parse_phases_xml.
    """
    mi = data.get("matchInfo", {})
    match_id    = mi.get("id", "unknown")
    description = mi.get("description", "")

    contestant_map: dict[str, str] = {}
    for c in mi.get("contestant", []):
        contestant_map[c.get("id", "")] = c.get("name", c.get("officialName", ""))

    match_info = {
        "match_id":       match_id,
        "description":    description,
        "contestant_map": contestant_map,
    }

    ld  = data.get("liveData", {})
    pbp = ld.get("phaseByPhase", {})
    phases_raw = pbp.get("phase", [])

    rows: list[dict] = []
    for phase in phases_raw:
        row: dict = {
            "phase_id":               str(phase.get("id", "")),
            "possessionContestantId": phase.get("possessionContestantId"),
            "periodId":               str(phase.get("periodId", "")),
            "startTime":              int(phase.get("startTime", 0)),
            "endTime":                int(phase.get("endTime", 0)),
            "initiatorPlayerId":      phase.get("initiatorPlayerId"),
            "firstTouchPlayerId":     phase.get("firstTouchPlayerId"),
            "firstTouchEventId":      phase.get("firstTouchEventId"),
        }

        for key in ("startFrame", "endFrame"):
            v = phase.get(key)
            if v is not None:
                row[key] = int(v)

        for coord in ("startX", "startY", "endX", "endY"):
            v = phase.get(coord)
            if v is not None:
                try:
                    row[coord] = float(v)
                except (ValueError, TypeError):
                    pass

        # phaseLabel
        pl = phase.get("phaseLabel", {})
        row["phaseLabel"] = pl.get("value", "Unknown") if isinstance(pl, dict) else str(pl)

        # phaseSummary stats
        ps = phase.get("phaseSummary", {})
        for stat in ps.get("stat", []):
            stype = stat.get("type", "")
            sval  = stat.get("value")
            if stype and sval is not None:
                try:
                    row[stype] = float(sval)
                except (ValueError, TypeError):
                    row[stype] = sval

        # Overload summary — present when the phase contains an overload
        overload = phase.get("overloadSummary", {})
        row["containsOverload"] = bool(overload and overload.get("overload"))

        rows.append(row)

    phases_df = pd.DataFrame(rows)

    if not phases_df.empty and contestant_map:
        phases_df["team_name"] = phases_df["possessionContestantId"].map(contestant_map)

    if "phaseDuration" in phases_df.columns:
        phases_df["duration_seconds"] = phases_df["phaseDuration"] / 1000.0
    elif not phases_df.empty:
        phases_df["duration_seconds"] = (phases_df["endTime"] - phases_df["startTime"]) / 1000.0

    return match_info, phases_df


def parse_runs_json(data: dict) -> pd.DataFrame:
    """Parse a runs JSON dict (same schema as remotePlayerRuns JSON feed)."""
    mi = data.get("matchInfo", {})
    contestant_map: dict[str, str] = {}
    for c in mi.get("contestant", []):
        contestant_map[c.get("id", "")] = c.get("name", c.get("officialName", ""))

    runs_raw = data.get("liveData", {}).get("runByRun", {}).get("run", [])

    rows: list[dict] = []
    for run in runs_raw:
        row: dict = {
            "run_id":       str(run.get("id", "")),
            "playerId":     run.get("playerId"),
            "contestantId": run.get("contestantId"),
            "runType":      run.get("type"),
            "periodId":     str(run.get("periodId", "")),
        }

        for key in ("startFrame", "endFrame"):
            v = run.get(key)
            if v is not None:
                row[key] = int(v)

        for attr in ("startX", "startY", "endX", "endY"):
            v = run.get(attr)
            if v is not None:
                try:
                    row[attr] = float(v)
                except (ValueError, TypeError):
                    pass

        # startTime (keep as int ms for hover display)
        st_val = run.get("startTime")
        if st_val is not None:
            try:
                row["startTime"] = int(st_val)
            except (ValueError, TypeError):
                pass

        # master label
        labels_obj = run.get("labels", {})
        label_list = labels_obj.get("label", []) if isinstance(labels_obj, dict) else []
        if isinstance(label_list, dict):
            label_list = [label_list]
        for lbl in label_list:
            if isinstance(lbl, dict) and lbl.get("type") == "master":
                row["masterLabel"] = lbl.get("value", "")
                break

        # qualifiers
        quals_obj  = run.get("qualifiers", {})
        qual_list  = quals_obj.get("qualifier", []) if isinstance(quals_obj, dict) else []
        if isinstance(qual_list, dict):
            qual_list = [qual_list]
        for q in qual_list:
            if not isinstance(q, dict):
                continue
            qtype = q.get("type", "")
            if qtype == "defensiveLineBroken":
                v = q.get("value")
                if v is not None:
                    row["defensiveLineBroken"] = float(v)
            elif qtype == "dangerous":
                v = q.get("value")
                if v is not None:
                    row["dangerous"] = float(v)
            elif qtype == "expectedThreat":
                v = q.get("max")
                if v is not None:
                    row["expectedThreat_max"] = float(v)

        if contestant_map and row.get("contestantId"):
            row["team_name"] = contestant_map.get(row["contestantId"], "")

        rows.append(row)

    return pd.DataFrame(rows)


def _load_game(game_meta: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load and parse a single game's phases + runs JSON files.

    Returns (match_info, phases_df, runs_df).
    """
    with open(game_meta["phases_path"], encoding="utf-8") as f:
        phases_data = json.load(f)
    with open(game_meta["runs_path"], encoding="utf-8") as f:
        runs_data = json.load(f)

    match_info, phases_df = parse_phases_json(phases_data)
    runs_df = parse_runs_json(runs_data)

    # Tag each row with game context for multi-game merges
    game_id = game_meta["game_id"]
    desc    = match_info.get("description", game_id)
    if not phases_df.empty:
        phases_df["game_id"]          = game_id
        phases_df["match_description"] = desc
    if not runs_df.empty:
        runs_df["game_id"]            = game_id
        runs_df["match_description"]  = desc
        # Composite key: run_id values are simple integers scoped per game,
        # so we need a globally unique identifier when multiple games are loaded.
        runs_df["composite_run_id"] = game_id + "_" + runs_df["run_id"].astype(str)

    return match_info, phases_df, runs_df


# ──────────────────────────────────────────────────────────────────────────────
# Pitch drawing helpers
# ──────────────────────────────────────────────────────────────────────────────


def _opta_pitch_shapes() -> list[dict]:
    """Plotly shapes for an Opta 0-100 x 0-100 pitch (105m × 68m)."""
    line = {"type": "line",   "line": {"color": "#aaaaaa", "width": 1.5}, "xref": "x", "yref": "y"}
    rect = {"type": "rect",   "line": {"color": "#aaaaaa", "width": 1.5}, "fillcolor": "rgba(0,0,0,0)", "xref": "x", "yref": "y"}
    circ = {"type": "circle", "line": {"color": "#aaaaaa", "width": 1.5}, "fillcolor": "rgba(0,0,0,0)", "xref": "x", "yref": "y"}

    def ln(x0, y0, x1, y1): return {**line, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    def bx(x0, y0, x1, y1): return {**rect, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    def cl(x0, y0, x1, y1): return {**circ, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    def ox(m): return m / 105 * 100
    def oy(m): return m / 68 * 100

    pa_d  = ox(16.5);  pa_y0 = oy((68 - 40.32) / 2);  pa_y1 = oy((68 + 40.32) / 2)
    sb_d  = ox(5.5);   sb_y0 = oy((68 - 18.32) / 2);   sb_y1 = oy((68 + 18.32) / 2)
    gl_d  = ox(2.44);  gl_y0 = oy((68 - 7.32) / 2);    gl_y1 = oy((68 + 7.32) / 2)
    ps_x  = ox(11.0)
    cc_rx = ox(9.15);  cc_ry = oy(9.15);  cc_cx = 50.0;  cc_cy = 50.0
    arc_rx = ox(9.15); arc_ry = oy(9.15)

    return [
        bx(0, 0, 100, 100),
        ln(50, 0, 50, 100),
        cl(cc_cx - cc_rx, cc_cy - cc_ry, cc_cx + cc_rx, cc_cy + cc_ry),
        ln(cc_cx - 0.3, cc_cy, cc_cx + 0.3, cc_cy),
        bx(0, pa_y0, pa_d, pa_y1),
        bx(0, sb_y0, sb_d, sb_y1),
        ln(ps_x - 0.3, 50, ps_x + 0.3, 50),
        cl(ps_x - arc_rx, cc_cy - arc_ry, ps_x + arc_rx, cc_cy + arc_ry),
        bx(100 - pa_d, pa_y0, 100, pa_y1),
        bx(100 - sb_d, sb_y0, 100, sb_y1),
        ln(100 - ps_x - 0.3, 50, 100 - ps_x + 0.3, 50),
        cl(100 - ps_x - arc_rx, cc_cy - arc_ry, 100 - ps_x + arc_rx, cc_cy + arc_ry),
        bx(-gl_d, gl_y0, 0, gl_y1),
        bx(100, gl_y0, 100 + gl_d, gl_y1),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# VOD streaming helper
# ──────────────────────────────────────────────────────────────────────────────


def _get_vod_api_key() -> str:
    """Return the VOD API key from env var, Streamlit secrets, or sidebar input.
    Never logs or echoes the value."""
    return (
        os.environ.get("VOD_API_KEY", "")
        or st.secrets.get("VOD_API_KEY", "")
        or st.session_state.get("_vod_k", "")
    )


def _get_vod_base_url() -> str:
    """Return the VOD base URL from env var or Streamlit secrets only (no hardcoded fallback)."""
    return (
        os.environ.get("VOD_BASE_URL", "")
        or st.secrets.get("VOD_BASE_URL", "")
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_vod_streaming(game_uuid: str, period: int, time_in: int, time_out: int,
                      before_time: int = 0, after_time: int = 0,
                      api_key: str = "") -> str:
    """Retrieve a streaming URL from the VOD StreamingLinks API.

    Parameters
    ----------
    game_uuid : str
        Match UUID.
    period : int
        Match period (half).
    time_in, time_out : int
        Elapsed match time **in seconds** (e.g. 2nd-half kick-off = 2700).
    before_time, after_time : int
        Pre-/post-buffer in seconds.
    api_key : str
        VOD API key (passed explicitly so the cached function is key-aware).

    Returns
    -------
    str
        The streaming URL for the clip.
    """
    if not api_key:
        raise ValueError("VOD API key is not configured. Set the VOD_API_KEY environment variable or enter it in the sidebar.")

    base_url = _get_vod_base_url()
    if not base_url:
        raise ValueError("VOD base URL is not configured. Set the VOD_BASE_URL environment variable or Streamlit secret.")

    payload = {
        "fixtureUuid": game_uuid,
        "periodId": period,
        "timeIn": time_in,
        "timeOut": time_out,
        "preRoll": -before_time,
        "postRoll": after_time,
    }

    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "APIKey": api_key,
        "Content-Type": "application/json-patch+json",
    }

    try:
        res = requests.post(base_url, data=json.dumps(payload), headers=headers, timeout=15)
        res.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        # Re-raise with only the status code/reason — strip the URL from the message
        raise requests.exceptions.HTTPError(
            f"{exc.response.status_code} {exc.response.reason}",
            response=exc.response,
        ) from None
    except requests.exceptions.RequestException as exc:
        # Strip any URL/connection details from network-level errors
        raise requests.exceptions.RequestException(type(exc).__name__) from None

    url = json.loads(res.text)[0].get("http", "").encode("ascii", "ignore").decode()
    if not url:
        raise ValueError("The API returned an empty streaming URL.")
    return url


def _render_runs_pitch_map(result_df: pd.DataFrame, match_info: dict, squad_map: dict[str, str] | None = None) -> None:
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

    palette = [_BRAND_PURPLE, _BRAND_ORANGE, _BRAND_AMBER, "#222222", "#aaaaaa", "#555555"]
    # Colour by team when more than one team is present in the (already-filtered) data
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

    plot_df["_start_mmss"] = plot_df["startTime"].apply(_ms_to_mmss)
    if "_player_name" not in plot_df.columns:
        plot_df["_player_name"] = plot_df.get("playerId", "")

    fig = go.Figure()
    for shape in _opta_pitch_shapes():
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
            # Draw arrows as line segments + arrowhead markers grouped by colour,
            # avoiding thousands of individual add_annotation calls.
            arrow_colours = arr_df["_colour"].unique()
            for colour in arrow_colours:
                grp_a = arr_df[arr_df["_colour"] == colour]
                sx = grp_a["startX"].values
                ex = grp_a["endX"].values
                sy = grp_a["startY"].values
                ey = grp_a["endY"].values
                # Interleave start→end→None for disconnected line segments
                xs = np.empty(len(sx) * 3, dtype=object)
                xs[0::3] = sx
                xs[1::3] = ex
                xs[2::3] = None
                ys = np.empty(len(sy) * 3, dtype=object)
                ys[0::3] = sy
                ys[1::3] = ey
                ys[2::3] = None
                fig.add_trace(go.Scatter(
                    x=xs.tolist(), y=ys.tolist(),
                    mode="lines",
                    line={"color": colour, "width": 1.5},
                    hoverinfo="skip",
                    showlegend=False,
                ))
                # Arrowhead markers: interleave start (size=0, invisible) then end
                # (size=10, visible arrow) so "angleref": "previous" computes the
                # correct angle from start→end for every arrowhead.
                mx = np.empty(len(sx) * 2, dtype=object)
                mx[0::2] = sx
                mx[1::2] = ex
                my = np.empty(len(sy) * 2, dtype=object)
                my[0::2] = sy
                my[1::2] = ey
                msizes = np.tile([0, 10], len(sx))
                msymbols = np.tile(["circle", "arrow"], len(sx))
                fig.add_trace(go.Scatter(
                    x=mx.tolist(), y=my.tolist(),
                    mode="markers",
                    marker={
                        "symbol": msymbols.tolist(),
                        "color": colour,
                        "size": msizes.tolist(),
                        "angleref": "previous",
                        "line": {"width": 0},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                ))

    groups = plot_df.groupby("team_name") if colour_by_team else [("Runs", plot_df)]
    for label, grp in groups:
        colour = team_colour.get(str(label), palette[0]) if colour_by_team else palette[0]
        fig.add_trace(go.Scatter(
            x=grp["startX"], y=grp["startY"], mode="markers", name=str(label),
            marker={"color": colour, "size": 9, "line": {"color": "white", "width": 1}},
            customdata=grp[custom_cols].values, hovertemplate=hover_tmpl,
            hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": "#9E07AE", "font": {"size": 12}},
        ))

    fig.update_layout(
        height=700, plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        xaxis={"range": [-4, 104], "showgrid": False, "zeroline": False, "showticklabels": False, "scaleanchor": "y", "scaleratio": 105 / 68},
        yaxis={"range": [-4, 104], "showgrid": False, "zeroline": False, "showticklabels": False},
        showlegend=colour_by_team,
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(0,0,0,0.6)", "font": {"color": "#f0f0f0"}},
        title={"text": f"{len(plot_df)} run(s) plotted", "font": {"size": 13, "color": "#f0f0f0"}, "x": 0.5},
        hoverlabel={"bgcolor": "#1a1a1a", "bordercolor": "#9E07AE", "font": {"size": 12, "color": "#f0f0f0"}, "align": "left", "namelength": 0},
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, key="runs_pitch_map")


# ──────────────────────────────────────────────────────────────────────────────
# Pitch zone selector widget
# ──────────────────────────────────────────────────────────────────────────────


def _pitch_zone_selector(key_prefix: str, has_start: bool = True, has_end: bool = True) -> dict:
    """Render start/end zone sliders side-by-side, with the pitch preview below.

    Sliders and returned bounds all use the 0–100 Opta coordinate space.
    """
    bounds: dict = {}

    # ── Sliders: start (left col) and end (right col) ─────────────────────
    start_col, end_col = st.columns(2)

    with start_col:
        if has_start:
            st.markdown(
                f'<p style="font-weight:700;margin-bottom:4px;">'
                f'<span style="color:{_BRAND_AMBER};">&#9632;</span> Start zone</p>',
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
                f'<span style="color:{_BRAND_RED};">&#9632;</span> End zone</p>',
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


    # ── Pitch preview: full width below the sliders ───────────────────────
    fig = go.Figure()
    for shape in _opta_pitch_shapes():
        fig.add_shape(**shape)
    if has_start:
        fig.add_shape(type="rect",
                      x0=sx_range[0], y0=sy_range[0], x1=sx_range[1], y1=sy_range[1],
                      xref="x", yref="y",
                      fillcolor="rgba(250,165,26,0.15)", line={"color": _BRAND_AMBER, "width": 2})
    if has_end:
        fig.add_shape(type="rect",
                      x0=ex_range[0], y0=ey_range[0], x1=ex_range[1], y1=ey_range[1],
                      xref="x", yref="y",
                      fillcolor="rgba(229,32,47,0.15)", line={"color": _BRAND_RED, "width": 2})
    fig.update_layout(
        height=260, plot_bgcolor="#0d0d0d", paper_bgcolor="#000000",
        margin={"l": 2, "r": 2, "t": 2, "b": 2},
        xaxis={"range": [-2, 102], "showgrid": False, "zeroline": False, "showticklabels": False, "scaleanchor": "y", "scaleratio": 105 / 68},
        yaxis={"range": [-2, 102], "showgrid": False, "zeroline": False, "showticklabels": False},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key_prefix)

    return bounds


# ──────────────────────────────────────────────────────────────────────────────
# Runs ↔ Phases analysis
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Searching runs…")
def _compute_runs_result(
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
    et_lo: float,
    et_hi: float,
    run_coord_bounds: tuple | None,
    selected_team_name: str,
    selected_player_name: str,
    squad_map: tuple,
) -> pd.DataFrame:
    """Heavy phase↔run interval join + all filters. Cached so that changing
    display-only widgets (view mode, top-N, etc.) does not re-trigger work."""

    # Reconstruct mutable types from hashable args
    squad_map_dict: dict = dict(squad_map)
    coord_bounds: dict | None = dict(run_coord_bounds) if run_coord_bounds is not None else None
    et_col = "expectedThreat_max"
    _et_same = (
        not (et_col in runs_df.columns and runs_df[et_col].notna().any())
        or runs_df[et_col].min(skipna=True) == runs_df[et_col].max(skipna=True)
    )

    # ── Apply Phase Criteria ──────────────────────────────────────────────
    _phase_label_filter = list(selected_labels) if selected_labels else list(available_labels)
    filtered_phases = phases_df[phases_df["phaseLabel"].isin(_phase_label_filter)].copy()
    if includes_shots_choice != "Any" and "includesShots" in filtered_phases.columns:
        filtered_phases = filtered_phases[filtered_phases["includesShots"] == includes_shots_choice]
    if includes_goal_choice != "Any" and "includesGoal" in filtered_phases.columns:
        filtered_phases = filtered_phases[filtered_phases["includesGoal"] == includes_goal_choice]
    if filtered_phases.empty:
        return pd.DataFrame()

    # ── Vectorised phase ↔ run interval join ──────────────────────────────
    ph = filtered_phases.copy()
    ph["_gid"]    = ph["game_id"].astype(str) if "game_id" in ph.columns else ""
    ph["_pid"]    = ph["phase_id"].astype(str)
    ph["_period"] = ph["periodId"].astype(str)
    ph["_team"]   = ph["possessionContestantId"].astype(str)
    ph["_psf"]    = ph["startFrame"].astype(int)
    ph["_pef"]    = ph["endFrame"].astype(int)
    ph_cols = ["_gid", "_pid", "_period", "_team", "_psf", "_pef",
               "phaseLabel", "includesShots", "includesGoal", "startTime"]
    ph_cols = [c for c in ph_cols if c in ph.columns]
    ph_slim = ph[ph_cols].copy()
    for _c in ("includesShots", "includesGoal", "startTime"):
        if _c not in ph_slim.columns:
            ph_slim[_c] = ""

    ru = runs_df.copy()
    ru["_gid"]    = ru["game_id"].astype(str) if "game_id" in ru.columns else ""
    ru["_period"] = ru["periodId"].astype(str)
    ru["_team"]   = ru["contestantId"].astype(str)
    ru["_rsf"]    = ru["startFrame"]
    ru["_ref"]    = ru["endFrame"]
    ru = ru.dropna(subset=["_rsf", "_ref"])
    ru["_rsf"]    = ru["_rsf"].astype(int)
    ru["_ref"]    = ru["_ref"].astype(int)
    ru["_crid"]   = ru["composite_run_id"] if "composite_run_id" in ru.columns else ru["run_id"]

    run_keep_cols = ["_gid", "_period", "_team", "_rsf", "_ref", "_crid",
                     "run_id", "game_id", "periodId", "playerId", "contestantId",
                     "masterLabel", "runType", "defensiveLineBroken",
                     "dangerous", "expectedThreat_max",
                     "startX", "startY", "endX", "endY"]
    run_keep_cols = [c for c in run_keep_cols if c in ru.columns]
    ru_slim = ru[run_keep_cols].copy()

    # Split by runType: inPossession joins on team; outOfPossession joins on opposite team
    if "runType" in ru_slim.columns:
        ru_inp   = ru_slim[ru_slim["runType"] == "inPossession"].copy()
        ru_oop   = ru_slim[ru_slim["runType"] == "outOfPossession"].copy()
        ru_other = ru_slim[~ru_slim["runType"].isin(["inPossession", "outOfPossession"])].copy()
    else:
        ru_inp   = ru_slim.copy()
        ru_oop   = pd.DataFrame(columns=ru_slim.columns)
        ru_other = pd.DataFrame(columns=ru_slim.columns)

    merged_inp = ru_inp.merge(
        ph_slim,
        left_on=["_gid", "_period", "_team"],
        right_on=["_gid", "_period", "_team"],
        how="inner",
    )

    if not ru_oop.empty:
        merged_oop = ru_oop.merge(
            ph_slim.rename(columns={"_team": "_ph_team"}),
            left_on=["_gid", "_period"],
            right_on=["_gid", "_period"],
            how="inner",
        )
        merged_oop = merged_oop[merged_oop["_team"] != merged_oop["_ph_team"]]
        merged_oop = merged_oop.drop(columns=["_ph_team"], errors="ignore")
    else:
        merged_oop = pd.DataFrame(columns=merged_inp.columns if not merged_inp.empty else [])

    if not ru_other.empty:
        merged_other = ru_other.merge(
            ph_slim,
            left_on=["_gid", "_period", "_team"],
            right_on=["_gid", "_period", "_team"],
            how="inner",
        )
    else:
        merged_other = pd.DataFrame(columns=merged_inp.columns if not merged_inp.empty else [])

    merged = pd.concat([merged_inp, merged_oop, merged_other], ignore_index=True)

    if not merged.empty:
        overlap = (merged["_rsf"] < merged["_pef"]) & (merged["_ref"] > merged["_psf"])
        merged = merged[overlap]

    result_df = merged.rename(columns={
        "_crid":  "composite_run_id",
        "_pid":   "phase_id",
        "_rsf":   "run_startFrame",
        "_ref":   "run_endFrame",
    }).drop(columns=["_gid", "_period", "_team", "_psf", "_pef"], errors="ignore")

    for _id_col in ("run_id", "phase_id"):
        if _id_col in result_df.columns:
            result_df[_id_col] = pd.to_numeric(result_df[_id_col], errors="coerce").astype("Int64")

    if result_df.empty:
        return result_df

    if contestant_map and "contestantId" in result_df.columns:
        result_df["team_name"] = result_df["contestantId"].map(contestant_map)

    # ── Apply Run Criteria ────────────────────────────────────────────────
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
    if has_et and not _et_same:
        result_df = result_df[result_df[et_col].isna() | ((result_df[et_col] >= et_lo) & (result_df[et_col] <= et_hi))]

    # ── Apply Run Coordinates filter ──────────────────────────────────────
    if coord_bounds:
        for coord, b_min, b_max in [
            ("startX", coord_bounds["start_x_min"], coord_bounds["start_x_max"]),
            ("startY", coord_bounds["start_y_min"], coord_bounds["start_y_max"]),
            ("endX",   coord_bounds.get("end_x_min", 0),   coord_bounds.get("end_x_max", 100)),
            ("endY",   coord_bounds.get("end_y_min", 0),   coord_bounds.get("end_y_max", 100)),
        ]:
            if coord in result_df.columns and (b_min != 0 or b_max != 100):
                result_df = result_df[result_df[coord].notna() & (result_df[coord] >= b_min) & (result_df[coord] <= b_max)]

    # ── Apply Team / Player filter ────────────────────────────────────────
    if selected_team_name != "All teams":
        _cid_by_name = {contestant_map.get(str(c), str(c)): str(c) for c in (result_df["contestantId"].dropna().unique() if "contestantId" in result_df.columns else [])}
        selected_cid = _cid_by_name.get(selected_team_name, "")
        result_df = result_df[result_df["contestantId"] == selected_cid]
        if selected_player_name != "All players":
            if squad_map_dict:
                matching_pids = {
                    pid for pid in result_df["playerId"].dropna().unique()
                    if squad_map_dict.get(str(pid), str(pid)) == selected_player_name
                }
            else:
                matching_pids = {selected_player_name}
            result_df = result_df[result_df["playerId"].isin(matching_pids)]

    # ── Consolidate: each unique run counted once, phase labels joined ────
    if not result_df.empty:
        phase_cols_set = {"phase_id", "phaseLabel", "startTime", "includesShots", "includesGoal"}

        def _join_unique(x):
            return ", ".join(sorted({str(v) for v in x if pd.notna(v) and str(v) != "nan"}))

        agg_rules = {c: (_join_unique if c in phase_cols_set else "first")
                     for c in result_df.columns if c != "composite_run_id"}
        result_df = result_df.groupby("composite_run_id", as_index=False).agg(agg_rules)
        if contestant_map and "contestantId" in result_df.columns:
            result_df["team_name"] = result_df["contestantId"].map(contestant_map)

    if squad_map_dict and "playerId" in result_df.columns:
        result_df["player_name"] = result_df["playerId"].map(
            lambda x: squad_map_dict.get(str(x), str(x)) if pd.notna(x) else ""
        )

    return result_df


def _analysis_runs_by_phase(phases_df: pd.DataFrame, runs_df: pd.DataFrame, match_info: dict, squad_map: dict | None = None):
    """List all runs that occur during phases matching phaseSummary and run-level criteria."""
    if squad_map is None:
        squad_map = {}
    st.subheader("🏃 Runs Search")
    st.markdown("Use the **Phase Criteria** expander to choose which phases to search within, then open **Run Criteria** to refine by individual run properties.")

    if "startFrame" not in phases_df.columns or "endFrame" not in phases_df.columns:
        st.error("Phase feed is missing startFrame / endFrame – cannot link to runs.")
        return
    if "startFrame" not in runs_df.columns or "endFrame" not in runs_df.columns:
        st.error("Run feed is missing startFrame / endFrame – cannot link to phases.")
        return

    # Pre-compute options from source data (before result_df exists)
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
    _et_input_min = min(_et_min_src, 0.0)
    _et_input_max = max(_et_max_src, 1.0)

    # ── Filters expander (tabs inside — nested expanders not supported) ───
    with st.expander("🔍 Filters", expanded=True):
        ftab_run, ftab_phase, ftab_coords, ftab_team = st.tabs(
            ["🏃 Run Criteria", "📋 Phase Criteria", "📍 Run Coordinates", "👥 Team / Player"]
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
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                selected_run_labels = st.multiselect("Main label", _all_run_master_labels, default=_all_run_master_labels, key="run_master_labels")
                run_type_choice = st.radio("**Run type**", ["Any", "inPossession", "outOfPossession"], horizontal=True, key="run_type_filter")
                dlb_choice = st.radio("**Defensive line broken**", ["Any", "Yes", "No"], horizontal=True, key="run_dlb")
            with r_col2:
                dangerous_choice = st.radio("**Dangerous**", ["Any", "Yes", "No"], horizontal=True, key="run_dangerous")
                if _has_et_src and not _et_same:
                    st.markdown("**Max Expected Threat**")
                    c1, c2 = st.columns(2)
                    with c1:
                        et_lo = st.number_input("Min expected threat", min_value=_et_input_min, max_value=_et_input_max, value=0.0, format="%.4f", key="run_et_min")
                    with c2:
                        et_hi = st.number_input("Max expected threat", min_value=_et_input_min, max_value=_et_input_max, value=1.0, format="%.4f", key="run_et_max")
                else:
                    et_lo = _et_min_src
                    et_hi = _et_max_src

        with ftab_coords:
            run_coord_bounds = _pitch_zone_selector(
                key_prefix="run_coords",
                has_start="startX" in runs_df.columns,
                has_end="endX" in runs_df.columns,
            )

        with ftab_team:
            tf_col1, tf_col2 = st.columns(2)
            with tf_col1:
                selected_team_name = st.selectbox(
                    "Filter by team",
                    options=["All teams"] + _available_run_team_names,
                    key="run_team_filter",
                )
            with tf_col2:
                selected_player_name = "All players"
                if selected_team_name != "All teams":
                    _selected_cid_pre = _run_cid_by_name.get(selected_team_name, "")
                    _team_run_pids = sorted(
                        runs_df.loc[runs_df["contestantId"] == _selected_cid_pre, "playerId"]
                        .dropna().unique()
                    ) if "contestantId" in runs_df.columns else []
                    if squad_map:
                        _player_options_pre = sorted({squad_map.get(str(p), str(p)) for p in _team_run_pids})
                    else:
                        _player_options_pre = sorted(str(p) for p in _team_run_pids)
                    selected_player_name = st.selectbox(
                        "Filter by player",
                        options=["All players"] + _player_options_pre,
                        key="run_player_filter",
                    )

    # ── Generate Outputs button ───────────────────────────────────────────
    if st.button("▶ Generate Outputs", type="primary", key="runs_generate"):
        st.session_state["runs_committed"] = {
            "selected_labels":      selected_labels,
            "includes_shots":       includes_shots_choice,
            "includes_goal":        includes_goal_choice,
            "selected_run_labels":  selected_run_labels,
            "run_type_choice":      run_type_choice,
            "dlb_choice":           dlb_choice,
            "dangerous_choice":     dangerous_choice,
            "et_lo":                et_lo,
            "et_hi":                et_hi,
            "run_coord_bounds":     run_coord_bounds,
            "selected_team_name":   selected_team_name,
            "selected_player_name": selected_player_name,
        }

    committed = st.session_state.get("runs_committed")
    if not committed:
        st.info("Set your filters above and click **▶ Generate Outputs** to run the analysis.")
        return

    # ── Call cached computation ───────────────────────────────────────────
    # Convert mutable types to hashable equivalents for the cache key
    _coord_bounds_hashable = tuple(sorted(committed["run_coord_bounds"].items())) if committed["run_coord_bounds"] else None
    result_df = _compute_runs_result(
        phases_df=phases_df,
        runs_df=runs_df,
        contestant_map=_run_cmap,
        available_labels=tuple(available_labels),
        selected_labels=tuple(committed["selected_labels"]),
        includes_shots_choice=committed["includes_shots"],
        includes_goal_choice=committed["includes_goal"],
        selected_run_labels=tuple(committed["selected_run_labels"]),
        run_type_choice=committed["run_type_choice"],
        dlb_choice=committed["dlb_choice"],
        dangerous_choice=committed["dangerous_choice"],
        et_lo=float(committed["et_lo"]),
        et_hi=float(committed["et_hi"]),
        run_coord_bounds=_coord_bounds_hashable,
        selected_team_name=committed["selected_team_name"],
        selected_player_name=committed["selected_player_name"],
        squad_map=tuple(sorted(squad_map.items())),
    )

    if result_df.empty:
        # Distinguish between "no phases matched" (columns absent) and "no runs found"
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
        display_cols = ["game_id", "run_id", "phase_id", "masterLabel", "phaseLabel"]
        if "team_name" in result_df.columns:
            display_cols.append("team_name")
        if "player_name" in result_df.columns:
            display_cols.append("player_name")
        elif "playerId" in result_df.columns:
            display_cols.append("playerId")
        display_cols = [c for c in display_cols if c in result_df.columns]
        st.dataframe(result_df[display_cols].reset_index(drop=True), use_container_width=True, height=min(600, max(200, len(result_df) * 38)))

    elif view_mode == "Pitch map":
        _render_runs_pitch_map(result_df, match_info, squad_map)

    else:
        if group_by == "Team":
            group_cols = ["team_name"] if "team_name" in result_df.columns else ["contestantId"]
            label_col = group_cols[0]
            id_col = None
        else:
            group_cols = ["playerId", "team_name"] if "team_name" in result_df.columns else ["playerId"]
            label_col = "playerId"
            id_col = "playerId"

        agg_df = (
            result_df.groupby(group_cols, dropna=False)
            .agg(total_runs=("run_id", "count"))
            .sort_values("total_runs", ascending=False)
            .reset_index()
        )
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
            fig = px.bar(plot_df_agg, x="total_runs", y=display_label_col, orientation="h", text="total_runs",
                         labels={"total_runs": "Total Runs", display_label_col: group_by}, height=max(300, len(plot_df_agg) * 36 + 80))
            fig.update_traces(textposition="outside")
            fig.update_layout(margin={"l": 10, "r": 40, "t": 30, "b": 10}, yaxis={"automargin": True})
            st.plotly_chart(fig, use_container_width=True, key="runs_agg_bar")


# ──────────────────────────────────────────────────────────────────────────────
# Phase search / analysis
# ──────────────────────────────────────────────────────────────────────────────


def _analysis_phase_analysis(phases_df: pd.DataFrame, match_info: dict, squad_map: dict[str, str] | None = None):
    """Merged Phase Search + Phase Analysis: shared filters, two result views."""
    if squad_map is None:
        squad_map = {}
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
        pa_ftab_labels, pa_ftab_coords, pa_ftab_counts, pa_ftab_outcomes, pa_ftab_team = st.tabs(
            ["🏷️ Phase Labels", "📍 Coordinates", "⚡ Action Counts", "🎯 Attacking Outcomes", "👥 Team / Player"]
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
                    f'<small><span style="color:{_BRAND_AMBER};">&#9632;</span> <b>Start zone</b> filters on the <b>start</b> of the 1st phase. &nbsp; '
                    f'<span style="color:{_BRAND_RED};">&#9632;</span> <b>End zone</b> filters on the <b>end</b> of the 2nd (Leads to) phase.</small>',
                    unsafe_allow_html=True,
                )
            pa_coord_bounds = _pitch_zone_selector(
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
            for idx, (col_key, lbl, widget_key) in enumerate([
                ("includesShots",   "Includes Shots",    "pa_shots_filter"),
                ("includesGoal",    "Includes Goal",     "pa_goal_filter"),
                ("containsOverload","Contains Overload", "pa_overload_filter"),
            ]):
                if col_key not in phases_df.columns:
                    continue
                container = ao_left if idx % 2 == 0 else ao_right
                with container:
                    choice = st.radio(f"**{lbl}**", ["Any", "True", "False"], horizontal=True, key=widget_key)
                    pa_bool_choices[col_key] = choice

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
            "selected_team_name":   pa_selected_team_name,
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
    pa_selected_team_name  = pa_committed["selected_team_name"]

    # ── Apply filters ─────────────────────────────────────────────────────
    pa_sequence_mode = pa_label_mode == "Leads to (sequence)"

    if not pa_sequence_mode:
        filtered = phases_df.copy()
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
        # Show initiator player name if available
        if "initiatorPlayerId" in filtered.columns and filtered["initiatorPlayerId"].notna().any():
            if squad_map:
                filtered["initiator_name"] = filtered["initiatorPlayerId"].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else "")
                result_cols.append("initiator_name")
            else:
                result_cols.append("initiatorPlayerId")
        result_cols = [c for c in result_cols if c in filtered.columns]

        display_df = filtered[result_cols].reset_index(drop=True).copy()

        # Format startTime / endTime as mm:ss for readability (display only —
        # filtered retains raw milliseconds for video playback)
        def _ms_to_mmss_phase(val) -> str:
            try:
                total_s = int(float(val)) // 1000
                return f"{total_s // 60}:{total_s % 60:02d}"
            except (TypeError, ValueError):
                return str(val)

        for _tc in ("startTime", "endTime"):
            if _tc in display_df.columns:
                display_df[_tc] = display_df[_tc].apply(_ms_to_mmss_phase)

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
                api_key = _get_vod_api_key()
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

        # Pre-build name columns for player aggregation
        if squad_map and has_initiator:
            filtered = filtered.copy()
            filtered["_initiator_name"] = filtered["initiatorPlayerId"].map(lambda x: squad_map.get(str(x), str(x)) if pd.notna(x) else "")
        if squad_map and has_first_touch:
            filtered = filtered.copy() if "_initiator_name" not in filtered.columns else filtered
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
                fig.update_layout(barmode="stack", margin={"l": 10, "r": 50, "t": 50 if chart_title else 30, "b": 10}, yaxis={"automargin": True})
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
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    margin={"l": 10, "r": 50, "t": 50 if chart_title else 30, "b": 10},
                    yaxis={"automargin": True},
                )
            st.plotly_chart(fig, use_container_width=True, key="pa_agg_bar")


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────


def _parse_squad_json(data: dict) -> dict[str, str]:
    """Parse a squad_lists JSON dict into a playerId → name mapping.

    Accepts both top-level ``squad`` arrays (competition-level feed) and a
    plain ``person`` list at the root, so uploads of either file type work.
    Name preference: knownName > shortFirstName + " " + shortLastName.
    """
    player_map: dict[str, str] = {}

    def _add_person(person: dict) -> None:
        pid = person.get("id", "")
        if not pid:
            return
        known = person.get("knownName", "")
        if known:
            player_map[pid] = known
        else:
            first = person.get("shortFirstName", "")
            last  = person.get("shortLastName", "")
            player_map[pid] = f"{first} {last}".strip()

    for squad_entry in data.get("squad", []):
        for person in squad_entry.get("person", []):
            _add_person(person)
    for person in data.get("person", []):
        _add_person(person)
    return player_map


def _load_game_from_bytes(phases_bytes: bytes, runs_bytes: bytes, game_label: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Parse phases + runs from raw JSON bytes (in-memory upload path).

    Returns (match_info, phases_df, runs_df).
    """
    phases_data = json.loads(phases_bytes)
    runs_data   = json.loads(runs_bytes)

    match_info, phases_df = parse_phases_json(phases_data)
    runs_df = parse_runs_json(runs_data)

    game_id = match_info.get("match_id", game_label)
    desc    = match_info.get("description", game_label)

    if not phases_df.empty:
        phases_df["game_id"]           = game_id
        phases_df["match_description"] = desc
    if not runs_df.empty:
        runs_df["game_id"]             = game_id
        runs_df["match_description"]   = desc
        runs_df["composite_run_id"]    = game_id + "_" + runs_df["run_id"].astype(str)

    return match_info, phases_df, runs_df


_ZIP_MAX_SINGLE_FILE_BYTES = 200 * 1024 * 1024   # 200 MB per file
_ZIP_MAX_TOTAL_BYTES      = 500 * 1024 * 1024   # 500 MB total uncompressed


def _safe_zip_member_name(raw_name: str) -> str | None:
    """Return the normalised member name, or None if it looks like a zip-slip path."""
    norm = raw_name.replace("\\", "/")
    # Reject any path component that is ".." (zip-slip attack)
    if any(part == ".." for part in norm.split("/")):
        return None
    return norm


def _discover_games_from_zip(zf: "zipfile.ZipFile") -> tuple[dict[str, bytes], dict[str, dict]]:
    """Scan a ZipFile for phase/run/squad files matching the expected folder layout.

    Expected paths inside the ZIP (the competition folder may or may not be the
    ZIP root)::

        <any prefix>/remote/non_aggregated/phases/<game_id>.json
        <any prefix>/remote/non_aggregated/runs/<game_id>.json
        <any prefix>/squad_lists.json

    Raises
    ------
    ValueError
        If the total uncompressed size exceeds ``_ZIP_MAX_TOTAL_BYTES`` (zip-bomb
        guard) or a single member exceeds ``_ZIP_MAX_SINGLE_FILE_BYTES``.

    Returns
    -------
    squad_bytes_map : dict[str, bytes]
        ``{"squad": <raw bytes>}`` if a squad file was found, else ``{}``.
    games : dict[str, dict]
        ``game_id → {"phases_member": str, "runs_member": str}``
        where the values are zip-internal member names.
    """
    # ── Zip-bomb guard: check declared uncompressed sizes before reading ──
    total_uncompressed = sum(info.file_size for info in zf.infolist())
    if total_uncompressed > _ZIP_MAX_TOTAL_BYTES:
        raise ValueError(
            f"ZIP total uncompressed size ({total_uncompressed / 1024 / 1024:.0f} MB) "
            f"exceeds the {_ZIP_MAX_TOTAL_BYTES // 1024 // 1024} MB limit."
        )

    phase_map: dict[str, str] = {}   # game_id → zip member name
    run_map:   dict[str, str] = {}
    squad_member: str | None = None

    for info in zf.infolist():
        # ── Zip-slip guard ────────────────────────────────────────────────
        norm = _safe_zip_member_name(info.filename)
        if norm is None:
            continue   # skip suspicious paths silently

        # ── Skip macOS metadata / resource-fork files (._<name>, .DS_Store) ──
        basename = norm.split("/")[-1]
        if basename.startswith("._") or basename == ".DS_Store":
            continue

        # ── Per-file size guard ───────────────────────────────────────────
        if info.file_size > _ZIP_MAX_SINGLE_FILE_BYTES:
            continue   # skip oversized individual files

        low = norm.lower()

        if low.endswith("squad_lists.json"):
            squad_member = info.filename
            continue

        if not low.endswith(".json"):
            continue

        parts = norm.split("/")
        fname = parts[-1]
        stem  = fname[:-5]   # strip .json

        # Identify phases vs runs by parent directory name
        parent_dirs = [p.lower() for p in parts[:-1]]
        if "phases" in parent_dirs:
            phase_map[stem] = info.filename
        elif "runs" in parent_dirs:
            run_map[stem] = info.filename

    common = sorted(phase_map.keys() & run_map.keys())
    games: dict[str, dict] = {
        gid: {"phases_member": phase_map[gid], "runs_member": run_map[gid]}
        for gid in common
    }

    squad_bytes_map: dict[str, bytes] = {}
    if squad_member:
        squad_bytes_map["squad"] = zf.read(squad_member)

    return squad_bytes_map, games


def _peek_description_from_bytes(data_bytes: bytes) -> str:
    """Fast-read matchInfo.description from raw JSON bytes (first 2 KB)."""
    try:
        head = data_bytes[:2048].decode("utf-8", errors="replace")
        m = re.search(r'"description"\s*:\s*"([^"]+)"', head)
        if m:
            return m.group(1)
        data = json.loads(data_bytes)
        return data.get("matchInfo", {}).get("description", "")
    except Exception:
        return ""


def _sidebar_upload_mode() -> None:
    """Sidebar UI: upload the competition folder as a ZIP, then select games to load."""
    st.markdown("### 📤 Upload competition folder")

    zip_file = st.file_uploader(
        "Competition folder (.zip)",
        type=["zip"],
        accept_multiple_files=False,
        key="upload_zip_file",
    )

    if zip_file is None:
        st.session_state.pop("_zip_squad_bytes", None)
        st.session_state.pop("_zip_games", None)
        st.session_state.pop("_zip_game_labels", None)
        return

    # ── Parse ZIP contents (re-scan only when the file changes) ──────────
    zip_key = (zip_file.name, zip_file.size)
    if st.session_state.get("_zip_key") != zip_key:
        # Buffer bytes first — ZipFile(zip_file) would exhaust the stream
        zf_bytes = zip_file.read()
        try:
            zf = zipfile.ZipFile(BytesIO(zf_bytes))
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")
            return

        with st.spinner("Scanning ZIP contents…"):
            try:
                squad_bytes_map, games = _discover_games_from_zip(zf)
            except ValueError as exc:
                st.error(f"❌ ZIP rejected: {exc}")
                return

            # Build human-readable labels for each game_id
            game_labels: dict[str, str] = {}
            for gid, meta in games.items():
                raw = zf.read(meta["phases_member"])
                desc = _peek_description_from_bytes(raw)
                game_labels[gid] = f"{desc}  [{gid[:8]}]" if desc else gid

        st.session_state["_zip_key"]         = zip_key
        st.session_state["_zip_zf_bytes"]    = zf_bytes
        st.session_state["_zip_squad_bytes"] = squad_bytes_map
        st.session_state["_zip_games"]       = games
        st.session_state["_zip_game_labels"] = game_labels
        # Reset selections when a new ZIP is uploaded
        st.session_state.pop("upload_selected_ids", None)

    games       = st.session_state.get("_zip_games", {})
    game_labels = st.session_state.get("_zip_game_labels", {})

    if not games:
        st.error(
            "No matching game pairs found in the ZIP. "
            "Make sure it contains `remote/non_aggregated/phases/` and "
            "`remote/non_aggregated/runs/` directories with `.json` files."
        )
        return

    squad_bytes_map = st.session_state.get("_zip_squad_bytes", {})
    has_squad = bool(squad_bytes_map)
    st.caption(
        f"{len(games)} game(s) found · "
        f"{'✅ squad list included' if has_squad else '⚠️ no squad_lists.json found'}"
    )

    all_ids = sorted(games.keys())

    if st.button("✅ Select all games", key="upload_select_all"):
        st.session_state["upload_selected_ids"] = all_ids

    selected_ids: list[str] = st.multiselect(
        "Choose game(s) to load",
        options=all_ids,
        format_func=lambda gid: game_labels.get(gid, gid),
        key="upload_selected_ids",
    )

    load_clicked = st.button(
        "🚀 Load selected games",
        disabled=not selected_ids,
        key="upload_load_btn",
    )

    if not load_clicked:
        return

    # ── Parse squad ───────────────────────────────────────────────────────
    squad_map: dict[str, str] = {}
    if has_squad:
        try:
            raw_squad = squad_bytes_map["squad"]
            # Strip UTF-8 BOM if present, then try to decode with common encodings
            if raw_squad.startswith(b"\xef\xbb\xbf"):
                raw_squad = raw_squad[3:]
            squad_text: str | None = None
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    squad_text = raw_squad.decode(enc)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue
            if squad_text is None:
                squad_text = raw_squad.decode("utf-8", errors="replace")
            squad_map = _parse_squad_json(json.loads(squad_text))
        except json.JSONDecodeError as exc:
            st.warning(f"squad_lists.json is not valid JSON and will be ignored: {exc}")
        except Exception as exc:
            st.warning(f"Could not parse squad_lists.json: {exc}")

    # ── Re-open ZIP from buffered bytes and load selected games ───────────
    zf_bytes = st.session_state.get("_zip_zf_bytes")
    if zf_bytes is None:
        st.error("ZIP data lost — please re-upload the file.")
        return

    # zip_file has been fully consumed; re-open from the buffer we saved earlier
    zf = zipfile.ZipFile(BytesIO(zf_bytes))

    with st.spinner(f"Loading {len(selected_ids)} game(s)…"):
        all_phases:   list[pd.DataFrame] = []
        all_runs:     list[pd.DataFrame] = []
        combined_map: dict[str, str]     = {}
        descriptions: list[str]         = []
        loaded_labels: list[str]        = []

        for gid in selected_ids:
            meta = games[gid]
            try:
                ph_bytes = zf.read(meta["phases_member"])
                ru_bytes = zf.read(meta["runs_member"])
                minfo, ph_df, ru_df = _load_game_from_bytes(ph_bytes, ru_bytes, gid)
                all_phases.append(ph_df)
                all_runs.append(ru_df)
                combined_map.update(minfo.get("contestant_map", {}))
                descriptions.append(minfo.get("description", gid))
                loaded_labels.append(game_labels.get(gid, gid))
            except json.JSONDecodeError as exc:
                st.warning(f"⚠️ **{game_labels.get(gid, gid)}**: feed file contains invalid JSON — {exc}")
            except Exception as exc:
                st.warning(f"⚠️ Could not load **{game_labels.get(gid, gid)}**: {exc}")

    if not all_phases:
        st.error("No data could be loaded from the selected games.")
        return

    merged_phases = pd.concat(all_phases, ignore_index=True)
    merged_runs   = pd.concat(all_runs,   ignore_index=True) if all_runs else pd.DataFrame()

    st.session_state["phases_df"]      = merged_phases
    st.session_state["runs_df"]        = merged_runs
    st.session_state["match_info"]     = {
        "match_id":       "multi-game" if len(loaded_labels) > 1 else (loaded_labels[0] if loaded_labels else "uploaded"),
        "description":    "; ".join(descriptions),
        "contestant_map": combined_map,
    }
    st.session_state["squad_map"]       = squad_map
    st.session_state["loaded_game_ids"] = loaded_labels
    st.session_state.pop("runs_committed", None)
    st.session_state.pop("pa_committed", None)
    st.success(
        f"Loaded {len(merged_phases):,} phases and {len(merged_runs):,} runs "
        f"from {len(loaded_labels)} game(s)."
    )


def _sidebar_local_mode() -> None:
    """Sidebar UI for the local feeds-directory data-loading mode."""
    # Competition (sub-folder) selector
    if not _COMPETITION_DIRS:
        st.error(f"No sub-folders found inside `{_FEEDS_BASE}`.")
        st.stop()

    selected_comp = st.selectbox(
        "📁 Competition folder",
        options=_COMPETITION_DIRS,
        help="Choose a competition sub-folder inside the feeds/ directory.",
        key="selected_competition",
    )

    squad_map: dict[str, str] = load_squad_map(selected_comp)
    all_games = discover_available_games(selected_comp)

    if not all_games:
        st.error(
            f"No games found for **{selected_comp}**. "
            "Make sure it contains "
            "`remote/non_aggregated/phases` and "
            "`remote/non_aggregated/runs` directories with `.json` files."
        )
        st.stop()

    st.caption(f"{len(all_games)} game(s) available")

    game_by_id = {g["game_id"]: g for g in all_games}

    @st.cache_data(show_spinner=False)
    def _build_game_labels(game_ids: tuple[str, ...], phases_paths: tuple[str, ...]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for gid, ppath in zip(game_ids, phases_paths):
            desc = _peek_description(ppath)
            labels[gid] = f"{desc}  [{gid[:8]}]" if desc else gid
        return labels

    _gids   = tuple(g["game_id"]     for g in all_games)
    _ppaths = tuple(g["phases_path"] for g in all_games)
    game_labels = _build_game_labels(_gids, _ppaths)

    all_ids    = [g["game_id"] for g in all_games]
    id_to_label = {gid: game_labels.get(gid, gid) for gid in all_ids}

    if st.button("✅ Select all games", key="select_all_games"):
        st.session_state["selected_game_ids"] = all_ids

    selected_ids: list[str] = st.multiselect(
        "Choose game(s) to load",
        options=all_ids,
        format_func=lambda gid: id_to_label.get(gid, gid),
        key="selected_game_ids",
    )

    load_clicked = st.button("🚀 Load selected games", disabled=not selected_ids)

    if load_clicked and selected_ids:
        with st.spinner(f"Loading {len(selected_ids)} game(s)…"):
            all_phases: list[pd.DataFrame] = []
            all_runs:   list[pd.DataFrame] = []
            combined_map: dict[str, str]   = {}
            descriptions: list[str]        = []

            for gid in selected_ids:
                gmeta = game_by_id[gid]
                try:
                    minfo, ph_df, ru_df = _load_game(gmeta)
                    all_phases.append(ph_df)
                    all_runs.append(ru_df)
                    combined_map.update(minfo.get("contestant_map", {}))
                    descriptions.append(minfo.get("description", gid))
                except Exception as exc:
                    st.warning(f"⚠️ Could not load {gid}: {exc}")

        if all_phases:
            merged_phases = pd.concat(all_phases, ignore_index=True)
            merged_runs   = pd.concat(all_runs,   ignore_index=True) if all_runs else pd.DataFrame()
            st.session_state["phases_df"]      = merged_phases
            st.session_state["runs_df"]        = merged_runs
            st.session_state["match_info"]     = {
                "match_id":       "multi-game" if len(selected_ids) > 1 else selected_ids[0],
                "description":    "; ".join(descriptions),
                "contestant_map": combined_map,
            }
            st.session_state["squad_map"]       = squad_map
            st.session_state["loaded_game_ids"] = selected_ids
            st.session_state.pop("runs_committed", None)
            st.session_state.pop("pa_committed", None)
            st.success(f"Loaded {len(merged_phases):,} phases and {len(merged_runs):,} runs from {len(selected_ids)} game(s).")
        else:
            st.error("No data could be loaded from the selected games.")


# ──────────────────────────────────────────────────────────────────────────────
# Brand colours & header
# ──────────────────────────────────────────────────────────────────────────────
# (Colours are loaded from .streamlit/config.toml [brand] at module top)

_LOGOS_DIR = _REPO_ROOT / "logos"
_LOGO_DARK  = _LOGOS_DIR / "opta-ai-logo_white.png"   # white logo → shown on dark banner
_LOGO_LIGHT = _LOGOS_DIR / "opta-ai-logo_black.png"   # black logo → fallback


def _logo_b64(path: Path) -> str | None:
    """Return a data-URI string for the logo at *path*, or None if missing."""
    try:
        data = path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode()
    except (FileNotFoundError, OSError):
        return None


def _render_header() -> None:
    """Inject brand CSS and render the top banner with logo + app title."""
    logo_uri = _logo_b64(_LOGO_DARK) or _logo_b64(_LOGO_LIGHT) or ""
    logo_html = (
        f'<img src="{logo_uri}" alt="Opta AI" style="height:44px;display:block;">'
        if logo_uri else ""
    )

    st.markdown(
        f"""
<style>
/* ── Google Fonts import ──────────────────────────────────────── */
@import url('{_FONT_GFX_URL}');

/* ── Font role definitions ────────────────────────────────────── */
/* Headline: Black Compressed (Barlow Condensed 900) — ≤8 words */
/* Wide headline: ExtraBold Wide (Barlow 800 expanded) — limited vertical space */
/* Title / subheader: SemiBold (Barlow 600) — all-caps subheaders ≤8 words */
/* Body / semilight: Light (Barlow 300) — body copy and sub-headlines */

/* Base body font — Semilight / body copy.
   NOTE: Do NOT include span or div here — Streamlit renders Material Icons
   as <span class="material-icons"> / <span class="material-symbols-*"> and
   overriding their font-family breaks icon ligatures (shows raw text instead). */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
p, li, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    font-family: '{_FONT_BODY}', sans-serif !important;
    font-weight: 300 !important;
}}

/* Explicitly protect Material Icons / Symbols from font override */
.material-icons,
.material-icons-outlined,
.material-icons-round,
.material-icons-sharp,
[class*="material-symbols"] {{
    font-family: 'Material Icons', 'Material Symbols Outlined', 'Material Symbols Rounded' !important;
    font-weight: normal !important;
}}

/* Headings h1 — Black Compressed headline */
h1,
[data-testid="stMarkdownContainer"] h1 {{
    font-family: '{_FONT_HEADLINE}', sans-serif !important;
    font-weight: 900 !important;
    letter-spacing: 0.01em !important;
    text-transform: uppercase !important;
}}

/* Headings h2 — ExtraBold Wide sub-headline */
h2,
[data-testid="stMarkdownContainer"] h2 {{
    font-family: '{_FONT_WIDE}', sans-serif !important;
    font-weight: 800 !important;
    font-stretch: expanded !important;
    letter-spacing: 0.04em !important;
}}

/* Headings h3/h4 — SemiBold title / subheader */
h3, h4,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    font-family: '{_FONT_TITLE}', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}}

/* Bold / strong emphasis — SemiBold */
strong, b,
[data-testid="stMarkdownContainer"] strong {{
    font-family: '{_FONT_TITLE}', sans-serif !important;
    font-weight: 600 !important;
}}

/* Widget labels — SemiBold */
[data-testid="stWidgetLabel"] p,
label {{
    font-family: '{_FONT_TITLE}', sans-serif !important;
    font-weight: 600 !important;
}}

/* Buttons — SemiBold */
button, .stButton > button {{
    font-family: '{_FONT_TITLE}', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}}

/* Tab labels — SemiBold */
.stTabs [data-baseweb="tab"] {{
    font-family: '{_FONT_TITLE}', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}}

/* Dataframe / code — keep monospace but inherit body otherwise */
code, pre {{
    font-family: 'SFMono-Regular', 'Consolas', monospace !important;
    font-weight: 400 !important;
}}

/* Banner title — Black Compressed */
.brand-header-title {{
    font-family: '{_FONT_HEADLINE}', sans-serif !important;
    font-weight: 900 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}}

/* Banner sub-label — ExtraBold Wide at ~half headline size */
.brand-header-sub {{
    font-family: '{_FONT_WIDE}', sans-serif !important;
    font-weight: 800 !important;
    font-stretch: expanded !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}}

/* ── Dark base ────────────────────────────────────────────────── */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {{
    background-color: #000000 !important;
    color: #f0f0f0 !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: #0d0d0d !important;
}}
section[data-testid="stSidebar"] * {{
    color: #f0f0f0 !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {_BRAND_AMBER} !important;
}}

/* ── Top banner ───────────────────────────────────────────────── */
.brand-header {{
    display: flex;
    align-items: center;
    gap: 20px;
    background: #000000;
    padding: 14px 28px;
    border-radius: 8px;
    margin-bottom: 18px;
    border-bottom: 3px solid {_BRAND_AMBER};
}}
.brand-header-title {{
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    line-height: 1.2;
    margin: 0;
}}
.brand-header-sub {{
    color: {_BRAND_AMBER};
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

/* ── Expanders ───────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background-color: #0d0d0d !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
}}
[data-testid="stExpander"] summary {{
    color: #f0f0f0 !important;
}}

/* ── Inputs, selects, text areas ─────────────────────────────── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
    border-color: #333333 !important;
}}

/* ── Multiselect ─────────────────────────────────────────────── */
/* Outer container gets the dark background */
[data-testid="stMultiSelect"] div[data-baseweb="select"] {{
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
}}
/* ALL inner layers transparent — prevents any inner div from showing a
   mismatched background that causes the grey-rectangle-on-first-tag bug */
[data-testid="stMultiSelect"] div[data-baseweb="select"] div {{
    background-color: transparent !important;
    color: #f0f0f0 !important;
}}
[data-testid="stMultiSelect"] input {{
    background-color: transparent !important;
    color: #f0f0f0 !important;
}}
/* Tags (selected value pills) — purple fill */
[data-baseweb="tag"] {{
    background-color: {_BRAND_PURPLE} !important;
    color: #ffffff !important;
}}
[data-baseweb="tag"] span,
[data-baseweb="tag"] * {{
    color: #ffffff !important;
    background-color: transparent !important;
}}
/* Dropdown menu needs its own dark background (it's outside the select div) */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {{
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
}}
[data-baseweb="popover"] li:hover {{
    background-color: #333333 !important;
}}

/* ── Dataframes / tables ─────────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] iframe,
.stDataFrame {{
    background-color: #0d0d0d !important;
}}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: #000000 !important;
    border-bottom: 1px solid #2a2a2a !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: #aaaaaa !important;
}}
.stTabs [aria-selected="true"] {{
    color: {_BRAND_AMBER} !important;
    border-bottom-color: {_BRAND_AMBER} !important;
}}

/* ── Primary buttons ─────────────────────────────────────────── */
.stButton > button[kind="primary"] {{
    background-color: {_BRAND_PURPLE} !important;
    border-color:     {_BRAND_PURPLE} !important;
    color: #ffffff !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {_BRAND_ORANGE} !important;
    border-color:     {_BRAND_ORANGE} !important;
}}

/* ── Secondary buttons ───────────────────────────────────────── */
.stButton > button[kind="secondary"] {{
    background-color: #1a1a1a !important;
    border-color: #444444 !important;
    color: #f0f0f0 !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {_BRAND_AMBER} !important;
    color: {_BRAND_AMBER} !important;
}}

/* ── Info / warning / success / error boxes ──────────────────── */
[data-testid="stAlert"] {{
    background-color: #1a1a1a !important;
    border-radius: 6px !important;
}}

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background-color: #0d0d0d !important;
    border-radius: 6px !important;
    padding: 8px !important;
}}

/* ── Progress / spinner ──────────────────────────────────────── */
.stProgress > div > div > div {{
    background-color: {_BRAND_PURPLE} !important;
}}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #000000; }}
::-webkit-scrollbar-thumb {{ background: #333333; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {_BRAND_AMBER}; }}
</style>

<div class="brand-header">
    {logo_html}
    <div>
        <p class="brand-header-sub">Stats Perform · Opta</p>
        <p class="brand-header-title">Phases of Play – Feed Analysis</p>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Phases of Play – Feed Analysis",
        layout="wide",
        page_icon=str(_LOGO_LIGHT) if _LOGO_LIGHT.exists() else "⚽",
    )
    _render_header()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 Load Data")

        # VOD API key – use env var / secrets if set, otherwise show a text input
        if not _get_vod_api_key():
            st.text_input(
                "🔑 VOD API Key",
                type="password",
                key="_vod_k",
                help="Required for video playback. Set the VOD_API_KEY environment variable or Streamlit secret to avoid entering it here.",
            )

        # Choose loading mode: upload (always available) vs local feeds directory
        _has_local_feeds = bool(_COMPETITION_DIRS)
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
            _sidebar_upload_mode()
        else:
            _sidebar_local_mode()

    # ── Main content ──────────────────────────────────────────────────────
    if "phases_df" not in st.session_state or st.session_state["phases_df"] is None:
        st.info("👈 Upload your feed files (or select from local directory) in the sidebar and click **Load** to begin.")
        st.stop()

    phases_df: pd.DataFrame = st.session_state["phases_df"]
    runs_df:   pd.DataFrame = st.session_state["runs_df"]
    match_info: dict        = st.session_state["match_info"]
    squad_map: dict[str, str] = st.session_state.get("squad_map", {})

    loaded_ids = st.session_state.get("loaded_game_ids", [])
    if len(loaded_ids) == 1:
        st.markdown(f"### {match_info['description']}")
    else:
        st.markdown(f"### {len(loaded_ids)} games loaded")
        with st.expander("Loaded games", expanded=False):
            for gid in loaded_ids:
                st.write(f"• {gid}")

    st.markdown(
        f"**Total phases:** {len(phases_df):,} &nbsp;|&nbsp; "
        f"**Total runs:** {len(runs_df):,}"
    )

    tab_runs, tab_analysis = st.tabs(["Runs Search", "Phase Analysis"])

    with tab_runs:
        if runs_df is None or runs_df.empty:
            st.info("No run data available for the loaded game(s).")
        else:
            _analysis_runs_by_phase(phases_df, runs_df, match_info, squad_map)

    with tab_analysis:
        _analysis_phase_analysis(phases_df, match_info, squad_map)


if __name__ == "__main__":
    main()


