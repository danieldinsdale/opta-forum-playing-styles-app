"""
Feed parsers — XML and JSON formats for phases and runs.

Also contains the game-state calculator (``compute_game_state``).

All functions in this module are **pure** (no Streamlit dependency).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame dtype optimiser — called after every parse to shrink memory
# ──────────────────────────────────────────────────────────────────────────────

# Small-range integer columns → Int16 (range ±32 767)
_INT16_COLS = frozenset({
    "game_state",
})

# Medium-range integer columns → Int32
_INT32_COLS = frozenset({
    "startTime", "endTime", "startFrame", "endFrame", "phaseDuration",
})

# Float columns that can safely be stored as float32
_FLOAT32_COLS = frozenset({
    "startX", "startY", "endX", "endY",
    "expectedThreat_max", "speed_max",
    "defensiveLineBroken", "dangerous",
    "runFollowedByTeamShot", "runFollowedByTeamGoal",
    "averageAttackingTeamHorizontalWidth",
    "averageAttackingTeamVerticalLength",
    "averageDefendingTeamHorizontalWidth",
    "averageDefendingTeamVerticalLength",
    "averageDefensiveAreaCoverage",
    "averageAttackingTeamHeightLastDefender",
    "averageDefendingTeamHeightLastDefender",
    # phaseSummary count columns (small non-negative ints stored as float)
    "numberDangerousRuns", "numberLineBreakingActions",
    "numberPasses", "numberHighPressureOnReceiver",
    "numberHighPressureOnTouches",
})

# Low-cardinality string columns → category dtype
_CATEGORY_COLS = frozenset({
    "phaseLabel", "periodId", "possessionContestantId",
    "runType", "masterLabel", "contestantId", "team_name",
    "overloadTypes", "playerId", "game_id", "match_description",
    "initiatorPlayerId", "firstTouchPlayerId",
    "mostCommonDefensiveCompactness",
    "includesShots", "includesGoal",
})


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Down-cast numeric columns and convert low-cardinality strings to
    ``category`` dtype to reduce memory usage.  Mutates *df* in-place and
    returns it for convenience."""
    if df.empty:
        return df

    for col in _INT16_COLS & set(df.columns):
        if df[col].dtype == object:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
        except (ValueError, TypeError):
            pass

    for col in _INT32_COLS & set(df.columns):
        if df[col].dtype == object:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
        except (ValueError, TypeError):
            pass

    for col in _FLOAT32_COLS & set(df.columns):
        if df[col].dtype == object:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        except (ValueError, TypeError):
            pass

    for col in _CATEGORY_COLS & set(df.columns):
        if df[col].dtype == object or str(df[col].dtype) == "string":
            try:
                df[col] = df[col].astype("category")
            except (ValueError, TypeError):
                pass

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phases – XML
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

        for key in ("startFrame", "endFrame"):
            v = phase.get(key)
            if v is not None:
                row[key] = int(v)

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

        label_el = phase.find("phaseLabel")
        row["phaseLabel"] = label_el.get("value") if label_el is not None else "Unknown"

        summary_el = phase.find("phaseSummary")
        if summary_el is not None:
            for stat in summary_el.findall("stat"):
                stat_type = stat.get("type", "")
                stat_value = stat.text
                if stat_value is not None:
                    try:
                        row[stat_type] = float(stat_value)
                    except ValueError:
                        row[stat_type] = stat_value

        overload_el = phase.find("overloadSummary")
        ol_children = list(overload_el) if overload_el is not None else []
        row["containsOverload"] = len(ol_children) > 0
        ol_types = sorted({o.get("overloadType", "") for o in ol_children if o.get("overloadType")})
        row["overloadTypes"] = ",".join(ol_types) if ol_types else ""

        rows.append(row)

    phases_df = pd.DataFrame(rows)

    if not phases_df.empty and contestant_map:
        phases_df["team_name"] = phases_df["possessionContestantId"].map(contestant_map)


    _optimize_dtypes(phases_df)
    return match_info, phases_df


# ──────────────────────────────────────────────────────────────────────────────
# Runs – XML
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

        for key in ("startFrame", "endFrame"):
            v = elem.get(key)
            if v is not None:
                row[key] = int(v)

        for attr in ("startX", "startY", "endX", "endY"):
            v = elem.get(attr)
            if v is not None:
                row[attr] = float(v)

        for t_attr in ("startTime", "endTime"):
            tv = elem.get(t_attr)
            if tv is not None:
                try:
                    row[t_attr] = int(tv)
                except (ValueError, TypeError):
                    pass

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
                elif qtype == "speed":
                    v = q.get("max")
                    if v is not None:
                        row["speed_max"] = round(float(v), 2)
                elif qtype == "runFollowedByTeamShot":
                    v = q.get("value")
                    if v is not None:
                        row["runFollowedByTeamShot"] = float(v)
                elif qtype == "runFollowedByTeamGoal":
                    v = q.get("value")
                    if v is not None:
                        row["runFollowedByTeamGoal"] = float(v)

        rows.append(row)
        elem.clear()

    df = pd.DataFrame(rows)
    _optimize_dtypes(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phases – JSON
# ──────────────────────────────────────────────────────────────────────────────


def parse_phases_json(data: dict) -> tuple[dict, pd.DataFrame]:
    """Parse a phases JSON dict (same schema as remotePhasesOfPlay JSON feed).

    Returns the same ``(match_info, phases_df)`` as :func:`parse_phases_xml`.
    """
    mi = data.get("matchInfo", {})
    match_id    = mi.get("id", "unknown")
    description = mi.get("description", "")

    contestant_map: dict[str, str] = {}
    contestant_position: dict[str, str] = {}
    for c in mi.get("contestant", []):
        contestant_map[c.get("id", "")] = c.get("name", c.get("officialName", ""))
        pos = c.get("position", "")
        if pos:
            contestant_position[c.get("id", "")] = pos

    ld = data.get("liveData", {})
    goal_events: list[dict] = []
    for g in ld.get("goal", []):
        tms = g.get("timeMinSec", "0:00")
        parts = tms.split(":")
        try:
            goal_ms = (int(parts[0]) * 60 + int(parts[1])) * 1000
        except (ValueError, IndexError):
            goal_ms = 0
        goal_events.append({
            "time_ms":   goal_ms,
            "periodId":  g.get("periodId"),
            "homeScore": int(g.get("homeScore", 0)),
            "awayScore": int(g.get("awayScore", 0)),
        })
    goal_events.sort(key=lambda x: x["time_ms"])

    match_info = {
        "match_id":            match_id,
        "description":         description,
        "contestant_map":      contestant_map,
        "contestant_position": contestant_position,
        "goal_events":         goal_events,
    }

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

        pl = phase.get("phaseLabel", {})
        row["phaseLabel"] = pl.get("value", "Unknown") if isinstance(pl, dict) else str(pl)

        ps = phase.get("phaseSummary", {})
        for stat in ps.get("stat", []):
            stype = stat.get("type", "")
            sval  = stat.get("value")
            if stype and sval is not None:
                try:
                    row[stype] = float(sval)
                except (ValueError, TypeError):
                    row[stype] = sval

        overload = phase.get("overloadSummary", {})
        overload_list = overload.get("overload", []) if isinstance(overload, dict) else []
        if isinstance(overload_list, dict):
            overload_list = [overload_list]
        row["containsOverload"] = bool(overload_list)
        ol_types = sorted({o.get("overloadType", "") for o in overload_list if o.get("overloadType")})
        row["overloadTypes"] = ",".join(ol_types) if ol_types else ""

        # ── Defensive area coverage (time-weighted across sub-entries) ──
        dcs = phase.get("defensiveCompactnessSummary", {})
        dc_list = dcs.get("defensiveCompactness", []) if isinstance(dcs, dict) else []
        if isinstance(dc_list, dict):
            dc_list = [dc_list]
        if dc_list:
            _total_dur = 0.0
            _weighted_sum = 0.0
            for dc in dc_list:
                try:
                    _dc_start = float(dc.get("startTime", 0))
                    _dc_end = float(dc.get("endTime", 0))
                    _dc_area = float(dc.get("averageDefensiveAreaCoverage", 0))
                except (ValueError, TypeError):
                    continue
                _dc_dur = max(_dc_end - _dc_start, 0.0)
                if _dc_dur > 0 and _dc_area > 0:
                    _weighted_sum += _dc_area * _dc_dur
                    _total_dur += _dc_dur
            if _total_dur > 0:
                row["averageDefensiveAreaCoverage"] = _weighted_sum / _total_dur

        rows.append(row)

    phases_df = pd.DataFrame(rows)

    if not phases_df.empty and contestant_map:
        phases_df["team_name"] = phases_df["possessionContestantId"].map(contestant_map)


    _optimize_dtypes(phases_df)
    return match_info, phases_df


# ──────────────────────────────────────────────────────────────────────────────
# Runs – JSON
# ──────────────────────────────────────────────────────────────────────────────


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

        for t_attr in ("startTime", "endTime"):
            tv = run.get(t_attr)
            if tv is not None:
                try:
                    row[t_attr] = int(tv)
                except (ValueError, TypeError):
                    pass

        labels_obj = run.get("labels", {})
        label_list = labels_obj.get("label", []) if isinstance(labels_obj, dict) else []
        if isinstance(label_list, dict):
            label_list = [label_list]
        for lbl in label_list:
            if isinstance(lbl, dict) and lbl.get("type") == "master":
                row["masterLabel"] = lbl.get("value", "")
                break

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
            elif qtype == "speed":
                v = q.get("max")
                if v is not None:
                    row["speed_max"] = round(float(v), 2)
            elif qtype == "runFollowedByTeamShot":
                v = q.get("value")
                if v is not None:
                    row["runFollowedByTeamShot"] = float(v)
            elif qtype == "runFollowedByTeamGoal":
                v = q.get("value")
                if v is not None:
                    row["runFollowedByTeamGoal"] = float(v)

        if contestant_map and row.get("contestantId"):
            row["team_name"] = contestant_map.get(row["contestantId"], "")

        rows.append(row)

    df = pd.DataFrame(rows)
    _optimize_dtypes(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Game-state calculation
# ──────────────────────────────────────────────────────────────────────────────


def compute_game_state(
    phases_df: pd.DataFrame,
    match_info: dict,
) -> pd.DataFrame:
    """Add a ``game_state`` column to *phases_df*.

    ``game_state`` is the goal-difference from the perspective of the
    **possessing** team at the moment each phase starts.

    Uses vectorised NumPy operations instead of row-by-row ``.apply()``.
    """
    goal_events = match_info.get("goal_events", [])
    contestant_position = match_info.get("contestant_position", {})

    home_ids = {cid for cid, pos in contestant_position.items() if pos == "home"}
    away_ids = {cid for cid, pos in contestant_position.items() if pos == "away"}

    if phases_df.empty:
        return phases_df

    df = phases_df.copy()

    if not goal_events:
        df["game_state"] = np.int8(0)
        return df

    # Build sorted goal timeline arrays for np.searchsorted
    goal_times = np.array([g["time_ms"] for g in goal_events], dtype=np.int64)
    home_scores = np.array([g["homeScore"] for g in goal_events], dtype=np.int32)
    away_scores = np.array([g["awayScore"] for g in goal_events], dtype=np.int32)

    start_times = df["startTime"].to_numpy(dtype=np.int64, na_value=0)
    # searchsorted with side='right' gives index of first goal *after* start_time
    idx = np.searchsorted(goal_times, start_times, side="right")

    # Score at each phase start = score from goal at idx-1 (or 0-0 if idx==0)
    h_scores = np.where(idx > 0, home_scores[np.clip(idx - 1, 0, len(home_scores) - 1)], 0)
    a_scores = np.where(idx > 0, away_scores[np.clip(idx - 1, 0, len(away_scores) - 1)], 0)

    cids = df["possessionContestantId"].astype(str).values
    is_home = np.isin(cids, list(home_ids))
    is_away = np.isin(cids, list(away_ids))

    game_state = np.where(is_home, h_scores - a_scores,
                          np.where(is_away, a_scores - h_scores, 0))
    df["game_state"] = game_state.astype(np.int8)
    return df

