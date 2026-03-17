"""Data loading — discovery, squad parsing, game loading, ZIP handling."""
from __future__ import annotations
import json, re, zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
from src.config import FEEDS_BASE, COMPETITION_DIRS
from src.parsers import parse_phases_json, parse_runs_json, compute_game_state


def discover_available_games(competition_id=None):
    results = []
    if not FEEDS_BASE.exists():
        return results
    comp_dirs = [competition_id] if competition_id else COMPETITION_DIRS
    for comp_id in comp_dirs:
        pd_ = FEEDS_BASE / comp_id / "remote" / "non_aggregated" / "phases"
        rd_ = FEEDS_BASE / comp_id / "remote" / "non_aggregated" / "runs"
        if not pd_.exists() or not rd_.exists():
            continue
        def _gm(d, sfx):
            m = {}
            for p in d.glob("*.json"):
                s = p.stem
                m[s[:-len(sfx)] if s.endswith(sfx) else s] = p
            return m
        pm = _gm(pd_, "_phase"); rm = _gm(rd_, "_run")
        for gid in sorted(pm.keys() & rm.keys()):
            results.append({"game_id": gid, "competition_id": comp_id,
                            "phases_path": str(pm[gid]), "runs_path": str(rm[gid])})
    return results


def peek_description(phases_path):
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


def peek_description_from_bytes(data_bytes):
    try:
        head = data_bytes[:2048].decode("utf-8", errors="replace")
        m = re.search(r'"description"\s*:\s*"([^"]+)"', head)
        if m:
            return m.group(1)
        return json.loads(data_bytes).get("matchInfo", {}).get("description", "")
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_squad_map(competition_id):
    p = FEEDS_BASE / competition_id / "squad_lists.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return parse_squad_json(json.load(f))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_jersey_map(competition_id):
    p = FEEDS_BASE / competition_id / "squad_lists.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return parse_squad_jersey_json(json.load(f))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_team_squad_map(competition_id):
    p = FEEDS_BASE / competition_id / "squad_lists.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    tm = {}
    for sq in data.get("squad", []):
        cid = sq.get("contestantId", "")
        if cid:
            tm[cid] = {"name": sq.get("contestantName", cid),
                       "player_ids": [pp["id"] for pp in sq.get("person", []) if pp.get("id") and pp.get("type") == "player"]}
    return tm


def parse_squad_json(data):
    pm = {}
    def _a(p):
        pid = p.get("id", "")
        if not pid: return
        k = p.get("knownName", "")
        pm[pid] = k if k else f"{p.get('shortFirstName','')} {p.get('shortLastName','')}".strip()
    for sq in data.get("squad", []):
        for p in sq.get("person", []): _a(p)
    for p in data.get("person", []): _a(p)
    return pm


def parse_squad_jersey_json(data):
    j = {}
    for sq in data.get("squad", []):
        for p in sq.get("person", []):
            pid, s = p.get("id", ""), p.get("shirtNumber")
            if pid and s is not None: j[pid] = str(s)
    for p in data.get("person", []):
        pid, s = p.get("id", ""), p.get("shirtNumber")
        if pid and s is not None: j[pid] = str(s)
    return j


def _tag_game(df, game_id, desc, is_runs=False):
    if not df.empty:
        df["game_id"] = game_id
        df["match_description"] = desc
        if is_runs:
            df["composite_run_id"] = game_id + "_" + df["run_id"].astype(str)


def load_game(game_meta):
    with open(game_meta["phases_path"], encoding="utf-8") as f:
        pd_ = json.load(f)
    with open(game_meta["runs_path"], encoding="utf-8") as f:
        rd_ = json.load(f)
    mi, ph = parse_phases_json(pd_)
    ru = parse_runs_json(rd_)
    ph = compute_game_state(ph, mi)
    gid = game_meta["game_id"]
    desc = mi.get("description", gid)
    _tag_game(ph, gid, desc)
    _tag_game(ru, gid, desc, True)
    return mi, ph, ru


def load_game_from_bytes(phases_bytes, runs_bytes, game_label):
    mi, ph = parse_phases_json(json.loads(phases_bytes))
    ru = parse_runs_json(json.loads(runs_bytes))
    ph = compute_game_state(ph, mi)
    gid = mi.get("match_id", game_label)
    desc = mi.get("description", game_label)
    _tag_game(ph, gid, desc)
    _tag_game(ru, gid, desc, True)
    return mi, ph, ru


ZIP_MAX_SINGLE_FILE_BYTES = 200 * 1024 * 1024
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024


def safe_zip_member_name(raw):
    n = raw.replace("\\", "/")
    return None if any(p == ".." for p in n.split("/")) else n


def discover_games_from_zip(zf):
    tot = sum(i.file_size for i in zf.infolist())
    if tot > ZIP_MAX_TOTAL_BYTES:
        raise ValueError(f"ZIP too large ({tot/1024/1024:.0f} MB)")
    pm, rm, sq = {}, {}, None
    for info in zf.infolist():
        n = safe_zip_member_name(info.filename)
        if not n: continue
        bn = n.split("/")[-1]
        if bn.startswith("._") or bn == ".DS_Store": continue
        if info.file_size > ZIP_MAX_SINGLE_FILE_BYTES: continue
        lo = n.lower()
        if lo.endswith("squad_lists.json"): sq = info.filename; continue
        if not lo.endswith(".json"): continue
        pts = n.split("/"); stem = pts[-1][:-5]
        pdirs = [p.lower() for p in pts[:-1]]
        if "phases" in pdirs: pm[stem] = info.filename
        elif "runs" in pdirs: rm[stem] = info.filename
    common = sorted(pm.keys() & rm.keys())
    games = {g: {"phases_member": pm[g], "runs_member": rm[g]} for g in common}
    sbm = {}
    if sq: sbm["squad"] = zf.read(sq)
    return sbm, games

