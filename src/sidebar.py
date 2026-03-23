"""Sidebar UI - upload and local data loading modes."""
from __future__ import annotations
import gc
import json, zipfile
from io import BytesIO
import pandas as pd
import streamlit as st
from src.config import FEEDS_BASE, COMPETITION_DIRS
from src.data_loading import (
    discover_available_games, peek_description, peek_description_from_bytes,
    load_squad_map, load_jersey_map, parse_squad_json, parse_squad_jersey_json,
    load_game, load_game_from_bytes, discover_games_from_zip,
)


def sidebar_upload_mode() -> None:
    """Sidebar UI: upload the competition folder as a ZIP, then select games to load."""
    st.markdown("### 📤 Upload competition folder")

    zip_file = st.file_uploader("Competition folder (.zip)", type=["zip"], accept_multiple_files=False, key="upload_zip_file")

    if zip_file is None:
        st.session_state.pop("_zip_squad_bytes", None)
        st.session_state.pop("_zip_games", None)
        st.session_state.pop("_zip_game_labels", None)
        return

    zip_key = (zip_file.name, zip_file.size)
    if st.session_state.get("_zip_key") != zip_key:
        zf_bytes = zip_file.read()
        try:
            zf = zipfile.ZipFile(BytesIO(zf_bytes))
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")
            return
        with st.spinner("Scanning ZIP contents…"):
            try:
                squad_bytes_map, games = discover_games_from_zip(zf)
            except ValueError as exc:
                st.error(f"❌ ZIP rejected: {exc}")
                return
            game_labels: dict[str, str] = {}
            for gid, meta in games.items():
                raw = zf.read(meta["phases_member"])
                desc = peek_description_from_bytes(raw)
                game_labels[gid] = f"{desc}  [{gid[:8]}]" if desc else gid

        st.session_state["_zip_key"] = zip_key
        st.session_state["_zip_zf_bytes"] = zf_bytes
        st.session_state["_zip_squad_bytes"] = squad_bytes_map
        st.session_state["_zip_games"] = games
        st.session_state["_zip_game_labels"] = game_labels
        st.session_state.pop("upload_selected_ids", None)

    games = st.session_state.get("_zip_games", {})
    game_labels = st.session_state.get("_zip_game_labels", {})

    if not games:
        st.error("No matching game pairs found in the ZIP. Make sure it contains `remote/non_aggregated/phases/` and `remote/non_aggregated/runs/` directories with `.json` files.")
        return

    squad_bytes_map = st.session_state.get("_zip_squad_bytes", {})
    has_squad = bool(squad_bytes_map)
    st.caption(f"{len(games)} game(s) found · {'✅ squad list included' if has_squad else '⚠️ no squad_lists.json found'}")

    all_ids = sorted(games.keys())
    if st.button("✅ Select all games", key="upload_select_all"):
        st.session_state["upload_selected_ids"] = all_ids

    selected_ids: list[str] = st.multiselect("Choose game(s) to load", options=all_ids, format_func=lambda gid: game_labels.get(gid, gid), key="upload_selected_ids")
    load_clicked = st.button("🚀 Load selected games", disabled=not selected_ids, key="upload_load_btn")

    if not load_clicked:
        return

    squad_map: dict[str, str] = {}
    jersey_map: dict[str, str] = {}
    if has_squad:
        try:
            raw_squad = squad_bytes_map["squad"]
            if raw_squad.startswith(b"\xef\xbb\xbf"):
                raw_squad = raw_squad[3:]
            squad_text = None
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    squad_text = raw_squad.decode(enc)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue
            if squad_text is None:
                squad_text = raw_squad.decode("utf-8", errors="replace")
            squad_json = json.loads(squad_text)
            squad_map = parse_squad_json(squad_json)
            jersey_map = parse_squad_jersey_json(squad_json)
        except json.JSONDecodeError as exc:
            st.warning(f"squad_lists.json is not valid JSON and will be ignored: {exc}")
        except Exception as exc:
            st.warning(f"Could not parse squad_lists.json: {exc}")

    zf_bytes = st.session_state.get("_zip_zf_bytes")
    if zf_bytes is None:
        st.error("ZIP data lost — please re-upload the file.")
        return
    zf = zipfile.ZipFile(BytesIO(zf_bytes))

    with st.spinner(f"Loading {len(selected_ids)} game(s)…"):
        all_phases, all_runs = [], []
        combined_map: dict[str, str] = {}
        descriptions, loaded_labels = [], []
        for gid in selected_ids:
            meta = games[gid]
            try:
                ph_bytes = zf.read(meta["phases_member"])
                ru_bytes = zf.read(meta["runs_member"])
                minfo, ph_df, ru_df = load_game_from_bytes(ph_bytes, ru_bytes, gid)
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

    merged_phases = pd.concat(all_phases, ignore_index=True, copy=False)
    merged_runs = pd.concat(all_runs, ignore_index=True, copy=False) if all_runs else pd.DataFrame()

    # Release intermediate lists before storing merged frames
    del all_phases, all_runs

    st.session_state["phases_df"] = merged_phases
    st.session_state["runs_df"] = merged_runs
    st.session_state["match_info"] = {"match_id": "multi-game" if len(loaded_labels) > 1 else (loaded_labels[0] if loaded_labels else "uploaded"), "description": "; ".join(descriptions), "contestant_map": combined_map}
    st.session_state["squad_map"] = squad_map
    st.session_state["jersey_map"] = jersey_map
    st.session_state["loaded_game_ids"] = loaded_labels
    st.session_state.pop("runs_committed", None)
    st.session_state.pop("pa_committed", None)
    st.session_state.pop("ba_committed", None)
    st.session_state.pop("tc_committed", None)

    # Free the large ZIP bytes from memory now that games are loaded
    st.session_state.pop("_zip_zf_bytes", None)
    st.session_state.pop("_zip_squad_bytes", None)
    st.session_state.pop("_zip_games", None)
    st.session_state.pop("_zip_game_labels", None)

    # Clear any stale cached computation results
    st.cache_data.clear()
    gc.collect()

    st.success(f"Loaded {len(merged_phases):,} phases and {len(merged_runs):,} runs from {len(loaded_labels)} game(s).")


def sidebar_local_mode() -> None:
    """Sidebar UI for the local feeds-directory data-loading mode."""
    if not COMPETITION_DIRS:
        st.error(f"No sub-folders found inside `{FEEDS_BASE}`.")
        st.stop()

    selected_comp = st.selectbox("📁 Competition folder", options=COMPETITION_DIRS, help="Choose a competition sub-folder inside the feeds/ directory.", key="selected_competition")

    squad_map = load_squad_map(selected_comp)
    jersey_map = load_jersey_map(selected_comp)
    all_games = discover_available_games(selected_comp)

    if not all_games:
        st.error(f"No games found for **{selected_comp}**. Make sure it contains `remote/non_aggregated/phases` and `remote/non_aggregated/runs` directories with `.json` files.")
        st.stop()

    st.caption(f"{len(all_games)} game(s) available")
    game_by_id = {g["game_id"]: g for g in all_games}

    @st.cache_data(show_spinner=False)
    def _build_game_labels(game_ids, phases_paths):
        labels = {}
        for gid, ppath in zip(game_ids, phases_paths):
            desc = peek_description(ppath)
            labels[gid] = f"{desc}  [{gid[:8]}]" if desc else gid
        return labels

    _gids = tuple(g["game_id"] for g in all_games)
    _ppaths = tuple(g["phases_path"] for g in all_games)
    game_labels = _build_game_labels(_gids, _ppaths)

    all_ids = [g["game_id"] for g in all_games]
    id_to_label = {gid: game_labels.get(gid, gid) for gid in all_ids}

    if st.button("✅ Select all games", key="select_all_games"):
        st.session_state["selected_game_ids"] = all_ids

    selected_ids = st.multiselect("Choose game(s) to load", options=all_ids, format_func=lambda gid: id_to_label.get(gid, gid), key="selected_game_ids")
    load_clicked = st.button("🚀 Load selected games", disabled=not selected_ids)

    if load_clicked and selected_ids:
        with st.spinner(f"Loading {len(selected_ids)} game(s)…"):
            all_phases, all_runs = [], []
            combined_map, descriptions = {}, []
            for gid in selected_ids:
                gmeta = game_by_id[gid]
                try:
                    minfo, ph_df, ru_df = load_game(gmeta)
                    all_phases.append(ph_df)
                    all_runs.append(ru_df)
                    combined_map.update(minfo.get("contestant_map", {}))
                    descriptions.append(minfo.get("description", gid))
                except Exception as exc:
                    st.warning(f"⚠️ Could not load {gid}: {exc}")

        if all_phases:
            merged_phases = pd.concat(all_phases, ignore_index=True, copy=False)
            merged_runs = pd.concat(all_runs, ignore_index=True, copy=False) if all_runs else pd.DataFrame()
            # Release intermediate lists
            del all_phases, all_runs
            st.session_state["phases_df"] = merged_phases
            st.session_state["runs_df"] = merged_runs
            st.session_state["match_info"] = {"match_id": "multi-game" if len(selected_ids) > 1 else selected_ids[0], "description": "; ".join(descriptions), "contestant_map": combined_map}
            st.session_state["squad_map"] = squad_map
            st.session_state["jersey_map"] = jersey_map
            st.session_state["loaded_game_ids"] = selected_ids
            st.session_state.pop("runs_committed", None)
            st.session_state.pop("pa_committed", None)
            st.session_state.pop("ba_committed", None)
            st.session_state.pop("tc_committed", None)
            # Clear stale cached computation results
            st.cache_data.clear()
            gc.collect()
            st.success(f"Loaded {len(merged_phases):,} phases and {len(merged_runs):,} runs from {len(selected_ids)} game(s).")
        else:
            st.error("No data could be loaded from the selected games.")

