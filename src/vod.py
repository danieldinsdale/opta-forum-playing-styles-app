"""VOD streaming helpers — API key retrieval and clip URL fetching."""

from __future__ import annotations

import json
import os

import requests
import streamlit as st


def get_vod_api_key() -> str:
    """Return the VOD API key from env var, Streamlit secrets, or sidebar input."""
    return (
        os.environ.get("VOD_API_KEY", "")
        or st.secrets.get("VOD_API_KEY", "")
        or st.session_state.get("_vod_k", "")
    )


def get_vod_base_url() -> str:
    """Return the VOD base URL from env var or Streamlit secrets only."""
    return (
        os.environ.get("VOD_BASE_URL", "")
        or st.secrets.get("VOD_BASE_URL", "")
    )


@st.cache_data(ttl=300, show_spinner=False, max_entries=5)
def get_vod_streaming(
    game_uuid: str,
    period: int,
    time_in: int,
    time_out: int,
    before_time: int = 0,
    after_time: int = 0,
    api_key: str = "",
) -> str:
    """Retrieve a streaming URL from the VOD StreamingLinks API."""
    if not api_key:
        raise ValueError(
            "VOD API key is not configured. "
            "Set the VOD_API_KEY environment variable or enter it in the sidebar."
        )

    base_url = get_vod_base_url()
    if not base_url:
        raise ValueError(
            "VOD base URL is not configured. "
            "Set the VOD_BASE_URL environment variable or Streamlit secret."
        )

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
        raise requests.exceptions.HTTPError(
            f"{exc.response.status_code} {exc.response.reason}",
            response=exc.response,
        ) from None
    except requests.exceptions.RequestException as exc:
        raise requests.exceptions.RequestException(type(exc).__name__) from None

    url = json.loads(res.text)[0].get("http", "").encode("ascii", "ignore").decode()
    if not url:
        raise ValueError("The API returned an empty streaming URL.")
    return url

