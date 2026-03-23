"""Small shared helpers used across multiple modules."""

from __future__ import annotations

import pandas as pd


def ms_to_mmss(val) -> str:
    """Convert a millisecond value to ``MM:SS`` string."""
    try:
        total_s = int(val) // 1000
    except (ValueError, TypeError):
        return str(val)
    m, s = divmod(abs(total_s), 60)
    sign = "-" if total_s < 0 else ""
    return f"{sign}{m}:{s:02d}"


def df_memory_mb(df: pd.DataFrame) -> float:
    """Return the memory footprint of *df* in megabytes (deep introspection)."""
    if df is None or df.empty:
        return 0.0
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def session_memory_report() -> dict[str, float]:
    """Return a dict of ``{key: size_mb}`` for every DataFrame in session state.

    Useful during development/debugging to understand memory usage.
    """
    import streamlit as st

    report: dict[str, float] = {}
    for key, val in st.session_state.items():
        if isinstance(val, pd.DataFrame):
            report[key] = df_memory_mb(val)
    return report


