"""Small shared helpers used across multiple modules."""

from __future__ import annotations


def ms_to_mmss(val) -> str:
    """Convert a millisecond value to ``MM:SS`` string."""
    try:
        total_s = int(val) // 1000
    except (ValueError, TypeError):
        return str(val)
    m, s = divmod(abs(total_s), 60)
    sign = "-" if total_s < 0 else ""
    return f"{sign}{m}:{s:02d}"

