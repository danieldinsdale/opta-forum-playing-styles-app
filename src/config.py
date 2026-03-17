"""
Centralised configuration — brand colours, fonts, paths.

Every other ``src.*`` module imports constants from here rather than
reading config.toml or inspecting the file-system itself.
"""

from __future__ import annotations

from pathlib import Path

import tomli

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parent.parent
FEEDS_BASE = REPO_ROOT / "feeds"
LOGOS_DIR  = REPO_ROOT / "logos"

# ──────────────────────────────────────────────────────────────────────────────
# config.toml loader
# ──────────────────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Read the full .streamlit/config.toml and return it as a dict."""
    config_path = REPO_ROOT / ".streamlit" / "config.toml"
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except Exception:
        return {}


CFG = _load_config()

# ──────────────────────────────────────────────────────────────────────────────
# Brand colours
# ──────────────────────────────────────────────────────────────────────────────

_BRAND = CFG.get("brand", {})

BRAND_PRIMARY = _BRAND.get("primary", "#222222")
BRAND_PURPLE  = _BRAND.get("purple",  "#9E07AE")
BRAND_AMBER   = _BRAND.get("amber",   "#FAA51A")
BRAND_ORANGE  = _BRAND.get("orange",  "#F06424")
BRAND_RED     = _BRAND.get("red",     "#E5202F")

# ──────────────────────────────────────────────────────────────────────────────
# Fonts
# ──────────────────────────────────────────────────────────────────────────────

_FONTS = CFG.get("fonts", {})

FONT_HEADLINE = _FONTS.get("headline_font",  "Barlow Condensed")
FONT_WIDE     = _FONTS.get("wide_font",      "Barlow")
FONT_TITLE    = _FONTS.get("title_font",     "Barlow")
FONT_BODY     = _FONTS.get("body_font",      "Barlow")
FONT_GFX_URL  = _FONTS.get(
    "google_fonts_url",
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@900"
    "&family=Barlow:wght@300;400;600;800&display=swap",
)

# ──────────────────────────────────────────────────────────────────────────────
# Available local competition directories
# ──────────────────────────────────────────────────────────────────────────────

COMPETITION_DIRS: list[str] = (
    sorted(p.name for p in FEEDS_BASE.iterdir() if p.is_dir())
    if FEEDS_BASE.exists()
    else []
)

