"""Header rendering and CSS injection."""
from __future__ import annotations
import base64
from functools import lru_cache
from pathlib import Path
import streamlit as st
from src.config import (
    LOGOS_DIR,
    BRAND_PURPLE, BRAND_AMBER, BRAND_ORANGE, BRAND_RED,
    FONT_HEADLINE, FONT_WIDE, FONT_TITLE, FONT_BODY, FONT_GFX_URL,
)

LOGO_DARK = LOGOS_DIR / "opta-ai-logo_white.png"
LOGO_LIGHT = LOGOS_DIR / "opta-ai-logo_black.png"


@lru_cache(maxsize=4)
def logo_b64(path: str | Path) -> str | None:
    try:
        data = Path(path).read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode()
    except (FileNotFoundError, OSError):
        return None


def render_header() -> None:
    """Inject brand CSS and render the top banner with logo + app title."""
    logo_uri = logo_b64(LOGO_DARK) or logo_b64(LOGO_LIGHT) or ""
    logo_html = (
        f'<img src="{logo_uri}" alt="Opta AI" style="height:44px;display:block;">'
        if logo_uri else ""
    )

    st.markdown(
        f"""
<style>
@import url('{FONT_GFX_URL}');
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"], p, li, label,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
    font-family: '{FONT_BODY}', sans-serif !important; font-weight: 300 !important;
}}
.material-icons, .material-icons-outlined, .material-icons-round, .material-icons-sharp,
[class*="material-symbols"] {{
    font-family: 'Material Icons', 'Material Symbols Outlined', 'Material Symbols Rounded' !important;
    font-weight: normal !important;
}}
h1, [data-testid="stMarkdownContainer"] h1 {{
    font-family: '{FONT_HEADLINE}', sans-serif !important; font-weight: 900 !important;
    letter-spacing: 0.01em !important; text-transform: uppercase !important;
}}
h2, [data-testid="stMarkdownContainer"] h2 {{
    font-family: '{FONT_WIDE}', sans-serif !important; font-weight: 800 !important;
    font-stretch: expanded !important; letter-spacing: 0.04em !important;
}}
h3, h4, [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{
    font-family: '{FONT_TITLE}', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}}
strong, b, [data-testid="stMarkdownContainer"] strong {{
    font-family: '{FONT_TITLE}', sans-serif !important; font-weight: 600 !important;
}}
[data-testid="stWidgetLabel"] p, label {{
    font-family: '{FONT_TITLE}', sans-serif !important; font-weight: 600 !important;
}}
button, .stButton > button {{
    font-family: '{FONT_TITLE}', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: '{FONT_TITLE}', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}}
code, pre {{ font-family: 'SFMono-Regular', 'Consolas', monospace !important; font-weight: 400 !important; }}
.brand-header-title {{
    font-family: '{FONT_HEADLINE}', sans-serif !important; font-weight: 900 !important;
    letter-spacing: 0.02em !important; text-transform: uppercase !important;
}}
.brand-header-sub {{
    font-family: '{FONT_WIDE}', sans-serif !important; font-weight: 800 !important;
    font-stretch: expanded !important; letter-spacing: 0.06em !important; text-transform: uppercase !important;
}}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"] {{ background-color: #000000 !important; color: #f0f0f0 !important; }}
section[data-testid="stSidebar"] {{ background-color: #0d0d0d !important; }}
section[data-testid="stSidebar"] * {{ color: #f0f0f0 !important; }}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: {BRAND_AMBER} !important; }}
.brand-header {{
    display: flex; align-items: center; gap: 20px; background: #000000;
    padding: 14px 28px; border-radius: 8px; margin-bottom: 18px;
    border-bottom: 3px solid {BRAND_AMBER};
}}
.brand-header-title {{ color: #ffffff; font-size: 1.45rem; font-weight: 700;
    letter-spacing: 0.01em; line-height: 1.2; margin: 0; }}
.brand-header-sub {{ color: {BRAND_AMBER}; font-size: 0.82rem; font-weight: 500;
    margin: 0; letter-spacing: 0.04em; text-transform: uppercase; }}
[data-testid="stExpander"] {{ background-color: #0d0d0d !important; border: 1px solid #2a2a2a !important; border-radius: 6px !important; }}
[data-testid="stExpander"] summary {{ color: #f0f0f0 !important; }}
input, textarea, select, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background-color: #1a1a1a !important; color: #f0f0f0 !important; border-color: #333333 !important;
}}
[data-testid="stMultiSelect"] div[data-baseweb="select"] {{ background-color: #1a1a1a !important; color: #f0f0f0 !important; }}
[data-testid="stMultiSelect"] div[data-baseweb="select"] div {{ background-color: transparent !important; color: #f0f0f0 !important; }}
[data-testid="stMultiSelect"] input {{ background-color: transparent !important; color: #f0f0f0 !important; }}
[data-baseweb="tag"] {{ background-color: {BRAND_PURPLE} !important; color: #ffffff !important; }}
[data-baseweb="tag"] span, [data-baseweb="tag"] * {{ color: #ffffff !important; background-color: transparent !important; }}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul,
[data-baseweb="popover"] li {{ background-color: #1a1a1a !important; color: #f0f0f0 !important; }}
[data-baseweb="popover"] li:hover {{ background-color: #333333 !important; }}
[data-testid="stDataFrame"], [data-testid="stDataFrame"] iframe, .stDataFrame {{
    background-color: #0d0d0d !important; border-top: 2px solid {BRAND_AMBER} !important; border-radius: 6px !important;
}}
.stTabs [data-baseweb="tab-list"] {{ background-color: #000000 !important; border-bottom: 1px solid #2a2a2a !important; }}
.stTabs [data-baseweb="tab"] {{ color: #bbbbbb !important; }}
.stTabs [aria-selected="true"] {{ color: {BRAND_AMBER} !important; border-bottom-color: {BRAND_AMBER} !important; }}
.stButton > button[kind="primary"] {{ background-color: {BRAND_PURPLE} !important; border-color: {BRAND_PURPLE} !important; color: #ffffff !important; }}
.stButton > button[kind="primary"]:hover {{ background-color: {BRAND_ORANGE} !important; border-color: {BRAND_ORANGE} !important; }}
.stButton > button[kind="secondary"] {{ background-color: #1a1a1a !important; border-color: #444444 !important; color: #f0f0f0 !important; }}
.stButton > button[kind="secondary"]:hover {{ border-color: {BRAND_AMBER} !important; color: {BRAND_AMBER} !important; }}
[data-testid="stAlert"] {{ background-color: #1a1a1a !important; border-radius: 6px !important; border-left: 4px solid {BRAND_AMBER} !important; }}
[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) {{ border-left-color: {BRAND_ORANGE} !important; }}
[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) {{ border-left-color: {BRAND_RED} !important; }}
[data-testid="stMetric"] {{
    background-color: #0d0d0d !important; border-radius: 6px !important;
    padding: 8px !important; border-left: 3px solid {BRAND_AMBER} !important;
}}
.stProgress > div > div > div {{ background-color: {BRAND_PURPLE} !important; }}
.stSpinner > div {{ border-top-color: {BRAND_AMBER} !important; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #000000; }}
::-webkit-scrollbar-thumb {{ background: #333333; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {BRAND_AMBER}; }}
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

