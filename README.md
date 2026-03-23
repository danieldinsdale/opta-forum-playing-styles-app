# Playing Styles – Opta Forum Feed Analysis

Streamlit application for exploring **Phases of Play** and **Player Runs** data feeds for the Opta Forum 2026

---

## Local setup

### Prerequisites
- Python 3.11+
- Access to Phases and Runs JSON feeds

### 1. Create & activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Add a VOD API key

Create `.streamlit/secrets.toml`:

```toml
VOD_API_KEY = "your-key-here"
VOD_BASE_URL = "https://your-vod-endpoint/StreamingLinks"
```

### 4. Run the app

```bash
streamlit run streamlit_phases_xml.py
```

The app will open at <http://localhost:8501>.

---

## Features

### 🏃 Runs Analysis
Search and visualise player runs linked to the phases they occurred within.

- **Run Criteria** — filter by Master Label, Expected Threat range (slider), Max Speed (m/s) range (slider), Defensive Line Broken, Dangerous, and Run Type
- **Phase Criteria** — filter by Phase Labels (multi-select with Select All / Clear), Includes Shots, Includes Goal
- **Run Coordinates** — filter by start and end zone on an interactive pitch map (0–100 coordinate system)
- **Attacking Outcomes** — filter In Possession runs by "Followed by Team Shot" and "Followed by Team Goal"
- **Team / Player** — scope results to a specific team and player
- **Individual Runs table** — shows Game ID, Run ID, Phase ID, Period, Start/End time (mm:ss), Master Label, Phase Label, Player Jersey number, player name, and more; IDs sort as integers
- **Pitch Map** — interactive Plotly pitch with direction arrows, colour-coded by team; selectable run table with VOD clip generation
- **Video playback** — generate a VOD clip for any individual run from either the table or pitch map view (requires VOD API key)

### 📊 Phase Analysis
Filter and aggregate phases across multiple dimensions.

- **Phase Labels** — multi-select or "Leads to" sequence mode (find phases immediately followed by a chosen label)
- **Coordinates** — filter phases by start and/or end zone on the pitch
- **Action Counts** — range sliders for Dangerous Runs, Line-Breaking Actions, Passes, High Pressure on Receiver/Touches (3-column layout)
- **Attacking Outcomes** — filter by Includes Shots / Includes Goal
- **Compactness** — filter by `mostCommonDefensiveCompactness` label; sliders for Attacking Team Width (0–68 m), Length & Height of Last Defender (0–105 m); same for Defending Team
- **Team / Player** — filter by possessing team and initiator / first-touch player
- **Phase List** — table with Phase ID, Period, Start/End time (mm:ss), Phase Label, Initiator Jersey & name, and compactness metrics
- **Aggregation** — time-weighted summaries and bar charts per team or player
- **Video playback** — generate a VOD clip for any phase row

### 🧱 Block Analysis
Analyse the proportion of Build Up time each team spends against each block type (Low / Medium / High), and how much time they deploy each block against opponents.

- **Block Faced** vs **Block Deployed** toggle (default: Deployed)
- **Game State** range slider — filter by goal difference from each team's perspective
- **Phase Labels** filter inside the Filters expander
- **Metrics** — count or time (seconds), stacked bar or grouped bar visualisation
- Default metric: **Time (seconds)**

### 📏 Team Compactness
Time-weighted team shape statistics derived from phase-level compactness data.

- **In Possession** — Attacking Team Horizontal Width, Vertical Length, Height of Last Defender (m); horizontal bar chart with metres on the x-axis
- **Out of Possession** — Defending Team Horizontal Width, Vertical Length, Height of Last Defender; toggle to show **Defensive Area Coverage (m²)** instead; horizontal bar chart
- **Phase Labels** filter — multi-select or "Leads to" sequence mode (calculation uses only the 1st phase of each pair)
- **Possession State** filter
- Summary data table below each chart

---

## Data loading

The app supports two data-loading modes selectable in the sidebar:

### 📤 Upload mode (ZIP)
Upload a competition folder as a `.zip` file. The ZIP must contain the standard feed directory structure:

```
<competition_id>/
├── squad_lists.json
└── remote/non_aggregated/
    ├── phases/   ← *_phase.json files
    └── runs/     ← *_run.json files
```

Multiple games can be selected and loaded together. Squad lists are detected automatically.

### 📁 Local feeds directory
If a `feeds/` directory is present at the repo root, the app exposes a competition selector and game multi-select backed by the local files. This mode is hidden when no local feeds are found.

---

## Video on Demand (VOD)

The Phase Analysis and Runs Analysis tabs include a **Generate Video** button per row. This requires:

| Setting | How to provide |
|---|---|
| `VOD_API_KEY` | Environment variable, `secrets.toml`, or the sidebar password input |
| `VOD_BASE_URL` | Environment variable or `secrets.toml` only (never exposed in the UI) |

Pre- and post-buffers are capped at **10 seconds** each.

---

## Project structure

```
.
├── streamlit_phases_xml.py   # App entry-point
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version pin for Streamlit Cloud
├── logos/                    # Brand logo assets
├── .streamlit/
│   ├── config.toml           # Theme, fonts, brand colours
│   └── secrets.toml          # Local secrets (gitignored)
├── feeds/                    # JSON data feeds (gitignored)
│   └── <competition_id>/
│       ├── squad_lists.json
│       └── remote/non_aggregated/
│           ├── phases/       ← *_phase.json
│           └── runs/         ← *_run.json
└── src/
    ├── config.py             # Brand colours, fonts, paths (reads config.toml)
    ├── data_loading.py       # Feed discovery, game loading, ZIP handling
    ├── parsers.py            # JSON/XML feed parsers + dtype optimiser
    ├── pitch.py              # Pitch shapes, pitch-map renderer, zone selector
    ├── sidebar.py            # Upload-mode and local-mode sidebar UI
    ├── tab_blocks.py         # Block Analysis tab
    ├── tab_compactness.py    # Team Compactness tab
    ├── tab_phases.py         # Phase Analysis tab
    ├── tab_runs.py           # Runs Analysis tab
    ├── ui.py                 # Header rendering, brand CSS injection
    ├── utils.py              # Shared helpers (ms→mm:ss, memory stats)
    └── vod.py                # VOD API key retrieval and clip URL fetching
```

---

## Memory & performance

- DataFrame columns unused downstream are dropped at load time (`_PHASE_DROP_COLS` in `data_loading.py`) to reduce memory footprint
- All numeric columns are downcast to the smallest safe type (`Int16`, `Int32`, `float32`)
- Low-cardinality string columns are stored as `category` dtype
- Heavy computation in each tab is cached with `@st.cache_data` and only triggered when the user clicks **▶ Generate Outputs**
- Each tab is wrapped in `@st.fragment` so filter interactions in one tab do not rerun other tabs
- A **🧠 Memory Usage** panel in the sidebar shows the current phases and runs DataFrame sizes in MB
