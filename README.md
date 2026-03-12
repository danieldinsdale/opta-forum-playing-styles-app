# ai-playing-styles-app

Streamlit application for exploring **Phases of Play** and **Player Runs** data feeds.

---

## Local setup

### Prerequisites
- Python 3.11+
- Access to the Phase and Run data for upload

### 1. Create & activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run streamlit_phases_xml.py
```

The app will open at <http://localhost:8501>.

---

## Project structure

```
.
├── streamlit_phases_xml.py   # Main Streamlit app
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for Streamlit Cloud
├── .streamlit/
│   ├── config.toml           # Server & theme settings
│   └── secrets.toml          # Local secrets (gitignored)
└── feeds/                    # JSON data feeds (gitignored)
    └── <competition_id>/
        ├── squad_lists.json
        └── remote/non_aggregated/
            ├── phases/
            └── runs/
```
