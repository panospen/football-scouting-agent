"""
Football Scouting Agent — Streamlit UI (Enhanced Dark Theme)
"""

import streamlit as st
import logging
import os
import sys
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Football Scouting Agent",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ──────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Font ─────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Dark Background ─────────── */
    .stApp {
        background-color: #080c14;
    }
    header[data-testid="stHeader"] {
        background-color: #080c14;
    }
    section[data-testid="stSidebar"] {
        background-color: #0c1220;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    .block-container {
        padding-top: 1rem;
    }

    /* ── Hide defaults ───────────── */
    #MainMenu, footer {visibility: hidden;}

    /* ── Sidebar Title ───────────── */
    .sidebar-title {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 16px;
    }
    .sidebar-title-icon {
        font-size: 28px;
    }
    .sidebar-title-text {
        font-family: 'Inter', sans-serif;
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-title-sub {
        font-size: 11px;
        color: #475569;
    }

    /* ── Section Headers ─────────── */
    .section-label {
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        color: #475569;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 20px 0 8px 0;
    }

    /* ── Stat Cards ──────────────── */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 8px;
    }
    .stat-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 14px 12px;
        text-align: center;
    }
    .stat-num {
        font-family: 'Inter', sans-serif;
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-lbl {
        font-size: 9px;
        color: #475569;
        letter-spacing: 0.12em;
        margin-top: 4px;
    }

    /* ── Position Pills ──────────── */
    .pos-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .pos-pill {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 100px;
    }
    .pos-fwd { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
    .pos-mid { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
    .pos-def { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.2); }
    .pos-gk  { background: rgba(168,85,247,0.12); color: #a78bfa; border: 1px solid rgba(168,85,247,0.2); }

    /* ── League List ─────────────── */
    .league-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        border-radius: 8px;
        background: rgba(255,255,255,0.015);
        margin-bottom: 4px;
    }
    .league-name {
        font-size: 12px;
        color: #cbd5e1;
    }
    .league-season {
        font-size: 10px;
        color: #475569;
    }

    /* ── Query Buttons ───────────── */
    section[data-testid="stSidebar"] .stButton > button {
        font-family: 'Inter', sans-serif;
        background: rgba(0, 245, 160, 0.04) !important;
        color: #8892b0 !important;
        border: 1px solid rgba(0, 245, 160, 0.1) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0, 245, 160, 0.1) !important;
        color: #00f5a0 !important;
        border-color: rgba(0, 245, 160, 0.3) !important;
    }

    /* ── Sidebar Footer ──────────── */
    .sidebar-footer {
        font-size: 10px;
        color: #334155;
        line-height: 1.6;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 16px;
    }

    /* ── Hero Header ─────────────── */
    .hero {
        text-align: center;
        padding: 16px 0 8px;
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-sub {
        color: #5a6a8a;
        font-size: 14px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 245, 160, 0.06);
        border: 1px solid rgba(0, 245, 160, 0.15);
        color: #00f5a0;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 100px;
        margin-top: 8px;
        letter-spacing: 0.04em;
    }

    /* ── Chat Input ──────────────── */
    .stChatInput > div {
        background-color: rgba(255,255,255,0.02) !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(0, 245, 160, 0.3) !important;
        box-shadow: 0 0 12px rgba(0, 245, 160, 0.06) !important;
    }
    .stChatInput input {
        color: #e2e8f0 !important;
    }

    /* ── Chat Messages ───────────── */
    div[data-testid="stChatMessage"] {
        background-color: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px;
    }

    /* ── Metric display fix ─────── */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 12px;
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00f5a0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Tool label map ──────────────────────────────────────
TOOL_LABELS = {
    "search_players":          "Searching player database",
    "find_similar_players":    "Finding similar players",
    "compare_players":         "Comparing players",
    "generate_radar_chart":    "Generating radar chart",
    "get_player_stats":        "Fetching player stats",
    "generate_shot_map":       "Generating shot map",
    "generate_heatmap":        "Generating heatmap",
    "generate_pass_map":       "Generating pass map",
    "generate_scouting_report":"Generating scouting report",
}


# ── File snapshot helpers ────────────────────────────────
def _snapshot(directory: Path, pattern: str) -> set:
    if not directory.exists():
        return set()
    return {str(p) for p in directory.glob(pattern)}


def _new_files(before: set, directory: Path, pattern: str) -> list[str]:
    return sorted(_snapshot(directory, pattern) - before)


# ── PDF display helper ───────────────────────────────────
def display_pdf(pdf_path: Path):
    """Embed a single PDF with a download button."""
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(f"**Scouting Report: `{pdf_path.stem}`**")
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="700px" style="border:none;border-radius:8px;"></iframe>',
        height=720,
        scrolling=False,
    )
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=pdf_path.name,
        mime="application/pdf",
        key=f"dl_{pdf_path.name}_{int(os.path.getmtime(pdf_path))}",
    )


# ── Load data ───────────────────────────────────────────
from src.data.feature_store import FeatureStore

@st.cache_resource
def load_store():
    return FeatureStore()

store = load_store()
db = store.player_db


# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    # Title
    st.markdown("""
    <div class="sidebar-title">
        <span class="sidebar-title-icon">⚽</span>
        <div>
            <div class="sidebar-title-text">Scout Agent</div>
            <div class="sidebar-title-sub">AI-Powered Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not db.empty:
        # Stats
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-num">{len(db):,}</div>
                <div class="stat-lbl">PLAYERS</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{db['competition'].nunique()}</div>
                <div class="stat-lbl">COMPETITIONS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Positions
        st.markdown('<div class="section-label">Positions</div>', unsafe_allow_html=True)
        pos_counts = db["position_group"].value_counts()
        pos_html = '<div class="pos-row">'
        for group in ["FWD", "MID", "DEF", "GK"]:
            count = pos_counts.get(group, 0)
            css = {"FWD": "pos-fwd", "MID": "pos-mid", "DEF": "pos-def", "GK": "pos-gk"}.get(group, "")
            pos_html += f'<span class="pos-pill {css}">{group} {count}</span>'
        pos_html += '</div>'
        st.markdown(pos_html, unsafe_allow_html=True)

        # Leagues
        st.markdown('<div class="section-label">Leagues</div>', unsafe_allow_html=True)
        leagues_html = ""
        for comp in sorted(db["competition"].unique()):
            seasons = sorted(db[db["competition"] == comp]["season"].unique())
            s_str = ", ".join(seasons[:2])
            if len(seasons) > 2:
                s_str += f" +{len(seasons)-2}"
            leagues_html += f"""
            <div class="league-row">
                <span class="league-name">{comp}</span>
                <span class="league-season">{s_str}</span>
            </div>
            """
        st.markdown(leagues_html, unsafe_allow_html=True)

    # Quick queries
    st.markdown('<div class="section-label">Quick Queries</div>', unsafe_allow_html=True)

    queries = [
        ("🔍", "Top scorers in the Premier League"),
        ("🎯", "Find players similar to Salah"),
        ("📊", "Compare Messi, Neymar and Mbappé"),
        ("📈", "Generate a radar chart for Bellingham"),
        ("💎", "Young forwards in Bundesliga with high xG"),
        ("🛡️", "Top defensive midfielders in Serie A"),
    ]

    for icon, q in queries:
        if st.button(f"{icon}  {q}", key=f"btn_{q[:20]}", use_container_width=True):
            st.session_state.pending_query = q

    # Clear conversation
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state._reset_agent = True
        st.rerun()

    # Footer
    st.markdown("""
    <div class="sidebar-footer">
        Data: StatsBomb · FBref<br>
        LLM: Llama 3.3 via Groq
    </div>
    """, unsafe_allow_html=True)


# ── Agent ───────────────────────────────────────────────
from src.agent.agent import ScoutingAgent

@st.cache_resource
def load_agent():
    return ScoutingAgent()

try:
    agent = load_agent()
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()

if st.session_state.pop("_reset_agent", False):
    agent.reset()


# ── Header ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">⚽ Football Scouting Agent</div>
    <div class="hero-sub">AI-powered player analysis across Europe's top leagues</div>
    <div class="hero-badge">● LIVE 2024/25 · 3,000+ Players · 7 Competitions</div>
</div>
""", unsafe_allow_html=True)


# ── Chat ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Welcome! I'm your **AI Football Scouting Assistant** with data on "
            "**3,000+ players** across Europe's top leagues.\n\n"
            "I can help you:\n"
            "- 🔍 **Search** players by position, league, and stats\n"
            "- 🎯 **Find similar** players to any target\n"
            "- 📊 **Compare** players side-by-side\n"
            "- 📈 **Generate radar charts** for visual profiles\n\n"
            "Try: *\"Find me the best young forwards in the Premier League\"*"
        ),
    })

def _run_agent_with_status(user_input: str) -> tuple[str, list[str], list[str]]:
    """Run the agent with a live st.status panel. Returns (response, new_charts, new_pdfs)."""
    charts_dir = Path("data/charts")
    reports_dir = Path("data/reports")
    charts_before = _snapshot(charts_dir, "*.png")
    pdfs_before = _snapshot(reports_dir, "*.pdf")

    response = ""
    had_error = False
    with st.status("Analyzing...", expanded=True) as status:
        for step in agent.stream_steps(user_input):
            kind = step[0]
            if kind == "tool_call":
                label = TOOL_LABELS.get(step[1], f"Using {step[1]}")
                st.write(f"{label}...")
            elif kind == "error":
                st.warning(f"Stream error (falling back): {step[1]}")
                had_error = True
            elif kind == "response":
                response = step[1]
        state = "error" if (had_error and not response) else "complete"
        status.update(label="Done", state=state, expanded=False)

    st.markdown(response)

    new_charts = _new_files(charts_before, charts_dir, "*.png")
    new_pdfs = _new_files(pdfs_before, reports_dir, "*.pdf")

    for chart in new_charts:
        st.image(chart, width=480)
    for pdf_str in new_pdfs:
        display_pdf(Path(pdf_str))

    return response, new_charts, new_pdfs


# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for chart in message.get("charts", []):
            if Path(chart).exists():
                st.image(chart, width=480)
        for pdf_str in message.get("pdfs", []):
            pdf_path = Path(pdf_str)
            if pdf_path.exists():
                display_pdf(pdf_path)


# ── Pending query ───────────────────────────────────────
if "pending_query" in st.session_state:
    pending = st.session_state.pending_query
    del st.session_state.pending_query

    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user"):
        st.markdown(pending)

    with st.chat_message("assistant"):
        response, new_charts, new_pdfs = _run_agent_with_status(pending)

    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "charts": new_charts, "pdfs": new_pdfs,
    })
    st.rerun()


# ── Chat input ──────────────────────────────────────────
if prompt := st.chat_input("Ask about players, comparisons, or scouting..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response, new_charts, new_pdfs = _run_agent_with_status(prompt)

    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "charts": new_charts, "pdfs": new_pdfs,
    })