"""
Football Scouting Agent — Streamlit UI
Interactive web interface for the AI scouting assistant.
"""

import streamlit as st
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent.agent import ScoutingAgent
from src.data.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Football Scouting Agent",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
    }
    .sidebar-info {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #1e1e2e;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Scouting Agent")
    st.markdown("---")
    
    st.markdown("### 📊 Database Info")
    
    @st.cache_resource
    def load_store():
        return FeatureStore()
    
    store = load_store()
    db = store.player_db
    
    if not db.empty:
        st.metric("Total Players", len(db))
        st.metric("Competitions", db["competition"].nunique())
        
        st.markdown("### 🏆 Competitions")
        for comp in db["competition"].unique():
            seasons = db[db["competition"] == comp]["season"].unique()
            st.markdown(f"**{comp}**: {', '.join(sorted(seasons))}")
        
        st.markdown("### 📋 Position Groups")
        for group, count in db["position_group"].value_counts().items():
            st.markdown(f"**{group}**: {count} players")
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    example_queries = [
        "Find me the top scoring forwards in La Liga",
        "Who are similar players to Messi?",
        "Compare Messi, Neymar and Mbappé",
        "Show me the best dribblers in the World Cup",
        "Find defensive midfielders with high pass completion",
        "Generate a radar chart for Busquets",
        "Find a left winger with good dribbling and goals",
    ]
    
    for query in example_queries:
        if st.button(query, key=f"btn_{query[:20]}", use_container_width=True):
            st.session_state.pending_query = query
    
    st.markdown("---")
    st.markdown(
        "Built with ⚽ + 🤖\n\n"
        "Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data)\n\n"
        "LLM: Llama 3.3 via [Groq](https://groq.com)"
    )

# ── Initialize agent ────────────────────────────────────
@st.cache_resource
def load_agent():
    return ScoutingAgent()

try:
    agent = load_agent()
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()

# ── Chat interface ──────────────────────────────────────
st.markdown("<h1 class='main-header'>⚽ AI Football Scouting Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "Ask me anything about players, comparisons, or scouting recommendations"
    "</p>",
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I'm your AI Football Scouting Assistant. "
            "I have data on 850+ players from La Liga, Champions League, "
            "World Cup, and Euro competitions.\n\n"
            "I can help you:\n"
            "- 🔍 **Search** for players by position, stats, and competition\n"
            "- 🎯 **Find similar** players to any player\n"
            "- 📊 **Compare** players side-by-side\n"
            "- 📈 **Generate radar charts** for visual profiles\n\n"
            "Try asking something like: *'Find me creative wingers from La Liga with good dribbling'*"
        ),
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display radar chart if path is in message
        if "radar chart saved to:" in message.get("content", "").lower():
            import re
            paths = re.findall(r'data[/\\]charts[/\\]\S+\.png', message["content"])
            for path in paths:
                if Path(path).exists():
                    st.image(path, width=500)

# ── Handle pending query from sidebar buttons ───────────
if "pending_query" in st.session_state:
    pending = st.session_state.pending_query
    del st.session_state.pending_query
    
    st.session_state.messages.append({"role": "user", "content": pending})
    
    with st.chat_message("user"):
        st.markdown(pending)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            response = agent.chat(pending)
        st.markdown(response)
        
        # Show any new radar charts
        charts_dir = Path("data/charts")
        if charts_dir.exists():
            import time
            for chart in sorted(charts_dir.glob("*.png"), key=os.path.getmtime, reverse=True):
                if time.time() - os.path.getmtime(chart) < 30:
                    st.image(str(chart), width=500)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Chat input ──────────────────────────────────────────
if prompt := st.chat_input("Ask about players, comparisons, or scouting..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            response = agent.chat(prompt)
        st.markdown(response)
        
        # Show any new radar charts
        charts_dir = Path("data/charts")
        if charts_dir.exists():
            for chart in sorted(charts_dir.glob("*.png"), key=os.path.getmtime, reverse=True):
                # Show charts created in the last 30 seconds
                import time
                if time.time() - os.path.getmtime(chart) < 30:
                    st.image(str(chart), width=500)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
