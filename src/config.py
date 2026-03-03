"""
Project configuration and constants.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# ── API Keys ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Data Settings ──
MIN_MINUTES = 450  # ~5 full matches minimum
DEFAULT_COMPETITION = "La Liga"
DEFAULT_SEASON = "2020/2021"

# ── Model Settings (Phase 3) ──
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048