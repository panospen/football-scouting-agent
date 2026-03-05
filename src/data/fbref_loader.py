"""
FBref Data Loader
Loads current-season player stats from a Kaggle FBref dataset (Excel/CSV).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FBrefLoader:
    """Loads current-season Big 5 Leagues player stats."""

    def __init__(self, data_path: str = "data/fbref_2425.xlsx"):
        self.data_path = Path(data_path)

    def load_and_process(self, min_minutes: int = 300) -> pd.DataFrame:
        """Load and process the FBref dataset.

        Args:
            min_minutes: Minimum minutes played to include

        Returns:
            Processed DataFrame ready to merge with StatsBomb data
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return pd.DataFrame()

        logger.info(f"Loading FBref data from {self.data_path}...")

        # Read file
        if str(self.data_path).endswith(".xlsx"):
            df = pd.read_excel(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        logger.info(f"Raw data: {len(df)} players")

        # Filter by minutes
        df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
        df = df[df["Min"] >= min_minutes].copy()
        logger.info(f"After {min_minutes} min filter: {len(df)} players")

        # Rename columns to match our StatsBomb format
        df = df.rename(columns={
            "Player": "player",
            "Squad": "team",
            "Comp": "competition",
            "Pos": "position_raw",
            "Min": "minutes_played",
            "MP": "matches_played",
            "90s": "nineties",
            "Gls": "goals",
            "Ast": "assists",
            "G+A": "goals_assists",
            "G-PK": "goals_non_penalty",
            "PK": "penalty_goals",
            "PKatt": "penalty_attempts",
            "CrdY": "yellow_cards",
            "CrdR": "red_cards",
            "xG": "xg",
            "npxG": "npxg",
            "xAG": "xag",
            "PrgC": "progressive_carries",
            "PrgP": "progressive_passes",
            "PrgR": "progressive_receptions",
            "Gls_90": "goals_per90",
            "Ast_90": "assists_per90",
            "G+A_90": "goals_assists_per90",
            "G-PK_90": "goals_non_penalty_per90",
            "xG_90": "xg_per90",
            "xAG_90": "xag_per90",
            "npxG_90": "npxg_per90",
        })

        # Map position groups
        df["position_group"] = df["position_raw"].apply(self._map_position_group)

        # Clean position for display
        df["position"] = df["position_raw"].apply(self._clean_position)

        # Clean competition names
        df["competition"] = df["competition"].apply(self._clean_competition)

        # Calculate additional per-90 stats from raw counts
        df["progressive_carries_per90"] = np.where(
            df["minutes_played"] > 0,
            df["progressive_carries"] / df["minutes_played"] * 90,
            0,
        )
        df["progressive_passes_per90"] = np.where(
            df["minutes_played"] > 0,
            df["progressive_passes"] / df["minutes_played"] * 90,
            0,
        )
        df["progressive_receptions_per90"] = np.where(
            df["minutes_played"] > 0,
            df["progressive_receptions"] / df["minutes_played"] * 90,
            0,
        )

        # Calculate percentiles within position groups
        per90_cols = [c for c in df.columns if c.endswith("_per90")]
        for col in per90_cols:
            pctl_col = col.replace("_per90", "_percentile")
            df[pctl_col] = df.groupby("position_group")[col].rank(pct=True) * 100

        # Add metadata
        df["season"] = "2024/2025"
        df["data_source"] = "FBref"

        logger.info(
            f"Processed FBref data: {len(df)} players, "
            f"{len(df.columns)} columns"
        )

        return df

    @staticmethod
    def _map_position_group(pos: str) -> str:
        """Map FBref position codes to groups."""
        if pd.isna(pos):
            return "Unknown"
        pos = str(pos).upper()
        if "GK" in pos:
            return "GK"
        elif "FW" in pos:
            return "FWD"
        elif "MF" in pos:
            return "MID"
        elif "DF" in pos:
            return "DEF"
        return "Unknown"

    @staticmethod
    def _clean_position(pos: str) -> str:
        """Convert FBref position codes to readable names."""
        if pd.isna(pos):
            return "Unknown"
        mapping = {
            "GK": "Goalkeeper",
            "DF": "Defender",
            "DF,MF": "Defensive Midfielder",
            "MF,DF": "Defensive Midfielder",
            "MF": "Midfielder",
            "MF,FW": "Attacking Midfielder",
            "FW,MF": "Forward/Midfielder",
            "FW": "Forward",
            "DF,FW": "Defender/Forward",
            "FW,DF": "Forward/Defender",
        }
        return mapping.get(pos, pos)

    @staticmethod
    def _clean_competition(comp: str) -> str:
        """Clean competition names."""
        if pd.isna(comp):
            return "Unknown"
        mapping = {
            "eng Premier League": "Premier League",
            "es La Liga": "La Liga",
            "it Serie A": "Serie A",
            "de Bundesliga": "Bundesliga",
            "fr Ligue 1": "Ligue 1",
        }
        return mapping.get(comp, comp)


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = FBrefLoader()
    stats = loader.load_and_process(min_minutes=300)

    if not stats.empty:
        print(f"\nTotal players: {len(stats)}")
        print(f"\nCompetitions:")
        print(stats["competition"].value_counts())
        print(f"\nPosition groups:")
        print(stats["position_group"].value_counts())
        print(f"\nTop 10 scorers (goals_per90):")
        top = stats.nlargest(10, "goals_per90")[
            ["player", "team", "competition", "minutes_played",
             "goals", "goals_per90", "xg_per90"]
        ]
        print(top.to_string(index=False))
