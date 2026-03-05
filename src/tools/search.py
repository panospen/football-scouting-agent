"""
Search Players Tool
Filters the player database based on multiple criteria.
"""

import logging
from typing import Optional

import pandas as pd

from src.data.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class PlayerSearch:
    """Search and filter players from the database."""

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store

    def search(
        self,
        position: Optional[str] = None,
        position_group: Optional[str] = None,
        team: Optional[str] = None,
        competition: Optional[str] = None,
        min_minutes: Optional[int] = None,
        max_minutes: Optional[int] = None,
        min_goals_per90: Optional[float] = None,
        min_assists_per90: Optional[float] = None,
        min_xg_per90: Optional[float] = None,
        min_dribbles_completed_per90: Optional[float] = None,
        min_progressive_passes_per90: Optional[float] = None,
        min_interceptions_per90: Optional[float] = None,
        min_tackles_won_per90: Optional[float] = None,
        top_n: int = 20,
        sort_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """Search players with flexible filters.

        Args:
            position: Exact position (e.g., "Left Wing", "Center Forward")
            position_group: Group (FWD, MID, DEF, GK)
            team: Team name (partial match)
            competition: Competition name
            min_minutes: Minimum minutes played
            min_goals_per90: Minimum goals per 90
            ... etc
            top_n: Number of results to return
            sort_by: Column to sort by (descending)

        Returns:
            Filtered DataFrame with matching players
        """
        db = self.store.player_db.copy()

        if db.empty:
            logger.warning("Player database is empty!")
            return db

        # Apply filters
        if position:
            db = db[db["position"].str.contains(position, case=False, na=False)]

        if position_group:
            db = db[db["position_group"] == position_group.upper()]

        if team:
            import unicodedata
            def remove_accents(text):
                if pd.isna(text):
                    return ""
                nfkd = unicodedata.normalize("NFKD", str(text))
                return "".join(c for c in nfkd if not unicodedata.combining(c))

            clean_team = remove_accents(team).lower()
            db = db[db["team"].apply(
                lambda x: clean_team in remove_accents(x).lower()
            )]

        if competition:
            db = db[db["competition"].str.contains(competition, case=False, na=False)]

        if min_minutes:
            db = db[db["minutes_played"] >= min_minutes]

        if max_minutes:
            db = db[db["minutes_played"] <= max_minutes]

        # Per-90 metric filters
        metric_filters = {
            "goals_per90": min_goals_per90,
            "assists_per90": min_assists_per90,
            "xg_per90": min_xg_per90,
            "dribbles_completed_per90": min_dribbles_completed_per90,
            "progressive_passes_per90": min_progressive_passes_per90,
            "interceptions_per90": min_interceptions_per90,
            "tackles_won_per90": min_tackles_won_per90,
        }

        # Try alternative column names for FBref data
        alt_metric_filters = {
            "progressive_carries_per90": min_dribbles_completed_per90,
        }
        for col, min_val in alt_metric_filters.items():
            if min_val is not None and col in db.columns and "dribbles_completed_per90" not in db.columns:
                db = db[db[col] >= min_val]

        for col, min_val in metric_filters.items():
            if min_val is not None and col in db.columns:
                db = db[db[col] >= min_val]

        # Sort
        if sort_by and sort_by in db.columns:
            db = db.sort_values(sort_by, ascending=False)
        elif "goals_per90" in db.columns:
            db = db.sort_values("goals_per90", ascending=False)

        return db.head(top_n)

    def format_results(self, results: pd.DataFrame) -> str:
        """Format search results as a readable string for the agent."""
        if results.empty:
            return "No players found matching the criteria."

        output = f"Found {len(results)} players:\n\n"

        display_cols = [
            "player", "team", "position", "competition",
            "minutes_played", "matches_played",
        ]

        # Add per-90 columns that exist
        per90_cols = [c for c in results.columns if c.endswith("_per90")]
        key_per90 = [
            "goals_per90", "xg_per90", "assists_per90",
            "total_passes_per90", "progressive_passes_per90",
            "dribbles_completed_per90", "tackles_won_per90",
            "interceptions_per90", "pressures_per90",
        ]
        display_per90 = [c for c in key_per90 if c in per90_cols]

        cols = [c for c in display_cols + display_per90 if c in results.columns]
        display = results[cols].copy()

        # Round per-90 stats
        for col in display_per90:
            if col in display.columns:
                display[col] = display[col].round(3)

        output += display.to_string(index=False)
        return output


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = FeatureStore()
    search = PlayerSearch(store)

    print("=== Search: Forwards from La Liga ===")
    results = search.search(
        position_group="FWD",
        competition="La Liga",
        sort_by="goals_per90",
        top_n=10,
    )
    print(search.format_results(results))

    print("\n=== Search: High dribbling wingers ===")
    results = search.search(
        position="Wing",
        min_dribbles_completed_per90=2.0,
        sort_by="dribbles_completed_per90",
        top_n=10,
    )
    print(search.format_results(results))
