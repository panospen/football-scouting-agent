"""
Compare Players Tool
Side-by-side statistical comparison of 2-4 players.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.data.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class PlayerComparison:
    """Compare multiple players side-by-side."""

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store

    def compare(
        self,
        player_names: list,
        metrics: Optional[list] = None,
    ) -> pd.DataFrame:
        """Compare 2-4 players on key metrics.

        Args:
            player_names: List of player names (2-4)
            metrics: Specific per-90 metrics to compare.
                    Auto-selects if None.

        Returns:
            DataFrame with metrics as rows, players as columns
        """
        db = self.store.player_db

        if len(player_names) < 2 or len(player_names) > 4:
            logger.warning("Please provide 2-4 players to compare!")
            return pd.DataFrame()

        # Find each player
        import unicodedata

        def remove_accents(text):
            if pd.isna(text):
                return ""
            nfkd = unicodedata.normalize("NFKD", str(text))
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        players = []
        for name in player_names:
            mask = db["player"].str.contains(name, case=False, na=False)

            # Try without accents if not found
            if not mask.any():
                clean_name = remove_accents(name).lower()
                mask = db["player"].apply(
                    lambda x: clean_name in remove_accents(x).lower()
                )

            if not mask.any():
                logger.warning(f"Player '{name}' not found!")
                continue

            # Prefer current season
            matches = db[mask]
            current = matches[matches.get("season", "") == "2024/2025"]
            if not current.empty:
                players.append(current.iloc[0])
            else:
                players.append(matches.iloc[0])

        if len(players) < 2:
            logger.warning("Need at least 2 valid players!")
            return pd.DataFrame()

        # Auto-select metrics if not provided
        if metrics is None:
            metrics = self._auto_select_metrics(players)

        # Build comparison table
        comparison = {}
        for player in players:
            full_name = player['player']
            name_parts = full_name.split()
            if len(name_parts) <= 2:
                short_name = full_name
            else:
                short_name = f"{name_parts[0]} {name_parts[-1]}"
            display_name = f"{short_name} ({player['team']})"
            values = {}

            # Metadata
            values["Position"] = player["position"]
            values["Competition"] = player["competition"]
            values["Minutes"] = int(player["minutes_played"])
            values["Matches"] = int(player["matches_played"])

            # Stats
            for metric in metrics:
                if metric in player.index:
                    val = player[metric]
                    if isinstance(val, (int, np.integer)):
                        values[self._pretty_name(metric)] = int(val)
                    else:
                        values[self._pretty_name(metric)] = round(float(val), 3)
                else:
                    values[self._pretty_name(metric)] = "N/A"

            comparison[display_name] = values

        result = pd.DataFrame(comparison)
        return result

    def format_comparison(self, comparison: pd.DataFrame) -> str:
        """Format comparison as readable string with highlights."""
        if comparison.empty:
            return "Could not generate comparison."

        output = "Player Comparison\n"
        output += "=" * 60 + "\n\n"
        output += comparison.to_string()
        output += "\n\n"

        # Add highlights — who leads in each metric
        output += "Key Highlights:\n"
        numeric_rows = comparison.iloc[4:]  # Skip metadata rows

        for metric_name in numeric_rows.index:
            row = numeric_rows.loc[metric_name]
            numeric_vals = pd.to_numeric(row, errors="coerce")
            if numeric_vals.notna().any():
                best_player = numeric_vals.idxmax()
                best_val = numeric_vals.max()
                output += f"  {metric_name}: {best_player} leads ({best_val})\n"

        return output

    def _auto_select_metrics(self, players: list) -> list:
        """Auto-select relevant per-90 metrics based on position groups."""
        groups = set(p["position_group"] for p in players)

        # Common metrics for all
        metrics = [
            "goals_per90", "xg_per90", "assists_per90",
            "total_passes_per90", "pass_completion_pct",
        ]

        if "FWD" in groups:
            metrics += [
                "total_shots_per90", "dribbles_completed_per90",
                "progressive_carries_per90",
            ]

        if "MID" in groups:
            metrics += [
                "progressive_passes_per90",
                "tackles_won_per90", "ball_recoveries_per90",
            ]

        if "DEF" in groups:
            metrics += [
                "tackles_won_per90", "interceptions_per90",
                "clearances_per90", "blocks_per90",
            ]

        metrics += ["pressures_per90"]

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for m in metrics:
            if m not in seen:
                seen.add(m)
                unique.append(m)

        return unique

    @staticmethod
    def _pretty_name(metric: str) -> str:
        """Convert metric column name to readable label."""
        name = metric.replace("_per90", "/90").replace("_", " ").title()
        name = name.replace("Xg", "xG").replace("Pct", "%")
        return name


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = FeatureStore()
    comp = PlayerComparison(store)

    print("=== Messi vs Neymar vs Mbappé ===\n")
    result = comp.compare(["Messi", "Neymar", "Mbappé"])
    print(comp.format_comparison(result))

    print("\n\n=== Busquets vs Kroos ===\n")
    result = comp.compare(["Busquets", "Kroos"])
    print(comp.format_comparison(result))
