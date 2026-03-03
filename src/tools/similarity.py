"""
Player Similarity Tool
Finds the most similar players using cosine similarity on per-90 stats.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.data.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Metrics used for similarity by position group
SIMILARITY_METRICS = {
    "FWD": [
        "goals_per90", "xg_per90", "total_shots_per90",
        "assists_per90", "dribbles_completed_per90",
        "progressive_carries_per90", "pressures_per90",
        "total_passes_per90", "passes_completed_per90",
    ],
    "MID": [
        "total_passes_per90", "passes_completed_per90",
        "progressive_passes_per90", "assists_per90",
        "ball_recoveries_per90", "tackles_won_per90",
        "interceptions_per90", "dribbles_completed_per90",
        "pressures_per90",
    ],
    "DEF": [
        "tackles_won_per90", "interceptions_per90",
        "clearances_per90", "blocks_per90",
        "ball_recoveries_per90", "total_passes_per90",
        "passes_completed_per90", "progressive_passes_per90",
        "pressures_per90",
    ],
    "GK": [
        "total_passes_per90", "passes_completed_per90",
    ],
}


class PlayerSimilarity:
    """Find similar players using statistical profiles."""

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store

    def find_similar(
        self,
        player_name: str,
        n_results: int = 10,
        position_group_filter: Optional[str] = None,
        competition_filter: Optional[str] = None,
        method: str = "cosine",
    ) -> pd.DataFrame:
        """Find the N most similar players to a given player.

        Args:
            player_name: Name of the target player
            n_results: Number of similar players to return
            position_group_filter: Filter by position group
            competition_filter: Filter by competition
            method: "cosine" or "euclidean"

        Returns:
            DataFrame with similar players and similarity scores
        """
        db = self.store.player_db.copy()

        if db.empty:
            logger.warning("Player database is empty!")
            return pd.DataFrame()

        # Find target player
        target_mask = db["player"].str.contains(
            player_name, case=False, na=False
        )
        if not target_mask.any():
            logger.warning(f"Player '{player_name}' not found!")
            return pd.DataFrame()

        target = db[target_mask].iloc[0]
        target_group = target["position_group"]

        logger.info(
            f"Finding players similar to {target['player']} "
            f"({target['team']}, {target['position']}, {target_group})"
        )

        # Filter candidates
        candidates = db.copy()

        # Use same position group by default
        group = position_group_filter or target_group
        candidates = candidates[candidates["position_group"] == group]

        if competition_filter:
            candidates = candidates[
                candidates["competition"].str.contains(
                    competition_filter, case=False, na=False
                )
            ]

        # Remove target player (same player-team-competition-season)
        candidates = candidates[
            ~(
                (candidates["player"] == target["player"])
                & (candidates["team"] == target["team"])
                & (candidates["competition"] == target["competition"])
                & (candidates["season"] == target["season"])
            )
        ]

        if candidates.empty:
            logger.warning("No candidates found after filtering!")
            return pd.DataFrame()

        # Get metrics for this position group
        metrics = SIMILARITY_METRICS.get(group, SIMILARITY_METRICS["MID"])
        available_metrics = [m for m in metrics if m in db.columns]

        if not available_metrics:
            logger.warning("No valid metrics found!")
            return pd.DataFrame()

        # Prepare feature matrix
        all_players = pd.concat(
            [target.to_frame().T, candidates], ignore_index=True
        )
        feature_matrix = all_players[available_metrics].fillna(0).values

        # Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)

        # Calculate similarity
        if method == "cosine":
            sim_matrix = cosine_similarity(scaled)
            scores = sim_matrix[0, 1:]  # First row = target player
        else:  # euclidean
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(scaled)
            scores = 1 / (1 + dist_matrix[0, 1:])  # Convert to similarity

        # Add scores to candidates
        candidates = candidates.copy()
        candidates["similarity_score"] = scores
        candidates = candidates.sort_values("similarity_score", ascending=False)

        return candidates.head(n_results)

    def format_results(
        self, target_name: str, results: pd.DataFrame
    ) -> str:
        """Format similarity results as readable string."""
        if results.empty:
            return f"No similar players found for '{target_name}'."

        output = f"Players most similar to {target_name}:\n\n"

        display_cols = [
            "player", "team", "position", "competition",
            "similarity_score", "minutes_played",
            "goals_per90", "xg_per90", "assists_per90",
            "dribbles_completed_per90", "progressive_passes_per90",
        ]
        cols = [c for c in display_cols if c in results.columns]
        display = results[cols].copy()

        # Round
        for col in display.columns:
            if display[col].dtype == "float64":
                display[col] = display[col].round(3)

        output += display.to_string(index=False)
        return output


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = FeatureStore()
    similarity = PlayerSimilarity(store)

    print("=== Players similar to Messi ===\n")
    results = similarity.find_similar("Messi", n_results=10)
    print(similarity.format_results("Messi", results))

    print("\n\n=== Players similar to Sergio Busquets ===\n")
    results = similarity.find_similar("Busquets", n_results=10)
    print(similarity.format_results("Busquets", results))
