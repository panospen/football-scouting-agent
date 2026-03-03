"""
Visualization Tool
Generates radar/pizza charts for player profiles using mplsoccer.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import PyPizza

from src.data.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Output directory for charts
CHARTS_DIR = Path("data/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Radar templates by position group
RADAR_TEMPLATES = {
    "FWD": {
        "metrics": [
            "goals_per90", "xg_per90", "total_shots_per90",
            "assists_per90", "dribbles_completed_per90",
            "progressive_carries_per90", "pressures_per90",
            "total_passes_per90", "passes_completed_per90",
        ],
        "labels": [
            "Goals", "xG", "Shots",
            "Assists", "Dribbles", "Prog.\nCarries",
            "Pressures", "Passes", "Pass\nCompletion",
        ],
    },
    "MID": {
        "metrics": [
            "goals_per90", "assists_per90",
            "total_passes_per90", "progressive_passes_per90",
            "passes_completed_per90", "dribbles_completed_per90",
            "tackles_won_per90", "interceptions_per90",
            "ball_recoveries_per90", "pressures_per90",
        ],
        "labels": [
            "Goals", "Assists",
            "Passes", "Prog.\nPasses", "Pass\nCompletion",
            "Dribbles", "Tackles\nWon", "Interceptions",
            "Ball\nRecoveries", "Pressures",
        ],
    },
    "DEF": {
        "metrics": [
            "tackles_won_per90", "interceptions_per90",
            "clearances_per90", "blocks_per90",
            "ball_recoveries_per90", "pressures_per90",
            "total_passes_per90", "passes_completed_per90",
            "progressive_passes_per90",
        ],
        "labels": [
            "Tackles\nWon", "Interceptions",
            "Clearances", "Blocks", "Ball\nRecoveries",
            "Pressures", "Passes", "Pass\nCompletion",
            "Prog.\nPasses",
        ],
    },
    "GK": {
        "metrics": [
            "total_passes_per90", "passes_completed_per90",
        ],
        "labels": [
            "Passes", "Pass\nCompletion",
        ],
    },
}


class PlayerVisualization:
    """Generate visual player profiles."""

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store

    def generate_pizza(
        self,
        player_name: str,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 8),
    ) -> Optional[str]:
        """Generate a pizza chart showing player percentile rankings.

        Args:
            player_name: Name of the player
            save_path: Custom save path (auto-generated if None)
            figsize: Figure size

        Returns:
            Path to saved chart image, or None if failed
        """
        db = self.store.player_db

        # Find player
        mask = db["player"].str.contains(player_name, case=False, na=False)
        if not mask.any():
            logger.warning(f"Player '{player_name}' not found!")
            return None

        player = db[mask].iloc[0]
        pos_group = player["position_group"]

        template = RADAR_TEMPLATES.get(pos_group)
        if not template:
            logger.warning(f"No template for position group: {pos_group}")
            return None

        # Get percentile values
        metrics = template["metrics"]
        labels = template["labels"]

        # Calculate percentiles within position group
        group_data = db[db["position_group"] == pos_group]
        values = []
        valid_labels = []
        valid_metrics = []

        for metric, label in zip(metrics, labels):
            if metric not in db.columns:
                continue

            player_val = player[metric]
            group_vals = group_data[metric].dropna()

            if len(group_vals) == 0:
                continue

            percentile = (group_vals < player_val).sum() / len(group_vals) * 100
            values.append(round(percentile, 1))
            valid_labels.append(label)
            valid_metrics.append(metric)

        if not values:
            logger.warning("No valid metrics to plot!")
            return None

        # Color slices based on percentile value
        slice_colors = []
        text_colors = []
        for v in values:
            if v >= 80:
                slice_colors.append("#2ecc71")  # Green — excellent
                text_colors.append("#000000")
            elif v >= 60:
                slice_colors.append("#3498db")  # Blue — good
                text_colors.append("#000000")
            elif v >= 40:
                slice_colors.append("#f39c12")  # Orange — average
                text_colors.append("#000000")
            else:
                slice_colors.append("#e74c3c")  # Red — below average
                text_colors.append("#000000")

        # Create pizza chart
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("#1a1a2e")

        baker = PyPizza(
            params=valid_labels,
            background_color="#1a1a2e",
            straight_line_color="#e8e8e8",
            straight_line_lw=1,
            last_circle_color="#e8e8e8",
            last_circle_lw=1,
            other_circle_lw=0.5,
            other_circle_color="#cccccc",
            inner_circle_size=20,
        )

        baker.make_pizza(
            values,
            figsize=figsize,
            color_blank_space=["#1a1a2e"] * len(values),
            slice_colors=slice_colors,
            value_colors=text_colors,
            value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#e8e8e8", zorder=2, linewidth=1),
            kwargs_params=dict(color="#e8e8e8", fontsize=11, va="center"),
            kwargs_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="cornflowerblue",
                    boxstyle="round,pad=0.2", lw=1,
                ),
            ),
        )

        # Title
        fig = plt.gcf()
        fig.text(
            0.5, 0.97,
            f"{player['player']}",
            ha="center", va="top",
            fontsize=18, fontweight="bold", color="#e8e8e8",
        )
        fig.text(
            0.5, 0.93,
            f"{player['team']} | {player['position']} | {player['competition']} {player.get('season', '')}",
            ha="center", va="top",
            fontsize=12, color="#b0b0b0",
        )
        fig.text(
            0.5, 0.02,
            "Percentile rank vs position peers | Data: StatsBomb Open Data",
            ha="center", va="bottom",
            fontsize=9, color="#808080",
        )

        # Legend
        fig.text(0.82, 0.05, "■ 80+ Elite", fontsize=9, color="#2ecc71")
        fig.text(0.82, 0.03, "■ 60+ Good", fontsize=9, color="#3498db")
        fig.text(0.64, 0.05, "■ 40+ Average", fontsize=9, color="#f39c12")
        fig.text(0.64, 0.03, "■ <40 Below avg", fontsize=9, color="#e74c3c")

        # Save
        if save_path is None:
            safe_name = player["player"].replace(" ", "_")[:30]
            save_path = str(
                CHARTS_DIR / f"pizza_{safe_name}_{player['competition']}.png"
            )

        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

        logger.info(f"Pizza chart saved: {save_path}")
        return save_path

    def generate_comparison(
        self,
        player_names: list,
        save_path: Optional[str] = None,
        figsize: tuple = (16, 8),
    ) -> Optional[str]:
        """Generate side-by-side pizza charts for player comparison.

        Args:
            player_names: List of 2 player names to compare
            save_path: Custom save path

        Returns:
            Path to saved chart image
        """
        if len(player_names) != 2:
            logger.warning("Comparison requires exactly 2 players!")
            return None

        paths = []
        for name in player_names:
            path = self.generate_pizza(name)
            if path:
                paths.append(path)

        if len(paths) == 2:
            logger.info(f"Comparison charts saved: {paths}")

        return paths


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = FeatureStore()
    viz = PlayerVisualization(store)

    print("Generating pizza chart for Messi...")
    path = viz.generate_pizza("Messi")
    if path:
        print(f"Saved to: {path}")

    print("\nGenerating pizza chart for Busquets...")
    path = viz.generate_pizza("Busquets")
    if path:
        print(f"Saved to: {path}")

    print("\nGenerating pizza chart for Neymar...")
    path = viz.generate_pizza("Neymar")
    if path:
        print(f"Saved to: {path}")

    print("\nDone! Check the data/charts/ folder.")
