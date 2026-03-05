"""
Pitch Visualizations — Shot Maps & Heatmaps
Uses StatsBomb event-level location data for spatial analysis.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("data/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Custom dark colormap
DARK_CMAP = LinearSegmentedColormap.from_list(
    "dark_heat",
    ["#0f172a", "#1e3a5f", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444"],
)


class PitchVisualizer:
    """Generate shot maps and heatmaps from StatsBomb event data."""

    def __init__(self):
        pass

    def generate_shot_map(
        self,
        events_df: pd.DataFrame,
        player_name: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a shot map for a player.

        Args:
            events_df: StatsBomb events DataFrame with columns:
                       location_x, location_y, shot_outcome, shot_statsbomb_xg
            player_name: For title/filename
            title: Custom title

        Returns:
            Path to saved image
        """
        shots = events_df[events_df["type"] == "Shot"].copy()

        if shots.empty:
            logger.warning(f"No shot data found for {player_name}")
            return None

        # Extract locations
        if "location" in shots.columns:
            shots["loc_x"] = shots["location"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
            shots["loc_y"] = shots["location"].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
        elif "location_x" in shots.columns:
            shots["loc_x"] = shots["location_x"]
            shots["loc_y"] = shots["location_y"]
        else:
            logger.warning("No location columns found in events")
            return None

        shots = shots.dropna(subset=["loc_x", "loc_y"])

        if shots.empty:
            return None

        # Get xG values
        xg_col = None
        for col in ["shot_statsbomb_xg", "xg", "shot_xg"]:
            if col in shots.columns:
                xg_col = col
                break

        # Create figure
        pitch = VerticalPitch(
            pitch_type="statsbomb",
            half=True,
            pitch_color="#0f172a",
            line_color="#1e293b",
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=(10, 8))
        fig.patch.set_facecolor("#0f172a")

        # Separate goals vs non-goals
        if "shot_outcome" in shots.columns:
            goals = shots[shots["shot_outcome"] == "Goal"]
            misses = shots[shots["shot_outcome"] != "Goal"]
        else:
            goals = pd.DataFrame()
            misses = shots

        # Plot misses
        if not misses.empty:
            sizes = misses[xg_col].fillna(0.05) * 800 + 30 if xg_col else [60] * len(misses)
            pitch.scatter(
                misses["loc_x"], misses["loc_y"],
                s=sizes,
                c="#64748b",
                alpha=0.5,
                edgecolors="#94a3b8",
                linewidth=0.5,
                zorder=2,
                ax=ax,
            )

        # Plot goals
        if not goals.empty:
            sizes = goals[xg_col].fillna(0.05) * 800 + 30 if xg_col else [80] * len(goals)
            pitch.scatter(
                goals["loc_x"], goals["loc_y"],
                s=sizes,
                c="#00f5a0",
                alpha=0.8,
                edgecolors="#ffffff",
                linewidth=1.5,
                zorder=3,
                marker="*",
                ax=ax,
            )

        # Title
        display_title = title or f"Shot Map — {player_name}"
        fig.text(0.5, 0.95, display_title, ha="center", fontsize=16,
                 fontweight="bold", color="#e2e8f0")

        # Stats summary
        total = len(shots)
        goal_count = len(goals) if not goals.empty else 0
        avg_xg = shots[xg_col].mean() if xg_col and not shots[xg_col].isna().all() else 0

        stats_text = f"{total} Shots  ·  {goal_count} Goals  ·  Avg xG: {avg_xg:.3f}"
        fig.text(0.5, 0.91, stats_text, ha="center", fontsize=10, color="#64748b")

        # Legend
        fig.text(0.15, 0.04, "★ Goal", fontsize=10, color="#00f5a0")
        fig.text(0.35, 0.04, "● Miss/Saved/Blocked", fontsize=10, color="#64748b")
        fig.text(0.65, 0.04, "Size = xG value", fontsize=10, color="#475569")

        # Save
        safe_name = player_name.replace(" ", "_")[:25]
        path = str(CHARTS_DIR / f"shotmap_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)

        logger.info(f"Shot map saved: {path}")
        return path

    def generate_heatmap(
        self,
        events_df: pd.DataFrame,
        player_name: str,
        event_types: Optional[list] = None,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a heatmap of player activity on pitch.

        Args:
            events_df: StatsBomb events DataFrame
            player_name: For title/filename
            event_types: Filter to specific types (e.g., ["Pass", "Carry"])
            title: Custom title

        Returns:
            Path to saved image
        """
        df = events_df.copy()

        # Filter event types
        if event_types:
            df = df[df["type"].isin(event_types)]

        if df.empty:
            logger.warning(f"No events found for heatmap")
            return None

        # Extract locations
        if "location" in df.columns:
            df["loc_x"] = df["location"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
            df["loc_y"] = df["location"].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
        elif "location_x" in df.columns:
            df["loc_x"] = df["location_x"]
            df["loc_y"] = df["location_y"]
        else:
            logger.warning("No location columns found")
            return None

        df = df.dropna(subset=["loc_x", "loc_y"])

        if df.empty:
            return None

        # Create figure
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#0f172a",
            line_color="#1e293b",
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.patch.set_facecolor("#0f172a")

        # KDE heatmap
        try:
            pitch.kdeplot(
                df["loc_x"], df["loc_y"],
                ax=ax,
                cmap=DARK_CMAP,
                fill=True,
                levels=100,
                thresh=0.05,
                alpha=0.7,
            )
        except Exception:
            # Fallback: hexbin
            pitch.hexbin(
                df["loc_x"], df["loc_y"],
                ax=ax,
                gridsize=15,
                cmap=DARK_CMAP,
                alpha=0.7,
                edgecolors="none",
            )

        # Title
        event_label = ", ".join(event_types) if event_types else "All Events"
        display_title = title or f"Heatmap — {player_name} ({event_label})"
        fig.text(0.5, 0.96, display_title, ha="center", fontsize=16,
                 fontweight="bold", color="#e2e8f0")

        fig.text(0.5, 0.92, f"{len(df)} events plotted", ha="center",
                 fontsize=10, color="#64748b")

        # Save
        safe_name = player_name.replace(" ", "_")[:25]
        etype = event_types[0].lower() if event_types else "all"
        path = str(CHARTS_DIR / f"heatmap_{safe_name}_{etype}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)

        logger.info(f"Heatmap saved: {path}")
        return path

    def generate_pass_map(
        self,
        events_df: pd.DataFrame,
        player_name: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a pass map showing completed/failed passes.

        Args:
            events_df: StatsBomb events DataFrame
            player_name: For title/filename
            title: Custom title

        Returns:
            Path to saved image
        """
        passes = events_df[events_df["type"] == "Pass"].copy()

        if passes.empty:
            logger.warning(f"No pass data found for {player_name}")
            return None

        # Extract start locations
        if "location" in passes.columns:
            passes["start_x"] = passes["location"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
            passes["start_y"] = passes["location"].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
        elif "location_x" in passes.columns:
            passes["start_x"] = passes["location_x"]
            passes["start_y"] = passes["location_y"]
        else:
            return None

        # Extract end locations
        if "pass_end_location" in passes.columns:
            passes["end_x"] = passes["pass_end_location"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
            passes["end_y"] = passes["pass_end_location"].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else None
            )
        else:
            return None

        passes = passes.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        if passes.empty:
            return None

        # Create figure
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#0f172a",
            line_color="#1e293b",
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.patch.set_facecolor("#0f172a")

        # Completed passes
        if "pass_outcome" in passes.columns:
            completed = passes[passes["pass_outcome"].isna()]  # NaN = completed
            failed = passes[passes["pass_outcome"].notna()]
        else:
            completed = passes
            failed = pd.DataFrame()

        # Draw completed
        if not completed.empty:
            pitch.arrows(
                completed["start_x"], completed["start_y"],
                completed["end_x"], completed["end_y"],
                ax=ax,
                color="#00f5a0",
                alpha=0.4,
                width=1.5,
                headwidth=5,
                headlength=3,
                zorder=2,
            )

        # Draw failed
        if not failed.empty:
            pitch.arrows(
                failed["start_x"], failed["start_y"],
                failed["end_x"], failed["end_y"],
                ax=ax,
                color="#ef4444",
                alpha=0.3,
                width=1,
                headwidth=4,
                headlength=3,
                zorder=2,
            )

        # Title
        display_title = title or f"Pass Map — {player_name}"
        fig.text(0.5, 0.96, display_title, ha="center", fontsize=16,
                 fontweight="bold", color="#e2e8f0")

        comp_count = len(completed) if not completed.empty else 0
        fail_count = len(failed) if not failed.empty else 0
        total = comp_count + fail_count
        pct = (comp_count / total * 100) if total > 0 else 0

        fig.text(0.5, 0.92,
                 f"{comp_count}/{total} passes completed ({pct:.0f}%)",
                 ha="center", fontsize=10, color="#64748b")

        # Legend
        fig.text(0.2, 0.03, "→ Completed", fontsize=10, color="#00f5a0")
        fig.text(0.5, 0.03, "→ Failed", fontsize=10, color="#ef4444")

        # Save
        safe_name = player_name.replace(" ", "_")[:25]
        path = str(CHARTS_DIR / f"passmap_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)

        logger.info(f"Pass map saved: {path}")
        return path


# ── Agent tool wrapper ──────────────────────────────────
def generate_shot_map_tool(player_name: str, events_df: pd.DataFrame) -> str:
    """Tool function for the agent to generate shot maps."""
    viz = PitchVisualizer()
    path = viz.generate_shot_map(events_df, player_name)
    if path:
        return f"Shot map generated and saved to {path}"
    return f"Could not generate shot map for {player_name}. No event data available."


def generate_heatmap_tool(
    player_name: str,
    events_df: pd.DataFrame,
    event_types: Optional[list] = None,
) -> str:
    """Tool function for the agent to generate heatmaps."""
    viz = PitchVisualizer()
    path = viz.generate_heatmap(events_df, player_name, event_types)
    if path:
        return f"Heatmap generated and saved to {path}"
    return f"Could not generate heatmap for {player_name}. No event data available."


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate sample data for testing
    np.random.seed(42)
    n_shots = 50
    sample_shots = pd.DataFrame({
        "type": ["Shot"] * n_shots,
        "location": [(np.random.uniform(80, 120), np.random.uniform(15, 65)) for _ in range(n_shots)],
        "shot_outcome": np.random.choice(["Goal", "Saved", "Off T", "Blocked"], n_shots, p=[0.15, 0.35, 0.3, 0.2]),
        "shot_statsbomb_xg": np.random.uniform(0.02, 0.6, n_shots),
    })

    viz = PitchVisualizer()

    path = viz.generate_shot_map(sample_shots, "Test Player")
    print(f"Shot map: {path}")

    # Sample events for heatmap
    n_events = 200
    sample_events = pd.DataFrame({
        "type": np.random.choice(["Pass", "Carry", "Dribble"], n_events),
        "location": [(np.random.uniform(20, 100), np.random.uniform(10, 70)) for _ in range(n_events)],
    })

    path = viz.generate_heatmap(sample_events, "Test Player", ["Pass", "Carry"])
    print(f"Heatmap: {path}")