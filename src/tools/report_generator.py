"""
Scouting Report Generator
Creates professional PDF scouting reports for individual players.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mplsoccer import PyPizza

from src.data.feature_store import FeatureStore
from src.tools.similarity import PlayerSimilarity

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Metrics by position group for the report
REPORT_METRICS = {
    "FWD": {
        "metrics": [
            "goals_per90", "xg_per90", "assists_per90",
            "progressive_carries_per90", "progressive_passes_per90",
        ],
        "labels": ["Goals/90", "xG/90", "Assists/90", "Prog Carries/90", "Prog Passes/90"],
    },
    "MID": {
        "metrics": [
            "goals_per90", "assists_per90",
            "progressive_passes_per90", "progressive_carries_per90",
        ],
        "labels": ["Goals/90", "Assists/90", "Prog Passes/90", "Prog Carries/90"],
    },
    "DEF": {
        "metrics": [
            "progressive_passes_per90", "progressive_carries_per90",
            "goals_per90", "assists_per90",
        ],
        "labels": ["Prog Passes/90", "Prog Carries/90", "Goals/90", "Assists/90"],
    },
    "GK": {
        "metrics": ["goals_per90", "assists_per90"],
        "labels": ["Goals/90", "Assists/90"],
    },
}


class ScoutingReportGenerator:
    """Generate professional PDF scouting reports."""

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
        self.similarity = PlayerSimilarity(feature_store)

    def generate_report(
        self,
        player_name: str,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a full scouting report PDF.

        Args:
            player_name: Name of the player
            save_path: Custom path (auto if None)

        Returns:
            Path to saved PDF
        """
        import unicodedata

        db = self.store.player_db

        # Find player (accent-proof)
        def remove_accents(text):
            if pd.isna(text):
                return ""
            nfkd = unicodedata.normalize("NFKD", str(text))
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        mask = db["player"].str.contains(player_name, case=False, na=False)
        if not mask.any():
            clean = remove_accents(player_name).lower()
            mask = db["player"].apply(lambda x: clean in remove_accents(x).lower())

        if not mask.any():
            logger.warning(f"Player '{player_name}' not found!")
            return None

        # Prefer current season
        matches = db[mask]
        current = matches[matches.get("season", "") == "2024/2025"]
        player = current.iloc[0] if not current.empty else matches.iloc[0]

        pos_group = player.get("position_group", "MID")

        # Find similar players
        similar = self.similarity.find_similar(player_name, n_results=5)

        # Generate PDF
        if save_path is None:
            safe_name = player["player"].replace(" ", "_")[:30]
            save_path = str(REPORTS_DIR / f"report_{safe_name}.pdf")

        with PdfPages(save_path) as pdf:
            # Page 1: Overview + Stats
            self._page_overview(pdf, player, pos_group, db)

            # Page 2: Radar chart
            self._page_radar(pdf, player, pos_group, db)

            # Page 3: Similar players
            self._page_similar(pdf, player, similar)

        logger.info(f"Report saved: {save_path}")
        return save_path

    def _page_overview(self, pdf, player, pos_group, db):
        """Page 1: Player overview with key stats."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0f172a")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 13)
        ax.axis("off")

        # Header bar
        ax.fill_between([0, 10], [12.5, 12.5], [13, 13], color="#00f5a0", alpha=0.15)

        # Title
        ax.text(0.5, 12.7, "SCOUTING REPORT", fontsize=10, color="#00f5a0",
                fontweight="bold", ha="left", va="center", family="monospace")
        ax.text(9.5, 12.7, datetime.now().strftime("%d %b %Y"),
                fontsize=8, color="#64748b", ha="right", va="center")

        # Player name
        ax.text(0.5, 11.8, player["player"], fontsize=22, color="#e2e8f0",
                fontweight="bold", ha="left")

        # Info line
        team = player.get("team", "Unknown")
        pos = player.get("position", "Unknown")
        comp = player.get("competition", "Unknown")
        season = player.get("season", "")
        ax.text(0.5, 11.2, f"{team}  ·  {pos}  ·  {comp} {season}",
                fontsize=10, color="#64748b", ha="left")

        # Divider
        ax.plot([0.5, 9.5], [10.8, 10.8], color="#1e293b", linewidth=1)

        # Key stats boxes
        stats_data = [
            ("Minutes", int(player.get("minutes_played", 0))),
            ("Matches", int(player.get("matches_played", 0))),
            ("Goals", int(player.get("goals", 0))),
            ("Assists", int(player.get("assists", 0))),
        ]

        for i, (label, value) in enumerate(stats_data):
            x = 0.5 + i * 2.3
            # Box
            rect = plt.Rectangle((x, 9.5), 2, 1.1, linewidth=1,
                                  edgecolor="#1e293b", facecolor="#111827", 
                                  clip_on=False)
            ax.add_patch(rect)
            ax.text(x + 1, 10.2, str(value), fontsize=18, color="#00f5a0",
                    fontweight="bold", ha="center", va="center")
            ax.text(x + 1, 9.7, label, fontsize=8, color="#64748b",
                    ha="center", va="center")

        # Per-90 stats section
        ax.text(0.5, 8.8, "PER 90 STATISTICS", fontsize=10, color="#00f5a0",
                fontweight="bold", family="monospace")
        ax.plot([0.5, 9.5], [8.5, 8.5], color="#1e293b", linewidth=0.5)

        template = REPORT_METRICS.get(pos_group, REPORT_METRICS["MID"])
        y = 8.0
        for metric, label in zip(template["metrics"], template["labels"]):
            if metric in player.index:
                val = player[metric]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    # Get percentile
                    group_data = db[db["position_group"] == pos_group]
                    pctl = (group_data[metric].dropna() < val).sum() / max(len(group_data[metric].dropna()), 1) * 100

                    ax.text(0.7, y, label, fontsize=10, color="#cbd5e1", ha="left")
                    ax.text(5.5, y, f"{val:.3f}", fontsize=10, color="#e2e8f0",
                            ha="right", fontweight="bold")

                    # Percentile bar
                    bar_x = 6.0
                    bar_w = 3.0
                    ax.fill_between([bar_x, bar_x + bar_w], [y - 0.12] * 2,
                                    [y + 0.12] * 2, color="#1e293b")
                    fill_w = bar_w * (pctl / 100)
                    bar_color = "#22c55e" if pctl >= 80 else "#3b82f6" if pctl >= 60 else "#f59e0b" if pctl >= 40 else "#ef4444"
                    ax.fill_between([bar_x, bar_x + fill_w], [y - 0.12] * 2,
                                    [y + 0.12] * 2, color=bar_color, alpha=0.7)
                    ax.text(bar_x + bar_w + 0.2, y, f"{pctl:.0f}%", fontsize=8,
                            color=bar_color, ha="left", va="center")

                    y -= 0.6

        # Footer
        ax.text(5, 0.3, "Data: StatsBomb · FBref  |  AI Football Scouting Agent",
                fontsize=7, color="#334155", ha="center")

        fig.tight_layout()
        pdf.savefig(fig, facecolor="#0f172a")
        plt.close(fig)

    def _page_radar(self, pdf, player, pos_group, db):
        """Page 2: Radar/Pizza chart."""
        template = REPORT_METRICS.get(pos_group, REPORT_METRICS["MID"])
        metrics = template["metrics"]
        labels = template["labels"]

        # Calculate percentiles
        group_data = db[db["position_group"] == pos_group]
        values = []
        valid_labels = []

        for metric, label in zip(metrics, labels):
            if metric not in db.columns:
                continue
            player_val = player.get(metric, 0)
            if pd.isna(player_val):
                player_val = 0
            group_vals = group_data[metric].dropna()
            if len(group_vals) == 0:
                continue
            pctl = (group_vals < player_val).sum() / len(group_vals) * 100
            values.append(round(pctl, 1))
            valid_labels.append(label)

        if not values or len(values) < 3:
            return

        # Colors based on percentile
        slice_colors = []
        for v in values:
            if v >= 80:
                slice_colors.append("#22c55e")
            elif v >= 60:
                slice_colors.append("#3b82f6")
            elif v >= 40:
                slice_colors.append("#f59e0b")
            else:
                slice_colors.append("#ef4444")

        fig, ax = plt.subplots(figsize=(8.5, 11), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("#0f172a")

        baker = PyPizza(
            params=valid_labels,
            background_color="#0f172a",
            straight_line_color="#1e293b",
            straight_line_lw=1,
            last_circle_color="#1e293b",
            last_circle_lw=1,
            other_circle_lw=0.5,
            other_circle_color="#1e293b",
            inner_circle_size=20,
        )

        baker.make_pizza(
            values,
            figsize=(8.5, 11),
            color_blank_space=["#0f172a"] * len(values),
            slice_colors=slice_colors,
            value_colors=["#000000"] * len(values),
            value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#1e293b", zorder=2, linewidth=1),
            kwargs_params=dict(color="#94a3b8", fontsize=11, va="center"),
            kwargs_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(edgecolor="#000000", facecolor="cornflowerblue",
                          boxstyle="round,pad=0.2", lw=1),
            ),
        )

        fig = plt.gcf()
        fig.text(0.5, 0.96, f"{player['player']} — Percentile Rankings",
                 ha="center", fontsize=16, fontweight="bold", color="#e2e8f0")
        fig.text(0.5, 0.93,
                 f"vs {pos_group} peers | {player.get('competition', '')} {player.get('season', '')}",
                 ha="center", fontsize=10, color="#64748b")

        # Legend
        fig.text(0.15, 0.04, "■ 80+ Elite", fontsize=9, color="#22c55e")
        fig.text(0.35, 0.04, "■ 60+ Good", fontsize=9, color="#3b82f6")
        fig.text(0.55, 0.04, "■ 40+ Average", fontsize=9, color="#f59e0b")
        fig.text(0.75, 0.04, "■ <40 Below avg", fontsize=9, color="#ef4444")

        pdf.savefig(fig, facecolor="#0f172a")
        plt.close(fig)

    def _page_similar(self, pdf, player, similar):
        """Page 3: Similar players."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0f172a")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 13)
        ax.axis("off")

        # Header
        ax.fill_between([0, 10], [12.5, 12.5], [13, 13], color="#00f5a0", alpha=0.15)
        ax.text(0.5, 12.7, "SIMILAR PLAYERS", fontsize=10, color="#00f5a0",
                fontweight="bold", family="monospace")

        ax.text(0.5, 11.8, f"Players with similar profiles to {player['player']}",
                fontsize=12, color="#94a3b8")

        ax.plot([0.5, 9.5], [11.3, 11.3], color="#1e293b", linewidth=1)

        if similar.empty:
            ax.text(5, 8, "No similar players found in database.",
                    fontsize=12, color="#64748b", ha="center")
        else:
            # Table header
            y = 10.8
            ax.text(0.7, y, "Player", fontsize=9, color="#64748b", fontweight="bold")
            ax.text(4.5, y, "Team", fontsize=9, color="#64748b", fontweight="bold")
            ax.text(7.0, y, "Competition", fontsize=9, color="#64748b", fontweight="bold")
            ax.text(9.3, y, "Sim %", fontsize=9, color="#64748b",
                    fontweight="bold", ha="right")

            y -= 0.15
            ax.plot([0.5, 9.5], [y, y], color="#1e293b", linewidth=0.5)
            y -= 0.35

            for i, (_, row) in enumerate(similar.head(10).iterrows()):
                if y < 1:
                    break

                name = str(row.get("player", "Unknown"))
                if len(name) > 30:
                    name = name[:28] + "..."
                team = str(row.get("team", ""))[:20]
                comp = str(row.get("competition", ""))[:15]
                sim = row.get("similarity_score", 0) * 100

                bg_alpha = 0.04 if i % 2 == 0 else 0
                if bg_alpha > 0:
                    ax.fill_between([0.5, 9.5], [y - 0.2] * 2, [y + 0.2] * 2,
                                    color="white", alpha=bg_alpha)

                ax.text(0.7, y, name, fontsize=9, color="#e2e8f0")
                ax.text(4.5, y, team, fontsize=9, color="#94a3b8")
                ax.text(7.0, y, comp, fontsize=9, color="#94a3b8")

                sim_color = "#22c55e" if sim >= 90 else "#3b82f6" if sim >= 80 else "#f59e0b"
                ax.text(9.3, y, f"{sim:.1f}%", fontsize=9, color=sim_color,
                        fontweight="bold", ha="right")

                y -= 0.5

        # Footer
        ax.text(5, 0.3, "Data: StatsBomb · FBref  |  AI Football Scouting Agent",
                fontsize=7, color="#334155", ha="center")

        fig.tight_layout()
        pdf.savefig(fig, facecolor="#0f172a")
        plt.close(fig)

    def format_result(self, path: str) -> str:
        """Format result string for the agent."""
        if path:
            return f"Scouting report saved to: {path}"
        return "Could not generate report."


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = FeatureStore()
    gen = ScoutingReportGenerator(store)

    print("Generating report for Salah...")
    path = gen.generate_report("Salah")
    if path:
        print(f"Saved: {path}")

    print("\nGenerating report for Messi...")
    path = gen.generate_report("Messi")
    if path:
        print(f"Saved: {path}")