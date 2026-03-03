"""
Data Preprocessor
Converts raw StatsBomb events into player-level per-90 statistics.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Position group mappings
POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Right Back": "DEF", "Left Back": "DEF",
    "Right Wing Back": "DEF", "Left Wing Back": "DEF",
    "Center Back": "DEF", "Right Center Back": "DEF",
    "Left Center Back": "DEF",
    "Right Defensive Midfield": "MID", "Left Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Right Midfield": "MID", "Left Midfield": "MID",
    "Right Center Midfield": "MID", "Left Center Midfield": "MID",
    "Center Midfield": "MID",
    "Right Attacking Midfield": "MID", "Left Attacking Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Right Wing": "FWD", "Left Wing": "FWD",
    "Right Center Forward": "FWD", "Left Center Forward": "FWD",
    "Center Forward": "FWD", "Striker": "FWD",
    "Secondary Striker": "FWD",
}

MIN_MINUTES = 450


class Preprocessor:
    """Converts raw StatsBomb event data to player statistics."""

    def __init__(self, min_minutes: int = MIN_MINUTES):
        self.min_minutes = min_minutes

    def extract_player_stats(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract per-player aggregated stats from event data."""
        logger.info("Extracting player stats from events...")

        minutes = self._calculate_minutes(events)
        shots = self._extract_shooting(events)
        passes = self._extract_passing(events)
        dribbles = self._extract_dribbling(events)
        defense = self._extract_defensive(events)
        carrying = self._extract_carrying(events)
        pressures = self._extract_pressures(events)

        # Merge all
        stats = minutes
        for df in [shots, passes, dribbles, defense, carrying, pressures]:
            if not df.empty:
                stats = stats.merge(df, on="player", how="left")

        stats = stats.fillna(0)
        stats = self._add_player_metadata(stats, events)

        logger.info(f"Extracted stats for {len(stats)} players")
        return stats

    def normalize_per90(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Normalize counting stats to per-90-minute rates."""
        df = stats.copy()
        df = df[df["minutes_played"] >= self.min_minutes].copy()
        logger.info(
            f"After min_minutes filter ({self.min_minutes}): "
            f"{len(df)} players remain"
        )

        # Columns NOT to normalize
        skip_cols = {
            "player", "team", "position", "position_group",
            "minutes_played", "matches_played", "competition", "season",
            "pass_completion_pct", "shot_conversion_rate",
            "dribble_success_pct",
        }

        count_cols = [col for col in df.columns if col not in skip_cols]

        for col in count_cols:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                df[f"{col}_per90"] = (df[col] / df["minutes_played"]) * 90

        return df

    def calculate_percentiles(
        self, stats: pd.DataFrame, group_by: str = "position_group"
    ) -> pd.DataFrame:
        """Calculate percentile rankings within position groups."""
        df = stats.copy()
        per90_cols = [col for col in df.columns if col.endswith("_per90")]

        for col in per90_cols:
            pctl_col = col.replace("_per90", "_percentile")
            df[pctl_col] = df.groupby(group_by)[col].rank(pct=True) * 100

        return df

    # ── Private extraction methods ──────────────────────

    def _calculate_minutes(self, events: pd.DataFrame) -> pd.DataFrame:
        """Estimate minutes played per player from event timestamps."""
        player_match = (
            events.dropna(subset=["player"])
            .groupby(["player", "match_id"])["minute"]
            .max()
            .reset_index()
        )
        minutes = (
            player_match.groupby("player")
            .agg(
                minutes_played=("minute", "sum"),
                matches_played=("match_id", "nunique"),
            )
            .reset_index()
        )
        return minutes

    def _extract_shooting(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract shooting stats."""
        shots = events[events["type"] == "Shot"].copy()
        if shots.empty:
            return pd.DataFrame(columns=["player"])

        stats = shots.groupby("player").agg(
            total_shots=("type", "count"),
            goals=("shot_outcome", lambda x: (x == "Goal").sum()),
            shots_on_target=(
                "shot_outcome",
                lambda x: x.isin(["Goal", "Saved"]).sum(),
            ),
            xg=("shot_statsbomb_xg", "sum"),
        ).reset_index()

        stats["shot_conversion_rate"] = np.where(
            stats["total_shots"] > 0,
            stats["goals"] / stats["total_shots"],
            0,
        )
        stats["xg_difference"] = stats["goals"] - stats["xg"]

        return stats

    def _extract_passing(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract passing stats."""
        passes = events[events["type"] == "Pass"].copy()
        if passes.empty:
            return pd.DataFrame(columns=["player"])

         # Base passing stats
        agg_dict = {
            "total_passes": ("type", "count"),
            "passes_completed": (
                "pass_outcome",
                lambda x: x.isna().sum(),
            ),
        }

        # Only add columns that exist in the data
        if "pass_goal_assist" in passes.columns:
            agg_dict["assists"] = (
                "pass_goal_assist",
                lambda x: (x == True).sum(),
            )
        if "pass_cross" in passes.columns:
            agg_dict["crosses"] = (
                "pass_cross",
                lambda x: (x == True).sum(),
            )
        if "pass_shot_assist" in passes.columns:
            agg_dict["shot_assists"] = (
                "pass_shot_assist",
                lambda x: (x == True).sum(),
            )

        stats = passes.groupby("player").agg(**agg_dict).reset_index()

        # Fill missing columns with 0
        for col in ["assists", "crosses", "shot_assists"]:
            if col not in stats.columns:
                stats[col] = 0

        stats["pass_completion_pct"] = np.where(
            stats["total_passes"] > 0,
            stats["passes_completed"] / stats["total_passes"] * 100,
            0,
        )

        # Progressive passes
        if "pass_end_location" in passes.columns:
            passes = passes.copy()
            passes["progressive"] = passes.apply(
                self._is_progressive_pass, axis=1
            )
            prog = (
                passes[passes["progressive"]]
                .groupby("player")
                .size()
                .reset_index(name="progressive_passes")
            )
            stats = stats.merge(prog, on="player", how="left")
            stats["progressive_passes"] = stats["progressive_passes"].fillna(0)

        return stats

    def _extract_dribbling(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract dribbling stats."""
        dribbles = events[events["type"] == "Dribble"].copy()
        if dribbles.empty:
            return pd.DataFrame(columns=["player"])

        stats = dribbles.groupby("player").agg(
            dribbles_attempted=("type", "count"),
            dribbles_completed=(
                "dribble_outcome",
                lambda x: (x == "Complete").sum(),
            ),
        ).reset_index()

        stats["dribble_success_pct"] = np.where(
            stats["dribbles_attempted"] > 0,
            stats["dribbles_completed"] / stats["dribbles_attempted"] * 100,
            0,
        )
        return stats

    def _extract_defensive(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract defensive stats."""
        result = pd.DataFrame()

        # Tackles (from Duel events)
        duels = events[events["type"] == "Duel"]
        if not duels.empty:
            result = duels.groupby("player").agg(
                tackles_attempted=("type", "count"),
                tackles_won=(
                    "duel_outcome",
                    lambda x: x.isin(["Won", "Success", "Success In Play",
                                       "Success Out"]).sum(),
                ),
            ).reset_index()

        # Interceptions
        intercepts = events[events["type"] == "Interception"]
        if not intercepts.empty:
            int_df = (
                intercepts.groupby("player")
                .size()
                .reset_index(name="interceptions")
            )
            if result.empty:
                result = int_df
            else:
                result = result.merge(int_df, on="player", how="outer")

        # Clearances
        clearances = events[events["type"] == "Clearance"]
        if not clearances.empty:
            clr_df = (
                clearances.groupby("player")
                .size()
                .reset_index(name="clearances")
            )
            if result.empty:
                result = clr_df
            else:
                result = result.merge(clr_df, on="player", how="outer")

        # Ball recoveries
        recoveries = events[events["type"] == "Ball Recovery"]
        if not recoveries.empty:
            rec_df = (
                recoveries.groupby("player")
                .size()
                .reset_index(name="ball_recoveries")
            )
            if result.empty:
                result = rec_df
            else:
                result = result.merge(rec_df, on="player", how="outer")

        # Blocks
        blocks = events[events["type"] == "Block"]
        if not blocks.empty:
            blk_df = (
                blocks.groupby("player")
                .size()
                .reset_index(name="blocks")
            )
            if result.empty:
                result = blk_df
            else:
                result = result.merge(blk_df, on="player", how="outer")

        if result.empty:
            return pd.DataFrame(columns=["player"])

        return result.fillna(0)

    def _extract_carrying(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract ball carrying stats."""
        carries = events[events["type"] == "Carry"].copy()
        if carries.empty:
            return pd.DataFrame(columns=["player"])

        stats = (
            carries.groupby("player")
            .size()
            .reset_index(name="total_carries")
        )

        # Progressive carries
        if "carry_end_location" in carries.columns:
            carries = carries.copy()
            carries["progressive"] = carries.apply(
                self._is_progressive_carry, axis=1
            )
            prog = (
                carries[carries["progressive"]]
                .groupby("player")
                .size()
                .reset_index(name="progressive_carries")
            )
            stats = stats.merge(prog, on="player", how="left")
            stats["progressive_carries"] = stats["progressive_carries"].fillna(0)

        return stats

    def _extract_pressures(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract pressing stats."""
        pressures = events[events["type"] == "Pressure"]
        if pressures.empty:
            return pd.DataFrame(columns=["player"])

        return (
            pressures.groupby("player")
            .size()
            .reset_index(name="pressures")
        )

    def _add_player_metadata(
        self, stats: pd.DataFrame, events: pd.DataFrame
    ) -> pd.DataFrame:
        """Add team, position info to player stats."""
        player_events = events.dropna(subset=["player"])

        meta = (
            player_events.groupby("player")
            .agg(
                team=("team", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
                position=(
                    "position",
                    lambda x: x.dropna().mode().iloc[0]
                    if len(x.dropna()) > 0 else "Unknown",
                ),
            )
            .reset_index()
        )

        stats = stats.merge(meta, on="player", how="left")
        stats["position_group"] = stats["position"].map(POSITION_GROUPS)
        stats["position_group"] = stats["position_group"].fillna("Unknown")

        return stats

    @staticmethod
    def _is_progressive_pass(row) -> bool:
        """Check if a pass is progressive (>=10m towards goal)."""
        try:
            start = row["location"]
            end = row["pass_end_location"]
            if start is None or end is None:
                return False
            start_dist = 120 - start[0]
            end_dist = 120 - end[0]
            return (start_dist - end_dist) >= 10
        except (TypeError, IndexError, KeyError):
            return False

    @staticmethod
    def _is_progressive_carry(row) -> bool:
        """Check if a carry is progressive (>=10m towards goal)."""
        try:
            start = row["location"]
            end = row["carry_end_location"]
            if start is None or end is None:
                return False
            start_dist = 120 - start[0]
            end_dist = 120 - end[0]
            return (start_dist - end_dist) >= 10
        except (TypeError, IndexError, KeyError):
            return False


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import StatsBombLoader

    loader = StatsBombLoader()
    preprocessor = Preprocessor(min_minutes=0)  # 0 for single match test

    # Test with one match
    matches = loader.get_matches(competition_id=11, season_id=90)
    match_id = matches.iloc[0]["match_id"]
    print(f"Testing with match: {match_id}")

    events = loader.get_events(match_id=match_id)
    print(f"Events: {len(events)}")

    stats = preprocessor.extract_player_stats(events)
    print(f"\nPlayer stats: {len(stats)} players")
    print(stats[["player", "team", "position", "minutes_played",
                  "goals", "total_passes", "dribbles_completed"]].head(10))
