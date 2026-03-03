"""
Feature Store
Orchestrates: load → preprocess → normalize → cache.
Provides the clean player database for the agent tools.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.loader import StatsBombLoader, COMPETITIONS
from src.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class FeatureStore:
    """Central player stats database with per-90 and percentile stats."""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        min_minutes: int = 450,
    ):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.loader = StatsBombLoader()
        self.preprocessor = Preprocessor(min_minutes=min_minutes)
        self._player_db: Optional[pd.DataFrame] = None

    @property
    def player_db(self) -> pd.DataFrame:
        """Lazy-load the player database."""
        if self._player_db is None:
            self._player_db = self.load_player_db()
        return self._player_db

    def load_player_db(self) -> pd.DataFrame:
        """Load the preprocessed player database from cache."""
        cache_path = self.processed_dir / "player_database.parquet"
        if cache_path.exists():
            logger.info("Loading player database from cache...")
            return pd.read_parquet(cache_path)

        logger.warning(
            "No cached player database found. Run build_player_db() first."
        )
        return pd.DataFrame()

    def build_player_db(
        self,
        competitions: Optional[list] = None,
        seasons: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Build the full player database from scratch.

        Args:
            competitions: List of competition names to include.
            seasons: Dict mapping competition name to list of seasons.

        Example:
            store.build_player_db(
                competitions=["La Liga"],
                seasons={"La Liga": ["2020/2021"]}
            )
        """
        if competitions is None:
            competitions = list(COMPETITIONS.keys())

        all_stats = []

        for comp_name in competitions:
            comp_info = COMPETITIONS.get(comp_name)
            if comp_info is None:
                logger.warning(f"Unknown competition: {comp_name}")
                continue

            comp_seasons = (
                seasons.get(comp_name, list(comp_info["seasons"].keys()))
                if seasons
                else list(comp_info["seasons"].keys())
            )

            for season_name in comp_seasons:
                logger.info(f"Processing {comp_name} {season_name}...")

                try:
                    events = self.loader.build_match_events_dataset(
                        comp_name, season_name
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch {comp_name} {season_name}: {e}")
                    continue

                if events.empty:
                    continue

                # Extract player stats
                stats = self.preprocessor.extract_player_stats(events)
                stats["competition"] = comp_name
                stats["season"] = season_name
                all_stats.append(stats)

        if not all_stats:
            logger.error("No data was processed!")
            return pd.DataFrame()

        # Combine all
        combined = pd.concat(all_stats, ignore_index=True)
        logger.info(f"Combined stats: {len(combined)} players")

        # Normalize per 90
        normalized = self.preprocessor.normalize_per90(combined)

        # Calculate percentiles
        final = self.preprocessor.calculate_percentiles(normalized)

        # Save
        cache_path = self.processed_dir / "player_database.parquet"
        final.to_parquet(cache_path)
        logger.info(
            f"Player database saved: {len(final)} players -> {cache_path}"
        )

        self._player_db = final
        return final

    def get_player(self, player_name: str) -> Optional[pd.Series]:
        """Get stats for a specific player."""
        db = self.player_db
        if db.empty:
            return None
        matches = db[
            db["player"].str.contains(player_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        return matches.iloc[0]

    def search(self, **filters) -> pd.DataFrame:
        """Search players with filters.

        Example:
            store.search(position_group="FWD", competition="La Liga")
        """
        db = self.player_db.copy()

        for key, value in filters.items():
            if key.startswith("min_"):
                col = key[4:]
                if col in db.columns:
                    db = db[db[col] >= value]
            elif key.startswith("max_"):
                col = key[4:]
                if col in db.columns:
                    db = db[db[col] <= value]
            elif key in db.columns:
                db = db[db[key] == value]

        return db


# ── Build script ────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = FeatureStore()

    # Build for La Liga 2020/2021
    print("Building player database for La Liga 2020/2021...")
    print("This will fetch ~35 matches — may take 5-10 minutes...\n")

    db = store.build_player_db(
        competitions=["La Liga"],
        seasons={"La Liga": ["2020/2021"]},
    )

    print(f"\n{'='*60}")
    print(f"Player database built: {len(db)} players")
    print(f"Columns: {len(db.columns)}")
    print(f"\nPosition groups:")
    print(db["position_group"].value_counts())
    print(f"\nTop 10 scorers (goals_per90):")
    top = db.nlargest(10, "goals_per90")[
        ["player", "team", "position", "minutes_played",
         "goals", "goals_per90", "xg_per90"]
    ]
    print(top.to_string())
