"""
StatsBomb Open Data Loader
Fetches competitions, matches, events, and lineups from StatsBomb's free data.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

logger = logging.getLogger(__name__)

# StatsBomb Open Data — Available competitions
COMPETITIONS = {
    "La Liga": {"competition_id": 11, "seasons": {
        "2020/2021": 90, "2019/2020": 42, "2018/2019": 4,
        "2017/2018": 1, "2016/2017": 27, "2015/2016": 26,
        "2014/2015": 25, "2013/2014": 24, "2012/2013": 23,
        "2011/2012": 22, "2010/2011": 21, "2009/2010": 40,
        "2008/2009": 41, "2007/2008": 39, "2006/2007": 38,
        "2005/2006": 37, "2004/2005": 36,
    }},
    "Premier League": {"competition_id": 2, "seasons": {
        "2003/2004": 44,
    }},
    "Champions League": {"competition_id": 16, "seasons": {
        "2018/2019": 4, "2017/2018": 1, "2016/2017": 27,
        "2015/2016": 26, "2014/2015": 25, "2013/2014": 24,
        "2012/2013": 23, "2011/2012": 22, "2010/2011": 21,
        "2009/2010": 40, "2008/2009": 41, "2006/2007": 38,
        "2004/2005": 36, "2003/2004": 44,
    }},
    "FIFA World Cup": {"competition_id": 43, "seasons": {
        "2022": 106, "2018": 3,
    }},
    "Euro": {"competition_id": 55, "seasons": {
        "2024": 282, "2020": 43,
    }},
    "Bundesliga": {"competition_id": 9, "seasons": {
        "2019/2020": 42, "2018/2019": 4,
        "2017/2018": 1, "2016/2017": 27, "2015/2016": 26,
    }},
}


class StatsBombLoader:
    """Handles all data fetching from StatsBomb Open Data."""

    def __init__(self, raw_dir: str = "data/raw", cache_dir: str = "data/cache"):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_competitions(self) -> pd.DataFrame:
        """Fetch all available free competitions."""
        cache_path = self.cache_dir / "competitions.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        comps = sb.competitions()
        comps.to_parquet(cache_path)
        logger.info(f"Fetched {len(comps)} competition-season entries")
        return comps

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Fetch all matches for a competition-season."""
        cache_path = self.cache_dir / f"matches_{competition_id}_{season_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        matches.to_parquet(cache_path)
        logger.info(
            f"Fetched {len(matches)} matches for comp={competition_id}, "
            f"season={season_id}"
        )
        return matches

    def get_events(self, match_id: int) -> pd.DataFrame:
        """Fetch all events for a single match."""
        cache_path = self.cache_dir / f"events_{match_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        events = sb.events(match_id=match_id)
        events.to_parquet(cache_path)
        return events

    def get_lineups(self, match_id: int) -> dict:
        """Fetch lineups for a match."""
        return sb.lineups(match_id=match_id)

    def get_player_season_stats(
        self, competition_id: int, season_id: int
    ) -> pd.DataFrame:
        """Fetch aggregated player stats for a full season."""
        cache_path = (
            self.cache_dir
            / f"player_season_stats_{competition_id}_{season_id}.parquet"
        )
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        try:
            stats = sb.player_season_stats(
                competition_id=competition_id, season_id=season_id
            )
            stats.to_parquet(cache_path)
            logger.info(f"Fetched player season stats: {len(stats)} players")
            return stats
        except Exception as e:
            logger.warning(
                f"player_season_stats not available: {e}. "
                "Use event aggregation instead."
            )
            return pd.DataFrame()

    def build_match_events_dataset(
        self, competition_name: str, season_name: str
    ) -> pd.DataFrame:
        """Fetch events match-by-match with progress bar.

        This is the main method for building the full dataset.
        """
        comp_info = COMPETITIONS.get(competition_name)
        if not comp_info:
            raise ValueError(
                f"Unknown competition: {competition_name}. "
                f"Available: {list(COMPETITIONS.keys())}"
            )

        season_id = comp_info["seasons"].get(season_name)
        if not season_id:
            raise ValueError(
                f"Unknown season: {season_name}. "
                f"Available: {list(comp_info['seasons'].keys())}"
            )

        comp_id = comp_info["competition_id"]
        cache_path = self.cache_dir / f"all_events_{comp_id}_{season_id}.parquet"

        if cache_path.exists():
            logger.info("Loading cached events...")
            return pd.read_parquet(cache_path)

        # Get matches
        matches = self.get_matches(comp_id, season_id)
        all_events = []

        for _, match in tqdm(
            matches.iterrows(),
            total=len(matches),
            desc=f"Fetching {competition_name} {season_name}",
        ):
            try:
                events = self.get_events(match["match_id"])
                events["competition"] = competition_name
                events["season"] = season_name
                all_events.append(events)
            except Exception as e:
                logger.warning(f"Error fetching match {match['match_id']}: {e}")

        if all_events:
            df = pd.concat(all_events, ignore_index=True)
            df.to_parquet(cache_path)
            logger.info(
                f"Built dataset: {len(df)} events from {len(all_events)} matches"
            )
            return df

        return pd.DataFrame()


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = StatsBombLoader()

    # Test: fetch competitions
    comps = loader.get_competitions()
    print(f"\nAvailable competitions: {len(comps)}")
    print(comps[["competition_name", "season_name", "country_name"]].head(10))

    # Test: fetch matches from La Liga 2020/2021
    matches = loader.get_matches(competition_id=11, season_id=90)
    print(f"\nLa Liga 2020/2021 matches: {len(matches)}")
    print(matches[["home_team", "away_team", "home_score", "away_score"]].head())
