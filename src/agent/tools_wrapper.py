"""
LangChain Tool Wrappers
Wraps our custom tools into LangChain-compatible tools for the agent.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from src.data.feature_store import FeatureStore
from src.tools.search import PlayerSearch
from src.tools.similarity import PlayerSimilarity
from src.tools.compare import PlayerComparison
from src.tools.visualization import PlayerVisualization

logger = logging.getLogger(__name__)

# Initialize shared feature store and tools
_store = None
_search = None
_similarity = None
_comparison = None
_visualization = None


def _get_store():
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


def _get_search():
    global _search
    if _search is None:
        _search = PlayerSearch(_get_store())
    return _search


def _get_similarity():
    global _similarity
    if _similarity is None:
        _similarity = PlayerSimilarity(_get_store())
    return _similarity


def _get_comparison():
    global _comparison
    if _comparison is None:
        _comparison = PlayerComparison(_get_store())
    return _comparison


def _get_visualization():
    global _visualization
    if _visualization is None:
        _visualization = PlayerVisualization(_get_store())
    return _visualization


@tool
def search_players(
    position_group: Optional[str] = None,
    position: Optional[str] = None,
    team: Optional[str] = None,
    competition: Optional[str] = None,
    min_goals_per90: Optional[float] = None,
    min_assists_per90: Optional[float] = None,
    min_dribbles_completed_per90: Optional[float] = None,
    min_progressive_passes_per90: Optional[float] = None,
    min_tackles_won_per90: Optional[float] = None,
    sort_by: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Search and filter players from the database.

    Args:
        position_group: Filter by group - FWD, MID, DEF, or GK
        position: Filter by specific position like 'Left Wing', 'Center Forward'
        team: Filter by team name (partial match)
        competition: Filter by competition - 'La Liga', 'Champions League', 'FIFA World Cup', 'Euro'
        min_goals_per90: Minimum goals per 90 minutes
        min_assists_per90: Minimum assists per 90 minutes
        min_dribbles_completed_per90: Minimum completed dribbles per 90
        min_progressive_passes_per90: Minimum progressive passes per 90
        min_tackles_won_per90: Minimum tackles won per 90
        sort_by: Column to sort by (e.g. 'goals_per90', 'xg_per90', 'dribbles_completed_per90')
        top_n: Number of results to return (default 10)

    Returns:
        Formatted string with matching players and their stats
    """
    search = _get_search()
    results = search.search(
        position_group=position_group,
        position=position,
        team=team,
        competition=competition,
        min_goals_per90=min_goals_per90,
        min_assists_per90=min_assists_per90,
        min_dribbles_completed_per90=min_dribbles_completed_per90,
        min_progressive_passes_per90=min_progressive_passes_per90,
        min_tackles_won_per90=min_tackles_won_per90,
        sort_by=sort_by,
        top_n=top_n,
    )
    return search.format_results(results)


@tool
def find_similar_players(
    player_name: str,
    n_results: int = 10,
    position_group_filter: Optional[str] = None,
    competition_filter: Optional[str] = None,
) -> str:
    """Find players with similar statistical profiles to a given player.

    Uses cosine similarity on per-90 stats within the same position group.

    Args:
        player_name: Name of the target player (partial match works)
        n_results: Number of similar players to return
        position_group_filter: Override position group filter (FWD, MID, DEF, GK)
        competition_filter: Filter by competition

    Returns:
        Formatted string with similar players and similarity scores
    """
    sim = _get_similarity()
    results = sim.find_similar(
        player_name=player_name,
        n_results=n_results,
        position_group_filter=position_group_filter,
        competition_filter=competition_filter,
    )
    return sim.format_results(player_name, results)


@tool
def compare_players(player_names: list) -> str:
    """Compare 2-4 players side by side on key statistics.

    Args:
        player_names: List of 2-4 player names to compare

    Returns:
        Formatted comparison table with key highlights
    """
    comp = _get_comparison()
    result = comp.compare(player_names)
    return comp.format_comparison(result)


@tool
def generate_radar_chart(player_name: str) -> str:
    """Generate a pizza/radar chart showing a player's percentile rankings.

    The chart shows how the player ranks compared to peers in their position group.
    Colors: Green (80+) = Elite, Blue (60+) = Good, Orange (40+) = Average, Red (<40) = Below avg.

    Args:
        player_name: Name of the player

    Returns:
        Path to the saved chart image
    """
    viz = _get_visualization()
    path = viz.generate_pizza(player_name)
    if path:
        return f"Radar chart saved to: {path}"
    return f"Could not generate chart for '{player_name}'. Player may not be in database."


@tool
def get_player_stats(player_name: str) -> str:
    """Get detailed statistics for a specific player.

    Returns per-90 stats, percentile rankings, and metadata.

    Args:
        player_name: Name of the player (partial match works)

    Returns:
        Formatted string with player's full statistical profile
    """
    store = _get_store()
    player = store.get_player(player_name)

    if player is None:
        return f"Player '{player_name}' not found in database."

    output = f"Player Profile: {player['player']}\n"
    output += f"Team: {player['team']}\n"
    output += f"Position: {player['position']} ({player['position_group']})\n"
    output += f"Competition: {player['competition']} {player.get('season', '')}\n"
    output += f"Minutes: {int(player['minutes_played'])} ({int(player['matches_played'])} matches)\n"
    output += "\n--- Per 90 Stats ---\n"

    per90_cols = [c for c in player.index if c.endswith("_per90")]
    for col in sorted(per90_cols):
        pretty = col.replace("_per90", "").replace("_", " ").title()
        val = player[col]
        if isinstance(val, (int, float)):
            output += f"  {pretty}: {val:.3f}\n"

    output += "\n--- Percentile Rankings (vs position group) ---\n"
    pctl_cols = [c for c in player.index if c.endswith("_percentile")]
    for col in sorted(pctl_cols):
        pretty = col.replace("_percentile", "").replace("_", " ").title()
        val = player[col]
        if isinstance(val, (int, float)):
            level = "Elite" if val >= 80 else "Good" if val >= 60 else "Average" if val >= 40 else "Below avg"
            output += f"  {pretty}: {val:.1f}th percentile ({level})\n"

    return output


# List of all tools for the agent
ALL_TOOLS = [
    search_players,
    find_similar_players,
    compare_players,
    generate_radar_chart,
    get_player_stats,
]
