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
_pitch_viz = None
_report_gen = None


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


def _get_pitch_viz():
    global _pitch_viz
    if _pitch_viz is None:
        from src.tools.pitch_visualizations import PitchVisualizer
        _pitch_viz = PitchVisualizer()
    return _pitch_viz


def _get_report_gen():
    global _report_gen
    if _report_gen is None:
        from src.tools.report_generator import ScoutingReportGenerator
        _report_gen = ScoutingReportGenerator(_get_store())
    return _report_gen


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
        competition: Filter by competition - 'La Liga', 'Champions League', 'FIFA World Cup', 'Euro', 'Premier League', 'Serie A', 'Bundesliga', 'Ligue 1'
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


@tool
def generate_shot_map(player_name: str, match_id: Optional[int] = None) -> str:
    """Generate a shot map showing where a player takes shots on the pitch.

    Shows goals as green stars and misses/saves as grey circles.
    Size of markers represents xG value. Only works for players with
    StatsBomb event data (La Liga, Champions League, World Cup, Euro).

    Args:
        player_name: Name of the player
        match_id: Optional specific match ID. If None, uses a recent match.

    Returns:
        Path to saved shot map image, or error message
    """
    try:
        from statsbombpy import sb
        import unicodedata
        import pandas as pd

        def remove_accents(text):
            if pd.isna(text):
                return ""
            nfkd = unicodedata.normalize("NFKD", str(text))
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        store = _get_store()
        player = store.get_player(player_name)

        if player is None:
            return f"Player '{player_name}' not found in database."

        # Check if StatsBomb data (not FBref)
        if player.get("data_source") == "FBref":
            return (
                f"Shot maps require StatsBomb event data. {player['player']} is from FBref "
                f"(2024/25 aggregated stats only — no location data). "
                f"Shot maps are available for players in La Liga, Champions League, World Cup, and Euro."
            )

        full_name = player["player"]
        comp = player.get("competition", "")
        season = player.get("season", "")

        # Map competition to StatsBomb IDs
        comp_map = {
            "La Liga": (11, {"2020/2021": 90, "2019/2020": 42, "2018/2019": 4}),
            "Champions League": (16, {"2018/2019": 4, "2017/2018": 3}),
            "FIFA World Cup": (43, {"2022": 106, "2018": 3}),
            "Euro": (55, {"2024": 282, "2020": 43}),
        }

        if comp not in comp_map:
            return f"Shot maps not available for {comp}. Available: La Liga, Champions League, World Cup, Euro."

        comp_id, season_map = comp_map[comp]

        # Find best season
        season_id = None
        for s, sid in season_map.items():
            if s in str(season):
                season_id = sid
                break
        if season_id is None:
            season_id = list(season_map.values())[0]

        # Get matches
        matches = sb.matches(competition_id=comp_id, season_id=season_id)

        if matches.empty:
            return f"No matches found for {comp} {season}."

        # Find matches with this player (check home/away teams)
        player_team = player.get("team", "")
        team_matches = matches[
            matches["home_team"].str.contains(player_team, case=False, na=False) |
            matches["away_team"].str.contains(player_team, case=False, na=False)
        ]

        if team_matches.empty:
            team_matches = matches.head(5)

        # Collect events from up to 5 matches
        all_events = []
        for _, match in team_matches.head(5).iterrows():
            try:
                events = sb.events(match_id=match["match_id"])
                # Find player by name (accent-proof)
                clean_target = remove_accents(full_name).lower()
                mask = events["player"].apply(
                    lambda x: clean_target in remove_accents(x).lower() if pd.notna(x) else False
                )
                player_events = events[mask]
                if not player_events.empty:
                    all_events.append(player_events)
            except Exception:
                continue

        if not all_events:
            return f"No event data found for {full_name} in {comp}."

        combined = pd.concat(all_events, ignore_index=True)
        shots = combined[combined["type"] == "Shot"]

        if shots.empty:
            return f"No shots found for {full_name} in the loaded matches."

        viz = _get_pitch_viz()
        path = viz.generate_shot_map(combined, full_name)

        if path:
            return f"Shot map generated for {full_name} ({len(shots)} shots from {min(5, len(team_matches))} matches). Saved to: {path}"
        return f"Could not generate shot map for {full_name}."

    except ImportError:
        return "statsbombpy is required for shot maps. Install with: pip install statsbombpy"
    except Exception as e:
        return f"Error generating shot map: {str(e)}"


@tool
def generate_heatmap(
    player_name: str,
    event_types: Optional[str] = None,
) -> str:
    """Generate a heatmap showing where a player is most active on the pitch.

    Shows density of actions using a color gradient. Only works for players
    with StatsBomb event data (La Liga, Champions League, World Cup, Euro).

    Args:
        player_name: Name of the player
        event_types: Comma-separated event types to include, e.g. "Pass,Carry,Dribble".
                     Default: "Pass,Carry,Dribble"

    Returns:
        Path to saved heatmap image, or error message
    """
    try:
        from statsbombpy import sb
        import unicodedata
        import pandas as pd

        def remove_accents(text):
            if pd.isna(text):
                return ""
            nfkd = unicodedata.normalize("NFKD", str(text))
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        store = _get_store()
        player = store.get_player(player_name)

        if player is None:
            return f"Player '{player_name}' not found in database."

        if player.get("data_source") == "FBref":
            return (
                f"Heatmaps require StatsBomb event data. {player['player']} is from FBref "
                f"(2024/25 aggregated stats only). "
                f"Heatmaps are available for players in La Liga, Champions League, World Cup, and Euro."
            )

        full_name = player["player"]
        comp = player.get("competition", "")
        season = player.get("season", "")

        comp_map = {
            "La Liga": (11, {"2020/2021": 90, "2019/2020": 42, "2018/2019": 4}),
            "Champions League": (16, {"2018/2019": 4, "2017/2018": 3}),
            "FIFA World Cup": (43, {"2022": 106, "2018": 3}),
            "Euro": (55, {"2024": 282, "2020": 43}),
        }

        if comp not in comp_map:
            return f"Heatmaps not available for {comp}. Available: La Liga, Champions League, World Cup, Euro."

        comp_id, season_map = comp_map[comp]
        season_id = None
        for s, sid in season_map.items():
            if s in str(season):
                season_id = sid
                break
        if season_id is None:
            season_id = list(season_map.values())[0]

        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        if matches.empty:
            return f"No matches found for {comp} {season}."

        player_team = player.get("team", "")
        team_matches = matches[
            matches["home_team"].str.contains(player_team, case=False, na=False) |
            matches["away_team"].str.contains(player_team, case=False, na=False)
        ]
        if team_matches.empty:
            team_matches = matches.head(5)

        all_events = []
        for _, match in team_matches.head(5).iterrows():
            try:
                events = sb.events(match_id=match["match_id"])
                clean_target = remove_accents(full_name).lower()
                mask = events["player"].apply(
                    lambda x: clean_target in remove_accents(x).lower() if pd.notna(x) else False
                )
                player_events = events[mask]
                if not player_events.empty:
                    all_events.append(player_events)
            except Exception:
                continue

        if not all_events:
            return f"No event data found for {full_name} in {comp}."

        combined = pd.concat(all_events, ignore_index=True)

        # Parse event types
        types_list = None
        if event_types:
            types_list = [t.strip() for t in event_types.split(",")]

        viz = _get_pitch_viz()
        path = viz.generate_heatmap(combined, full_name, types_list)

        if path:
            type_label = ", ".join(types_list) if types_list else "all events"
            return f"Heatmap generated for {full_name} ({type_label}, {min(5, len(team_matches))} matches). Saved to: {path}"
        return f"Could not generate heatmap for {full_name}."

    except ImportError:
        return "statsbombpy is required for heatmaps. Install with: pip install statsbombpy"
    except Exception as e:
        return f"Error generating heatmap: {str(e)}"


@tool
def generate_pass_map(player_name: str) -> str:
    """Generate a pass map showing completed and failed passes on the pitch.

    Green arrows = completed passes, Red arrows = failed passes.
    Only works for players with StatsBomb event data.

    Args:
        player_name: Name of the player

    Returns:
        Path to saved pass map image, or error message
    """
    try:
        from statsbombpy import sb
        import unicodedata
        import pandas as pd

        def remove_accents(text):
            if pd.isna(text):
                return ""
            nfkd = unicodedata.normalize("NFKD", str(text))
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        store = _get_store()
        player = store.get_player(player_name)

        if player is None:
            return f"Player '{player_name}' not found in database."

        if player.get("data_source") == "FBref":
            return (
                f"Pass maps require StatsBomb event data. {player['player']} is from FBref "
                f"(2024/25 aggregated stats only). "
                f"Pass maps are available for players in La Liga, Champions League, World Cup, and Euro."
            )

        full_name = player["player"]
        comp = player.get("competition", "")
        season = player.get("season", "")

        comp_map = {
            "La Liga": (11, {"2020/2021": 90, "2019/2020": 42, "2018/2019": 4}),
            "Champions League": (16, {"2018/2019": 4, "2017/2018": 3}),
            "FIFA World Cup": (43, {"2022": 106, "2018": 3}),
            "Euro": (55, {"2024": 282, "2020": 43}),
        }

        if comp not in comp_map:
            return f"Pass maps not available for {comp}."

        comp_id, season_map = comp_map[comp]
        season_id = None
        for s, sid in season_map.items():
            if s in str(season):
                season_id = sid
                break
        if season_id is None:
            season_id = list(season_map.values())[0]

        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        if matches.empty:
            return f"No matches found for {comp} {season}."

        player_team = player.get("team", "")
        team_matches = matches[
            matches["home_team"].str.contains(player_team, case=False, na=False) |
            matches["away_team"].str.contains(player_team, case=False, na=False)
        ]
        if team_matches.empty:
            team_matches = matches.head(5)

        all_events = []
        for _, match in team_matches.head(3).iterrows():
            try:
                events = sb.events(match_id=match["match_id"])
                clean_target = remove_accents(full_name).lower()
                mask = events["player"].apply(
                    lambda x: clean_target in remove_accents(x).lower() if pd.notna(x) else False
                )
                player_events = events[mask]
                if not player_events.empty:
                    all_events.append(player_events)
            except Exception:
                continue

        if not all_events:
            return f"No event data found for {full_name} in {comp}."

        combined = pd.concat(all_events, ignore_index=True)

        viz = _get_pitch_viz()
        path = viz.generate_pass_map(combined, full_name)

        if path:
            passes = combined[combined["type"] == "Pass"]
            return f"Pass map generated for {full_name} ({len(passes)} passes from {min(3, len(team_matches))} matches). Saved to: {path}"
        return f"Could not generate pass map for {full_name}."

    except ImportError:
        return "statsbombpy is required for pass maps. Install with: pip install statsbombpy"
    except Exception as e:
        return f"Error generating pass map: {str(e)}"


@tool
def generate_scouting_report(player_name: str) -> str:
    """Generate a comprehensive PDF scouting report for a player.

    The report includes:
    - Page 1: Player overview with key stats and percentile bars
    - Page 2: Radar/pizza chart with percentile rankings
    - Page 3: Top 10 similar players with similarity scores

    Args:
        player_name: Name of the player

    Returns:
        Path to saved PDF report
    """
    gen = _get_report_gen()
    path = gen.generate_report(player_name)
    if path:
        return f"Scouting report generated for {player_name}. Saved to: {path}"
    return f"Could not generate report for '{player_name}'. Player may not be in database."


# List of all tools for the agent
ALL_TOOLS = [
    search_players,
    find_similar_players,
    compare_players,
    generate_radar_chart,
    get_player_stats,
    generate_shot_map,
    generate_heatmap,
    generate_pass_map,
    generate_scouting_report,
]
