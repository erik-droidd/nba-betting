"""ESPN API client for NBA data (injuries, odds, depth charts, rosters).

ESPN exposes undocumented public APIs that require no authentication.
Base URL: https://site.api.espn.com/apis/site/v2/sports/basketball/nba
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from nba_betting.config import ESPN_API_BASE, ESPN_API_DELAY_SECONDS

_last_call_time = 0.0


def _rate_limit() -> None:
    """Enforce delay between ESPN API calls."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < ESPN_API_DELAY_SECONDS:
        time.sleep(ESPN_API_DELAY_SECONDS - elapsed)
    _last_call_time = time.time()


def _get(path: str, params: dict | None = None) -> dict:
    """Make a rate-limited GET request to ESPN API."""
    _rate_limit()
    url = f"{ESPN_API_BASE}/{path}" if not path.startswith("http") else path
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# ESPN team ID <-> NBA.com abbreviation mapping
# ---------------------------------------------------------------------------

# ESPN ID -> ESPN abbreviation (from /teams endpoint)
ESPN_TEAMS: dict[int, dict] = {
    1:  {"espn_abbr": "ATL", "nba_abbr": "ATL", "name": "Atlanta Hawks"},
    2:  {"espn_abbr": "BOS", "nba_abbr": "BOS", "name": "Boston Celtics"},
    17: {"espn_abbr": "BKN", "nba_abbr": "BKN", "name": "Brooklyn Nets"},
    30: {"espn_abbr": "CHA", "nba_abbr": "CHA", "name": "Charlotte Hornets"},
    4:  {"espn_abbr": "CHI", "nba_abbr": "CHI", "name": "Chicago Bulls"},
    5:  {"espn_abbr": "CLE", "nba_abbr": "CLE", "name": "Cleveland Cavaliers"},
    6:  {"espn_abbr": "DAL", "nba_abbr": "DAL", "name": "Dallas Mavericks"},
    7:  {"espn_abbr": "DEN", "nba_abbr": "DEN", "name": "Denver Nuggets"},
    8:  {"espn_abbr": "DET", "nba_abbr": "DET", "name": "Detroit Pistons"},
    9:  {"espn_abbr": "GS",  "nba_abbr": "GSW", "name": "Golden State Warriors"},
    10: {"espn_abbr": "HOU", "nba_abbr": "HOU", "name": "Houston Rockets"},
    11: {"espn_abbr": "IND", "nba_abbr": "IND", "name": "Indiana Pacers"},
    12: {"espn_abbr": "LAC", "nba_abbr": "LAC", "name": "LA Clippers"},
    13: {"espn_abbr": "LAL", "nba_abbr": "LAL", "name": "Los Angeles Lakers"},
    29: {"espn_abbr": "MEM", "nba_abbr": "MEM", "name": "Memphis Grizzlies"},
    14: {"espn_abbr": "MIA", "nba_abbr": "MIA", "name": "Miami Heat"},
    15: {"espn_abbr": "MIL", "nba_abbr": "MIL", "name": "Milwaukee Bucks"},
    16: {"espn_abbr": "MIN", "nba_abbr": "MIN", "name": "Minnesota Timberwolves"},
    3:  {"espn_abbr": "NO",  "nba_abbr": "NOP", "name": "New Orleans Pelicans"},
    18: {"espn_abbr": "NY",  "nba_abbr": "NYK", "name": "New York Knicks"},
    25: {"espn_abbr": "OKC", "nba_abbr": "OKC", "name": "Oklahoma City Thunder"},
    19: {"espn_abbr": "ORL", "nba_abbr": "ORL", "name": "Orlando Magic"},
    20: {"espn_abbr": "PHI", "nba_abbr": "PHI", "name": "Philadelphia 76ers"},
    21: {"espn_abbr": "PHX", "nba_abbr": "PHX", "name": "Phoenix Suns"},
    22: {"espn_abbr": "POR", "nba_abbr": "POR", "name": "Portland Trail Blazers"},
    23: {"espn_abbr": "SAC", "nba_abbr": "SAC", "name": "Sacramento Kings"},
    24: {"espn_abbr": "SA",  "nba_abbr": "SAS", "name": "San Antonio Spurs"},
    28: {"espn_abbr": "TOR", "nba_abbr": "TOR", "name": "Toronto Raptors"},
    26: {"espn_abbr": "UTAH","nba_abbr": "UTA", "name": "Utah Jazz"},
    27: {"espn_abbr": "WSH", "nba_abbr": "WAS", "name": "Washington Wizards"},
}

# Lookup helpers
ESPN_ABBR_TO_NBA: dict[str, str] = {v["espn_abbr"]: v["nba_abbr"] for v in ESPN_TEAMS.values()}
NBA_ABBR_TO_ESPN_ID: dict[str, int] = {v["nba_abbr"]: k for k, v in ESPN_TEAMS.items()}


def _espn_abbr_to_nba(abbr: str) -> str:
    """Convert ESPN abbreviation to NBA.com abbreviation."""
    return ESPN_ABBR_TO_NBA.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Endpoint functions
# ---------------------------------------------------------------------------

def fetch_scoreboard(date_str: str | None = None) -> list[dict]:
    """Fetch today's (or a specific date's) NBA scoreboard.

    Args:
        date_str: Date in YYYYMMDD format. None = today.

    Returns:
        List of game dicts with teams, odds, records, stats.
    """
    params = {}
    if date_str:
        params["dates"] = date_str

    data = _get("scoreboard", params)
    events = data.get("events", [])
    games = []

    for event in events:
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        # Parse competitors
        home_team = away_team = None
        for competitor in comp.get("competitors", []):
            team_data = competitor.get("team", {})
            parsed = {
                "espn_id": int(team_data.get("id", 0)),
                "abbr": _espn_abbr_to_nba(team_data.get("abbreviation", "")),
                "name": team_data.get("displayName", ""),
                "score": competitor.get("score", ""),
                "records": [],
                "statistics": [],
            }
            # Records: overall, home, road
            for rec in competitor.get("records", []):
                parsed["records"].append({
                    "type": rec.get("type", ""),
                    "summary": rec.get("summary", ""),
                })
            # Season stats (when available)
            for stat in competitor.get("statistics", []):
                parsed["statistics"].append({
                    "name": stat.get("name", ""),
                    "displayValue": stat.get("displayValue", ""),
                })

            if competitor.get("homeAway") == "home":
                home_team = parsed
            else:
                away_team = parsed

        if not home_team or not away_team:
            continue

        # Parse odds
        odds_data = {}
        odds_list = comp.get("odds", [])
        if odds_list:
            o = odds_list[0]  # Primary provider (usually DraftKings)
            odds_data = {
                "provider": o.get("provider", {}).get("name", ""),
                "spread": _safe_float(o.get("spread")),
                "over_under": _safe_float(o.get("overUnder")),
                "home_moneyline": _safe_float(o.get("homeTeamOdds", {}).get("moneyLine")),
                "away_moneyline": _safe_float(o.get("awayTeamOdds", {}).get("moneyLine")),
            }

        games.append({
            "espn_event_id": event.get("id", ""),
            "date": event.get("date", ""),
            "status": comp.get("status", {}).get("type", {}).get("name", ""),
            "home_team": home_team,
            "away_team": away_team,
            "odds": odds_data,
            "venue": comp.get("venue", {}).get("fullName", ""),
        })

    return games


def fetch_injuries() -> list[dict]:
    """Fetch current NBA injury report from ESPN.

    Returns:
        List of dicts with player_name, player_id, team_abbr, status, description, date.
    """
    data = _get("injuries")
    injuries = []

    for team_entry in data.get("injuries", []):
        # Extract team info
        team_espn_abbr = ""
        team_display = team_entry.get("displayName", "")

        # Try to find ESPN team ID from the team entry
        team_id = team_entry.get("id")
        if team_id:
            team_info = ESPN_TEAMS.get(int(team_id))
            if team_info:
                team_espn_abbr = team_info["nba_abbr"]

        # Fallback: match by display name
        if not team_espn_abbr:
            for info in ESPN_TEAMS.values():
                if info["name"] == team_display:
                    team_espn_abbr = info["nba_abbr"]
                    break

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            injuries.append({
                "player_name": athlete.get("displayName", ""),
                "player_id": str(athlete.get("id", "")),
                "team_abbr": team_espn_abbr,
                "status": inj.get("status", "Unknown"),
                "description": inj.get("longComment", inj.get("shortComment", "")),
                "date": inj.get("date", ""),
            })

    return injuries


def fetch_depth_chart(espn_team_id: int) -> dict[str, list[dict]]:
    """Fetch depth chart for a team.

    Args:
        espn_team_id: ESPN team ID (1-30).

    Returns:
        Dict mapping position abbreviation to ordered list of players.
        E.g., {"PG": [{"name": "...", "espn_id": "...", "rank": 1}, ...], ...}
    """
    data = _get(f"teams/{espn_team_id}/depthcharts")
    result: dict[str, list[dict]] = {}

    for chart in data.get("depthchart", []):
        positions = chart.get("positions", {})
        for pos_key, pos_data in positions.items():
            pos_abbr = pos_data.get("position", {}).get("abbreviation", pos_key.upper())
            players = []
            for i, athlete in enumerate(pos_data.get("athletes", []), 1):
                players.append({
                    "name": athlete.get("displayName", ""),
                    "espn_id": str(athlete.get("id", "")),
                    "rank": i,
                })
            result[pos_abbr] = players

    return result


def fetch_roster(espn_team_id: int) -> list[dict]:
    """Fetch team roster with player details.

    Returns:
        List of player dicts with name, id, position, salary, experience, etc.
    """
    data = _get(f"teams/{espn_team_id}/roster")
    players = []

    for athlete in data.get("athletes", []):
        players.append({
            "name": athlete.get("displayName", ""),
            "espn_id": str(athlete.get("id", "")),
            "position": athlete.get("position", {}).get("abbreviation", ""),
            "jersey": athlete.get("jersey", ""),
            "height": athlete.get("height", 0),
            "weight": athlete.get("weight", 0),
            "age": athlete.get("age", 0),
            "experience": athlete.get("experience", {}).get("years", 0),
            "salary": athlete.get("contract", {}).get("salary", 0),
            "status": athlete.get("status", {}).get("name", "Active"),
        })

    return players


def fetch_game_summary(espn_event_id: str) -> dict:
    """Fetch detailed game summary (box score, ATS, pickcenter, etc.).

    Returns:
        Dict with keys: against_the_spread, pickcenter, season_series, etc.
    """
    data = _get(f"summary", params={"event": espn_event_id})

    result = {
        "event_id": espn_event_id,
        "against_the_spread": data.get("againstTheSpread", {}),
        "pickcenter": data.get("pickcenter", []),
        "standings": data.get("standings", []),
    }

    # Extract season series from header
    header = data.get("header", {})
    for comp in header.get("competitions", []):
        result["season_series"] = comp.get("series", [])

    return result


def _safe_float(val) -> float | None:
    """Safely convert a value to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
