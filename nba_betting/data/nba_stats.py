"""NBA.com data fetching via nba_api with rate limiting."""
from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select

from nba_betting.config import NBA_API_DELAY_SECONDS, CURRENT_SEASON
from nba_betting.db.models import Team, Game, GameStats
from nba_betting.db.session import get_session

_last_request_time = 0.0

# NBA scheduling reference timezone. The NBA's "today" is always determined in
# US Eastern time, regardless of where the user is located. A user in Vienna
# at 8 AM CEST is at 2 AM ET — same NBA day. Using local system date here
# would skip games when the user's local date races ahead of ET.
_NBA_TZ = ZoneInfo("America/New_York")


def _today_et() -> date:
    """Return today's date in US Eastern time (NBA scheduling timezone)."""
    return datetime.now(_NBA_TZ).date()


def _game_dict_from_v3(g: dict) -> dict:
    """Build a normalized game dict from a ScoreboardV3 game record."""
    home = g.get("homeTeam", {})
    away = g.get("awayTeam", {})
    return {
        "game_id": g.get("gameId", ""),
        "home_team_id": home.get("teamId", 0),
        "home_team_abbr": home.get("teamTricode", ""),
        "home_team_name": home.get("teamName", ""),
        "away_team_id": away.get("teamId", 0),
        "away_team_abbr": away.get("teamTricode", ""),
        "away_team_name": away.get("teamName", ""),
        "status": g.get("gameStatusText", ""),
        "status_code": g.get("gameStatus", 0),
        "game_time_utc": g.get("gameTimeUTC", ""),
        "home_score": home.get("score", 0) or 0,
        "away_score": away.get("score", 0) or 0,
    }


def _rate_limit():
    """Enforce minimum delay between NBA.com API calls."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < NBA_API_DELAY_SECONDS:
        time.sleep(NBA_API_DELAY_SECONDS - elapsed)
    _last_request_time = time.time()


def fetch_season_games(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Fetch all team-level game logs for a season."""
    from nba_api.stats.endpoints import leaguegamelog

    _rate_limit()
    log = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Regular Season",
    )
    return log.get_data_frames()[0]


def _fetch_v3_games_for_date(target: date) -> list[dict]:
    """Query ScoreboardV3 for a specific date and return all game records.

    Returns the raw V3 game list (any status). Returns [] on API error or
    empty schedule. Caller is responsible for filtering by status.
    """
    from nba_api.stats.endpoints import scoreboardv3
    import warnings as _warnings

    try:
        _rate_limit()
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            sb = scoreboardv3.ScoreboardV3(game_date=target.strftime("%Y-%m-%d"))
        data = sb.get_dict()
    except Exception:
        return []
    return data.get("scoreboard", {}).get("games", []) or []


def fetch_todays_games(include_started: bool = False) -> list[dict]:
    """Fetch today's NBA games using the NBA's Eastern-time "today".

    Determines today's date in US Eastern time (the NBA's scheduling timezone)
    rather than the system's local timezone, so users in any timezone see the
    correct slate. Queries ScoreboardV3 with an explicit date instead of the
    live endpoint, which can return a stale prior-day scoreboard.

    Args:
        include_started: If True, include in-progress and finished games as
            well. Default False returns only scheduled games (status=1).

    Returns:
        List of game dicts for today (ET). Empty if no matching games.

    NBA `gameStatus`:
        1 = Scheduled (not yet started)
        2 = In progress
        3 = Final
    """
    raw_games = _fetch_v3_games_for_date(_today_et())
    games = []
    for g in raw_games:
        status_code = g.get("gameStatus", 0)
        if not include_started and status_code != 1:
            continue
        games.append(_game_dict_from_v3(g))
    return games


def fetch_upcoming_games(days_ahead: int = 7) -> list[dict]:
    """Fetch scheduled NBA games for the next N days using ScoreboardV3.

    Walks forward from tomorrow (in ET) until it finds the first day with at
    least one scheduled game, then returns all scheduled games for that day.
    Used as a fallback when today (ET) has no remaining scheduled games
    (e.g., all games already started/finished, or it's an off day).

    Args:
        days_ahead: Maximum days into the future to scan.

    Returns:
        List of upcoming game dicts (same format as fetch_todays_games()),
        all from the first non-empty future day. Empty if no games found.
    """
    from datetime import timedelta

    today = _today_et()
    for day_offset in range(1, days_ahead + 1):
        target = today + timedelta(days=day_offset)
        raw_games = _fetch_v3_games_for_date(target)
        upcoming = [
            _game_dict_from_v3(g) for g in raw_games
            if g.get("gameStatus", 0) == 1
        ]
        if upcoming:
            return upcoming
    return []


def sync_season(season: str = CURRENT_SEASON) -> int:
    """Sync a season's game data into the database. Returns count of new games added."""
    df = fetch_season_games(season)
    if df.empty:
        return 0

    session = get_session()
    new_games = 0

    try:
        # Build set of existing game IDs
        existing_ids = {
            row[0]
            for row in session.execute(select(Game.id)).all()
        }

        # Process each row - nba_api returns one row per team per game
        # Group by GAME_ID to get both teams' data
        for game_id, group in df.groupby("GAME_ID"):
            game_id_str = str(game_id)
            if game_id_str in existing_ids:
                continue

            if len(group) != 2:
                continue

            # Determine home/away from MATCHUP column (home has "vs.", away has "@")
            rows = group.to_dict("records")
            home_row = None
            away_row = None
            for row in rows:
                matchup = row.get("MATCHUP", "")
                if "vs." in matchup:
                    home_row = row
                elif "@" in matchup:
                    away_row = row

            if not home_row or not away_row:
                continue

            # Ensure teams exist
            for row in [home_row, away_row]:
                team_id = int(row["TEAM_ID"])
                team = session.get(Team, team_id)
                if not team:
                    team = Team(
                        id=team_id,
                        abbreviation=row["TEAM_ABBREVIATION"],
                        name=row["TEAM_NAME"],
                    )
                    session.add(team)

            # Parse date
            game_date = datetime.strptime(home_row["GAME_DATE"], "%Y-%m-%d").date()

            # Create game record
            home_pts = int(home_row["PTS"])
            away_pts = int(away_row["PTS"])
            game = Game(
                id=game_id_str,
                date=game_date,
                season=season,
                home_team_id=int(home_row["TEAM_ID"]),
                away_team_id=int(away_row["TEAM_ID"]),
                home_score=home_pts,
                away_score=away_pts,
                home_win=home_pts > away_pts,
            )
            session.add(game)

            # Create game stats for each team
            for row in [home_row, away_row]:
                stats = GameStats(
                    game_id=game_id_str,
                    team_id=int(row["TEAM_ID"]),
                    fgm=int(row.get("FGM", 0)),
                    fga=int(row.get("FGA", 0)),
                    fg_pct=float(row.get("FG_PCT", 0)),
                    fg3m=int(row.get("FG3M", 0)),
                    fg3a=int(row.get("FG3A", 0)),
                    fg3_pct=float(row.get("FG3_PCT", 0)),
                    ftm=int(row.get("FTM", 0)),
                    fta=int(row.get("FTA", 0)),
                    ft_pct=float(row.get("FT_PCT", 0)),
                    oreb=int(row.get("OREB", 0)),
                    dreb=int(row.get("DREB", 0)),
                    reb=int(row.get("REB", 0)),
                    ast=int(row.get("AST", 0)),
                    stl=int(row.get("STL", 0)),
                    blk=int(row.get("BLK", 0)),
                    tov=int(row.get("TOV", 0)),
                    pts=int(row.get("PTS", 0)),
                    plus_minus=float(row.get("PLUS_MINUS", 0)),
                )
                session.add(stats)

            new_games += 1

        session.commit()
    finally:
        session.close()

    return new_games
