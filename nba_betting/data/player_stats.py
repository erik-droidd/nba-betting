"""Player roster and stats synchronization from ESPN."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from nba_betting.config import CURRENT_SEASON
from nba_betting.data.espn import (
    fetch_roster, fetch_depth_chart, ESPN_TEAMS, NBA_ABBR_TO_ESPN_ID,
)
from nba_betting.db.models import PlayerStat, Team
from nba_betting.db.session import get_session


def sync_team_roster(team_abbr: str) -> int:
    """Fetch ESPN roster + depth chart for a team and upsert PlayerStat rows.

    Returns count of players synced.
    """
    espn_id = NBA_ABBR_TO_ESPN_ID.get(team_abbr)
    if not espn_id:
        return 0

    session = get_session()
    try:
        # Get NBA.com team_id from DB
        team = session.execute(
            select(Team).where(Team.abbreviation == team_abbr)
        ).scalars().first()
        if not team:
            return 0
        team_id = team.id

        # Fetch depth chart for starter ranking
        try:
            depth_chart = fetch_depth_chart(espn_id)
        except Exception:
            depth_chart = {}

        # Build player_id -> best rank lookup
        pid_to_rank: dict[str, int] = {}
        for pos_players in depth_chart.values():
            for p in pos_players:
                pid = p.get("espn_id", "")
                rank = p.get("rank", 99)
                if pid and (pid not in pid_to_rank or rank < pid_to_rank[pid]):
                    pid_to_rank[pid] = rank

        # Fetch roster
        roster = fetch_roster(espn_id)
        now = datetime.utcnow()
        count = 0

        for player in roster:
            pid = player["espn_id"]
            if not pid:
                continue

            # Check if already synced recently (within 24h)
            existing = session.execute(
                select(PlayerStat).where(
                    PlayerStat.espn_player_id == pid,
                    PlayerStat.season == CURRENT_SEASON,
                )
            ).scalars().first()

            if existing and existing.last_updated:
                if now - existing.last_updated < timedelta(hours=24):
                    count += 1
                    continue

            depth_rank = pid_to_rank.get(pid, 99)

            if existing:
                existing.player_name = player["name"]
                existing.team_id = team_id
                existing.position = player["position"]
                existing.depth_chart_rank = depth_rank
                existing.last_updated = now
            else:
                new_stat = PlayerStat(
                    espn_player_id=pid,
                    player_name=player["name"],
                    team_id=team_id,
                    season=CURRENT_SEASON,
                    position=player["position"],
                    depth_chart_rank=depth_rank,
                    minutes_per_game=0.0,
                    points_per_game=0.0,
                    assists_per_game=0.0,
                    rebounds_per_game=0.0,
                    plus_minus_per_game=0.0,
                    last_updated=now,
                )
                session.add(new_stat)

            count += 1

        session.commit()
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def sync_all_rosters() -> int:
    """Sync rosters for all 30 NBA teams. Returns total players synced."""
    total = 0
    for nba_abbr in NBA_ABBR_TO_ESPN_ID:
        try:
            n = sync_team_roster(nba_abbr)
            total += n
        except Exception:
            continue
    return total


def get_team_players(team_id: int) -> list[dict]:
    """Get current player stats for a team, sorted by depth chart rank.

    Returns list of dicts with player info.
    """
    session = get_session()
    try:
        rows = session.execute(
            select(PlayerStat)
            .where(
                PlayerStat.team_id == team_id,
                PlayerStat.season == CURRENT_SEASON,
            )
            .order_by(PlayerStat.depth_chart_rank)
        ).scalars().all()

        return [
            {
                "espn_player_id": r.espn_player_id,
                "player_name": r.player_name,
                "position": r.position,
                "depth_chart_rank": r.depth_chart_rank,
                "minutes_per_game": r.minutes_per_game or 0.0,
                "points_per_game": r.points_per_game or 0.0,
                "assists_per_game": r.assists_per_game or 0.0,
                "rebounds_per_game": r.rebounds_per_game or 0.0,
                "plus_minus_per_game": r.plus_minus_per_game or 0.0,
            }
            for r in rows
        ]
    finally:
        session.close()
