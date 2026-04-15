"""Injury tracking and impact estimation.

Injury data is difficult to scrape reliably (BBRef blocks, NBA.com has no
public API). This module provides:
1. Manual injury input via CLI
2. Impact estimation based on player usage/minutes
3. A hook for future automated scraping when a source becomes available
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import json

from nba_betting.config import DATA_DIR, TEAM_NAME_TO_ABBR

INJURIES_FILE = DATA_DIR / "injuries.json"


@dataclass
class PlayerInjury:
    player_name: str
    team_abbr: str
    status: str  # "Out", "Doubtful", "Questionable", "Probable", "Day-to-Day"
    reason: str
    impact_rating: float = 0.0  # 0-10 scale, estimated impact on team win prob
    date_reported: str = ""
    player_id: str = ""  # ESPN athlete ID for cross-referencing


def _status_multiplier(status: str) -> float:
    """Convert injury status to probability of missing the game."""
    return {
        "Out": 1.0,
        "Doubtful": 0.85,
        "Day-to-Day": 0.60,
        "Questionable": 0.50,
        "Probable": 0.15,
    }.get(status, 0.5)


def load_injuries() -> list[PlayerInjury]:
    """Load current injuries from the JSON file."""
    if not INJURIES_FILE.exists():
        return []
    data = json.loads(INJURIES_FILE.read_text())
    return [PlayerInjury(**entry) for entry in data]


def save_injuries(injuries: list[PlayerInjury]) -> None:
    """Save injuries to the JSON file."""
    DATA_DIR.mkdir(exist_ok=True)
    data = []
    for inj in injuries:
        data.append({
            "player_name": inj.player_name,
            "team_abbr": inj.team_abbr,
            "status": inj.status,
            "reason": inj.reason,
            "impact_rating": inj.impact_rating,
            "date_reported": inj.date_reported,
            "player_id": inj.player_id,
        })
    INJURIES_FILE.write_text(json.dumps(data, indent=2))


def add_injury(
    player_name: str,
    team_abbr: str,
    status: str = "Out",
    reason: str = "",
    impact_rating: float = 5.0,
) -> PlayerInjury:
    """Add a player injury."""
    injuries = load_injuries()
    # Remove existing entry for same player
    injuries = [i for i in injuries if i.player_name.lower() != player_name.lower()]
    injury = PlayerInjury(
        player_name=player_name,
        team_abbr=team_abbr.upper(),
        status=status,
        reason=reason,
        impact_rating=impact_rating,
        date_reported=str(date.today()),
    )
    injuries.append(injury)
    save_injuries(injuries)
    return injury


def remove_injury(player_name: str) -> bool:
    """Remove a player from the injury list. Returns True if found."""
    injuries = load_injuries()
    new_injuries = [i for i in injuries if i.player_name.lower() != player_name.lower()]
    if len(new_injuries) < len(injuries):
        save_injuries(new_injuries)
        return True
    return False


def clear_injuries() -> None:
    """Clear all injuries."""
    save_injuries([])


def _estimate_impact_rating(status: str, depth_rank: int) -> float:
    """Estimate player impact rating from depth chart position and injury status.

    Starters (rank 1) get higher base impact. Status scales the final value.
    """
    if depth_rank <= 1:
        base = 7.0  # Starter
    elif depth_rank == 2:
        base = 4.0  # Key backup
    else:
        base = 2.0  # Bench player

    # Status modifier (already handled by _status_multiplier in edge calc,
    # but we also want the stored rating to reflect severity)
    status_factor = {
        "Out": 1.0,
        "Doubtful": 0.9,
        "Day-to-Day": 0.6,
        "Questionable": 0.5,
        "Probable": 0.2,
    }.get(status, 0.5)

    return round(base * status_factor, 1)


def sync_injuries_from_espn() -> list[PlayerInjury]:
    """Fetch current injuries from ESPN and update the injury list.

    Replaces the current injury list with ESPN data while preserving
    any manual overrides (injuries with no player_id).

    Returns the updated injury list.
    """
    from nba_betting.data.espn import fetch_injuries, fetch_depth_chart, NBA_ABBR_TO_ESPN_ID

    espn_injuries = fetch_injuries()
    if not espn_injuries:
        return load_injuries()

    # Load existing manual overrides (entries with no ESPN player_id)
    existing = load_injuries()
    manual_overrides = {i.player_name.lower(): i for i in existing if not i.player_id}

    # Build depth chart lookup for impact estimation (cache per team)
    depth_cache: dict[str, dict[str, int]] = {}

    def _get_depth_rank(team_abbr: str, player_id: str) -> int:
        """Look up player's depth chart rank. Returns 99 if not found."""
        if team_abbr not in depth_cache:
            espn_id = NBA_ABBR_TO_ESPN_ID.get(team_abbr)
            if espn_id:
                try:
                    chart = fetch_depth_chart(espn_id)
                    pid_to_rank: dict[str, int] = {}
                    for pos_players in chart.values():
                        for p in pos_players:
                            pid = p.get("espn_id", "")
                            rank = p.get("rank", 99)
                            # Keep the best (lowest) rank across positions
                            if pid and (pid not in pid_to_rank or rank < pid_to_rank[pid]):
                                pid_to_rank[pid] = rank
                    depth_cache[team_abbr] = pid_to_rank
                except Exception:
                    depth_cache[team_abbr] = {}
            else:
                depth_cache[team_abbr] = {}
        return depth_cache[team_abbr].get(player_id, 99)

    new_injuries = []
    seen_players: set[str] = set()

    for inj in espn_injuries:
        name = inj["player_name"]
        name_lower = name.lower()

        # Skip if manual override exists for this player
        if name_lower in manual_overrides:
            new_injuries.append(manual_overrides[name_lower])
            seen_players.add(name_lower)
            continue

        team_abbr = inj["team_abbr"]
        status = inj["status"]
        player_id = inj["player_id"]

        depth_rank = _get_depth_rank(team_abbr, player_id)
        impact = _estimate_impact_rating(status, depth_rank)

        new_injuries.append(PlayerInjury(
            player_name=name,
            team_abbr=team_abbr,
            status=status,
            reason=inj.get("description", ""),
            impact_rating=impact,
            date_reported=inj.get("date", str(date.today())),
            player_id=player_id,
        ))
        seen_players.add(name_lower)

    # Keep manual overrides that weren't in ESPN data
    for name_lower, override in manual_overrides.items():
        if name_lower not in seen_players:
            new_injuries.append(override)

    save_injuries(new_injuries)

    # Persist a dated snapshot so future training runs can reconstruct
    # "who was out the day we predicted game X". We don't have an ESPN
    # archive API, so this accumulates going forward — every time the
    # sync runs it upserts one row per (date, player_id).
    try:
        persist_historical_injuries(new_injuries, snapshot_date=date.today())
    except Exception:
        pass  # Non-critical

    return new_injuries


def persist_historical_injuries(
    injuries: list[PlayerInjury],
    snapshot_date: date,
) -> int:
    """Upsert today's injury list into the `historical_injuries` table.

    Running the ESPN sync is idempotent per day — re-running will
    replace the row for any (date, player_id) pair we've already saved.
    This is what lets the backtest feature pipeline answer
    "who was listed Out on 2026-02-14?" months after the fact.

    Returns the number of rows written.
    """
    from sqlalchemy import select, delete

    from nba_betting.db.models import HistoricalInjury, Team
    from nba_betting.db.session import get_session

    session = get_session()
    try:
        teams = {
            t.abbreviation: t.id
            for t in session.execute(select(Team)).scalars().all()
        }
        # Idempotent: wipe anything already stored for this date, then
        # re-insert the current state. Keeps the table a clean daily
        # snapshot even if the cron re-runs multiple times in a day.
        session.execute(
            delete(HistoricalInjury).where(
                HistoricalInjury.snapshot_date == snapshot_date
            )
        )
        count = 0
        for inj in injuries:
            session.add(HistoricalInjury(
                snapshot_date=snapshot_date,
                player_id=inj.player_id or None,
                player_name=inj.player_name,
                team_abbr=inj.team_abbr.upper(),
                team_id=teams.get(inj.team_abbr.upper()),
                status=inj.status,
                reason=inj.reason,
                impact_rating=inj.impact_rating,
            ))
            count += 1
        session.commit()
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_team_impact_out_as_of(team_abbr: str, as_of_date: date) -> float:
    """Reconstruct total injury impact for a team on a given historical date.

    Used by `features/builder.py` to add an injury feature to training.
    Returns sum of `impact_rating * status_miss_prob` across everyone
    listed for that team on that date — the same weighting used by
    `get_team_injury_adjustment`, but reading from the dated snapshot.

    Returns 0.0 if no snapshot exists for that date (older games
    predating the snapshot collector will silently return 0 — the model
    will learn that "unknown" roughly averages out).
    """
    from sqlalchemy import select

    from nba_betting.db.models import HistoricalInjury
    from nba_betting.db.session import get_session

    session = get_session()
    try:
        rows = session.execute(
            select(HistoricalInjury).where(
                HistoricalInjury.snapshot_date == as_of_date,
                HistoricalInjury.team_abbr == team_abbr.upper(),
            )
        ).scalars().all()
        if not rows:
            return 0.0
        total = 0.0
        for r in rows:
            total += (r.impact_rating or 0.0) * _status_multiplier(r.status or "")
        return total
    finally:
        session.close()


def get_team_injury_adjustment(team_abbr: str) -> float:
    """Estimate win probability adjustment for a team's injuries.

    Returns a negative adjustment (e.g., -0.05 means reduce win prob by 5%).
    Based on the impact ratings of injured players weighted by miss probability.

    Rough calibration:
    - Star player (impact 8-10) out: -4% to -6% win prob
    - Starter (impact 5-7) out: -2% to -3% win prob
    - Rotation player (impact 2-4) out: -0.5% to -1% win prob
    """
    injuries = load_injuries()
    team_injuries = [i for i in injuries if i.team_abbr == team_abbr.upper()]

    if not team_injuries:
        return 0.0

    total_adjustment = 0.0
    for injury in team_injuries:
        miss_prob = _status_multiplier(injury.status)
        # Convert 0-10 impact to win prob adjustment
        # Impact 10 = ~6% adjustment, impact 5 = ~2.5%, impact 1 = ~0.5%
        impact_pct = injury.impact_rating * 0.006
        total_adjustment -= miss_prob * impact_pct

    # Cap total adjustment at -15% (even losing 3 starters shouldn't drop more)
    return max(total_adjustment, -0.15)
