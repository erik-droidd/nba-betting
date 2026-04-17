"""Odds snapshot storage for line movement tracking."""
from __future__ import annotations

from datetime import datetime, date, timedelta

from sqlalchemy import select

from nba_betting.db.models import OddsSnapshot, Team
from nba_betting.db.session import get_session

# Tier 3.4 — dedup window + tolerance. If a new snapshot arrives within
# ``_DEDUP_WINDOW`` of the last one for the same (game, source) and the
# key fields (``home_prob``, ``spread``, ``over_under``) haven't moved
# beyond ``_DEDUP_TOLERANCE_PROB`` / ``_DEDUP_TOLERANCE_SPREAD``, we skip
# the insert. This lets a cron run every 15 min without bloating the
# table when lines aren't moving — common overnight behavior.
_DEDUP_WINDOW = timedelta(hours=4)
_DEDUP_TOLERANCE_PROB = 0.005  # 0.5 pct-point
_DEDUP_TOLERANCE_SPREAD = 0.5  # half a point


def _is_duplicate(
    session,
    *,
    game_date: date,
    home_team_id: int,
    away_team_id: int,
    source: str,
    now: datetime,
    home_prob: float | None,
    spread: float | None,
    over_under: float | None,
) -> bool:
    """Return True if the most recent same-source snapshot is effectively
    identical to what we're about to insert — no line movement in the
    dedup window."""
    latest = session.execute(
        select(OddsSnapshot)
        .where(
            OddsSnapshot.game_date == game_date,
            OddsSnapshot.home_team_id == home_team_id,
            OddsSnapshot.away_team_id == away_team_id,
            OddsSnapshot.source == source,
        )
        .order_by(OddsSnapshot.timestamp.desc())
        .limit(1)
    ).scalars().first()

    if latest is None:
        return False
    if latest.timestamp is None or (now - latest.timestamp) > _DEDUP_WINDOW:
        return False

    def _close(a, b, tol) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= tol

    return (
        _close(latest.home_prob, home_prob, _DEDUP_TOLERANCE_PROB)
        and _close(latest.spread, spread, _DEDUP_TOLERANCE_SPREAD)
        and _close(latest.over_under, over_under, _DEDUP_TOLERANCE_SPREAD)
    )


def snapshot_current_odds(
    games: list[dict],
    polymarket_odds: list[dict],
    espn_odds: list[dict],
) -> int:
    """Save current odds snapshots for today's games.

    Args:
        games: Today's games from fetch_todays_games().
        polymarket_odds: Odds from Polymarket get_nba_odds().
        espn_odds: Odds from ESPN get_espn_odds().

    Returns count of snapshots saved.
    """
    session = get_session()
    try:
        # Build team abbr -> team_id lookup
        teams = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}
        now = datetime.utcnow()
        today = date.today()
        count = 0

        # Index market odds by team pair
        poly_by_teams: dict[frozenset, dict] = {}
        for o in polymarket_odds:
            t = o.get("teams", {})
            if len(t) == 2:
                poly_by_teams[frozenset(t.keys())] = o

        espn_by_teams: dict[frozenset, dict] = {}
        for o in espn_odds:
            t = o.get("teams", {})
            if len(t) == 2:
                espn_by_teams[frozenset(t.keys())] = o

        for game in games:
            home_abbr = game["home_team_abbr"]
            away_abbr = game["away_team_abbr"]
            home_id = teams.get(home_abbr)
            away_id = teams.get(away_abbr)
            if not home_id or not away_id:
                continue

            # If the game is already in the games table (e.g., we're
            # re-snapshotting during the game or after tipoff), attach the
            # game_id so backtest/history can join cleanly. Otherwise the
            # (date, home_id, away_id) tuple still uniquely identifies it
            # within a day.
            game_id = game.get("game_id")
            # game_time_utc is an ISO 8601 string ("2026-04-14T23:00:00Z");
            # parsing the date prefix lets us correctly file future-day
            # snapshots (when `fetch_upcoming_games` returns games past
            # midnight UTC) under the actual game date, so `get_closing_line`
            # joins by (game_date, home_id, away_id) work.
            game_date_val = game.get("game_date")
            if game_date_val is None:
                gtu = (game.get("game_time_utc") or "")[:10]
                if len(gtu) == 10 and gtu[4] == "-" and gtu[7] == "-":
                    try:
                        game_date_val = date.fromisoformat(gtu)
                    except ValueError:
                        game_date_val = today
                else:
                    game_date_val = today

            key = frozenset([home_abbr, away_abbr])

            # Polymarket snapshot
            poly = poly_by_teams.get(key)
            if poly:
                t = poly.get("teams", {})
                poly_home_prob = t.get(home_abbr)
                if not _is_duplicate(
                    session,
                    game_date=game_date_val,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    source="polymarket",
                    now=now,
                    home_prob=poly_home_prob,
                    spread=None,
                    over_under=None,
                ):
                    session.add(OddsSnapshot(
                        game_id=game_id,
                        game_date=game_date_val,
                        home_team_id=home_id,
                        away_team_id=away_id,
                        source="polymarket",
                        timestamp=now,
                        home_prob=poly_home_prob,
                        spread=None,
                        over_under=None,
                    ))
                    count += 1

            # ESPN snapshot
            espn = espn_by_teams.get(key)
            if espn:
                t = espn.get("teams", {})
                espn_home_prob = t.get(home_abbr)
                espn_spread = espn.get("spread")
                espn_ou = espn.get("over_under")
                if not _is_duplicate(
                    session,
                    game_date=game_date_val,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    source="espn",
                    now=now,
                    home_prob=espn_home_prob,
                    spread=espn_spread,
                    over_under=espn_ou,
                ):
                    session.add(OddsSnapshot(
                        game_id=game_id,
                        game_date=game_date_val,
                        home_team_id=home_id,
                        away_team_id=away_id,
                        source="espn",
                        timestamp=now,
                        home_prob=espn_home_prob,
                        spread=espn_spread,
                        over_under=espn_ou,
                    ))
                    count += 1

        session.commit()
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def capture_snapshot() -> dict:
    """Cron-friendly standalone snapshot: fetch today+near-future games,
    fetch current Polymarket + ESPN odds, persist one snapshot per source.

    Designed to be called by a scheduled task (cron / launchd / systemd
    timer) every 15–60 minutes during the NBA season so the
    `odds_snapshots` table accumulates a real line-movement history the
    backtest can later replay against.

    Returns a small status dict with `games`, `saved`, and any warnings —
    handy for cron log parsing.
    """
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds

    warnings = []
    games = fetch_todays_games()
    if not games:
        games = fetch_upcoming_games(days_ahead=2)
    if not games:
        return {"games": 0, "saved": 0, "warnings": ["no games scheduled"]}

    try:
        polymarket_odds = get_nba_odds()
    except Exception as e:
        polymarket_odds = []
        warnings.append(f"polymarket fetch failed: {e}")

    try:
        from nba_betting.data.espn_odds import get_espn_odds
        espn_odds = get_espn_odds()
    except Exception as e:
        espn_odds = []
        warnings.append(f"espn fetch failed: {e}")

    saved = snapshot_current_odds(games, polymarket_odds, espn_odds)
    return {
        "games": len(games),
        "saved": saved,
        "polymarket_lines": len(polymarket_odds),
        "espn_lines": len(espn_odds),
        "warnings": warnings,
    }


def get_closing_line(
    game_date: date,
    home_team_id: int,
    away_team_id: int,
    source_preference: tuple[str, ...] = ("polymarket", "espn"),
) -> dict | None:
    """Return the latest (by timestamp) snapshot for a finished game.

    Used by `backtest.py` to replay real historical odds when snapshots
    exist. Falls through `source_preference` in order — typically
    Polymarket first (more efficient) then ESPN as fallback.

    Returns a dict with home_prob, spread, over_under, source, timestamp,
    or None if no snapshot is found.
    """
    session = get_session()
    try:
        for source in source_preference:
            snap = session.execute(
                select(OddsSnapshot)
                .where(
                    OddsSnapshot.game_date == game_date,
                    OddsSnapshot.home_team_id == home_team_id,
                    OddsSnapshot.away_team_id == away_team_id,
                    OddsSnapshot.source == source,
                    OddsSnapshot.home_prob.isnot(None),
                )
                .order_by(OddsSnapshot.timestamp.desc())
                .limit(1)
            ).scalars().first()
            if snap is not None:
                return {
                    "source": snap.source,
                    "timestamp": snap.timestamp,
                    "home_prob": snap.home_prob,
                    "spread": snap.spread,
                    "over_under": snap.over_under,
                }
        return None
    finally:
        session.close()


def batch_line_movements_by_game() -> dict[tuple, dict]:
    """One-shot load of per-game line movement for the whole training set.

    Returns `{(game_date, home_team_id, away_team_id): {spread_movement, prob_movement, odds_disagreement}}`.
    Used by the feature builder to attach historical line-movement features
    to every row in a single pass instead of N round-trips to `get_line_movement`.

    Games with no snapshots are simply absent from the dict — callers fill
    zeros, which is the "unknown / pre-snapshot era" convention used
    throughout the pipeline.
    """
    session = get_session()
    try:
        rows = session.execute(
            select(OddsSnapshot)
            .order_by(OddsSnapshot.timestamp)
        ).scalars().all()
    finally:
        session.close()

    # Bucket snapshots by (date, home_id, away_id) so we can reduce each
    # bucket to one feature dict. This is O(N) over the snapshot table.
    buckets: dict[tuple, list] = {}
    for s in rows:
        if s.game_date is None:
            continue
        key = (s.game_date, s.home_team_id, s.away_team_id)
        buckets.setdefault(key, []).append(s)

    out: dict[tuple, dict] = {}
    for key, snaps in buckets.items():
        snaps.sort(key=lambda r: r.timestamp or datetime.min)

        espn_snaps = [s for s in snaps if s.source == "espn" and s.spread is not None]
        if len(espn_snaps) >= 2:
            spread_movement = float((espn_snaps[-1].spread or 0) - (espn_snaps[0].spread or 0))
        else:
            spread_movement = 0.0

        prob_snaps = [s for s in snaps if s.home_prob is not None]
        if len(prob_snaps) >= 2:
            prob_movement = float((prob_snaps[-1].home_prob or 0) - (prob_snaps[0].home_prob or 0))
        else:
            prob_movement = 0.0

        poly_latest = None
        espn_latest = None
        for s in reversed(snaps):
            if s.source == "polymarket" and s.home_prob and not poly_latest:
                poly_latest = s.home_prob
            if s.source == "espn" and s.home_prob and not espn_latest:
                espn_latest = s.home_prob
            if poly_latest and espn_latest:
                break
        odds_disagreement = (
            float(abs(poly_latest - espn_latest)) if poly_latest and espn_latest else 0.0
        )

        out[key] = {
            "spread_movement": spread_movement,
            "prob_movement": prob_movement,
            "odds_disagreement": odds_disagreement,
        }

    return out


def get_opening_line(
    game_date: date,
    home_team_id: int,
    away_team_id: int,
    source_preference: tuple[str, ...] = ("polymarket", "espn"),
) -> dict | None:
    """Return the EARLIEST (by timestamp) snapshot for a game.

    Mirrors `get_closing_line` but picks the first snapshot instead of
    the last — used for CLV tracking (closing_line - opening_line edge).
    """
    session = get_session()
    try:
        for source in source_preference:
            snap = session.execute(
                select(OddsSnapshot)
                .where(
                    OddsSnapshot.game_date == game_date,
                    OddsSnapshot.home_team_id == home_team_id,
                    OddsSnapshot.away_team_id == away_team_id,
                    OddsSnapshot.source == source,
                    OddsSnapshot.home_prob.isnot(None),
                )
                .order_by(OddsSnapshot.timestamp.asc())
                .limit(1)
            ).scalars().first()
            if snap is not None:
                return {
                    "source": snap.source,
                    "timestamp": snap.timestamp,
                    "home_prob": snap.home_prob,
                    "spread": snap.spread,
                    "over_under": snap.over_under,
                }
        return None
    finally:
        session.close()


def get_line_movement(
    game_date: date,
    home_team_id: int,
    away_team_id: int,
) -> dict:
    """Get line movement data for a specific game.

    Returns dict with opening/current spread and probability, plus movement.
    """
    session = get_session()
    try:
        snapshots = session.execute(
            select(OddsSnapshot)
            .where(
                OddsSnapshot.game_date == game_date,
                OddsSnapshot.home_team_id == home_team_id,
                OddsSnapshot.away_team_id == away_team_id,
            )
            .order_by(OddsSnapshot.timestamp)
        ).scalars().all()

        if not snapshots:
            return {
                "n_snapshots": 0,
                "spread_movement": 0.0,
                "prob_movement": 0.0,
                "odds_disagreement": 0.0,
            }

        # Opening = earliest snapshot, current = latest
        opening = snapshots[0]
        current = snapshots[-1]

        # Spread movement (from ESPN snapshots)
        espn_snaps = [s for s in snapshots if s.source == "espn" and s.spread is not None]
        if len(espn_snaps) >= 2:
            spread_movement = (espn_snaps[-1].spread or 0) - (espn_snaps[0].spread or 0)
        else:
            spread_movement = 0.0

        # Probability movement
        prob_snaps = [s for s in snapshots if s.home_prob is not None]
        if len(prob_snaps) >= 2:
            prob_movement = (prob_snaps[-1].home_prob or 0) - (prob_snaps[0].home_prob or 0)
        else:
            prob_movement = 0.0

        # Cross-source disagreement (Polymarket vs ESPN at same time)
        poly_latest = None
        espn_latest = None
        for s in reversed(snapshots):
            if s.source == "polymarket" and s.home_prob and not poly_latest:
                poly_latest = s.home_prob
            if s.source == "espn" and s.home_prob and not espn_latest:
                espn_latest = s.home_prob
            if poly_latest and espn_latest:
                break

        odds_disagreement = abs(poly_latest - espn_latest) if poly_latest and espn_latest else 0.0

        return {
            "n_snapshots": len(snapshots),
            "opening_home_prob": opening.home_prob,
            "current_home_prob": current.home_prob,
            "opening_spread": espn_snaps[0].spread if espn_snaps else None,
            "current_spread": espn_snaps[-1].spread if espn_snaps else None,
            "spread_movement": round(spread_movement, 2),
            "prob_movement": round(prob_movement, 4),
            "odds_disagreement": round(odds_disagreement, 4),
        }
    finally:
        session.close()
