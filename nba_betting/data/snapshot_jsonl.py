"""DB-free JSONL capture + import for odds snapshots.

Used by the GitHub Actions cron runner (see
``.github/workflows/snapshot-odds.yml``) which runs while the user — who
lives in Europe and is asleep during the NBA overnight window — cannot
hit `snapshot-odds` directly. The runner writes one JSONL line per
`(game, source)` to `data/odds_snapshots/YYYY-MM-DD.jsonl` (UTC day),
commits and pushes. On the user's machine ``nba-betting import-snapshots``
reads that file back and inserts rows into the local SQLite DB.

Design notes (locked with advisor):

* **No DB init on the runner.** We deliberately bypass
  ``capture_snapshot()`` / ``snapshot_current_odds()`` which call
  ``get_session()`` / ``init_db()``. GitHub Actions has no persistent
  SQLite, so the runner only needs network egress + JSONL append.
* **Per-day file, UTC boundary.** ``YYYY-MM-DD.jsonl`` uses the UTC date
  the runner sees. Appending to today's file keeps the filesystem
  git-friendly (small daily diffs) and makes offset-adjusted replay
  trivial locally.
* **Idempotent import.** ``import_snapshots_jsonl`` looks up the natural
  key ``(game_date, home_team_id, away_team_id, source, timestamp)``
  before inserting. Re-importing the same file yields zero new rows.
  We chose query-before-insert over a UNIQUE constraint migration
  because dev-era duplicate rows would break the migration on first
  run; and this path is only hit at import time, so the extra SELECT
  cost is negligible.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from sqlalchemy import select

from nba_betting.db.models import OddsSnapshot, Team
from nba_betting.db.session import get_session


# NBA scheduling timezone. Used for the ESPN fallback so "today" is an
# ET day (matching what `fetch_todays_games` uses) rather than a UTC
# day — a 22:00 ET tipoff on Apr 19 would flip to Apr 20 under UTC.
_NBA_TZ = ZoneInfo("America/New_York")


# Default directory, relative to repo root. The GH Actions workflow
# writes here; the import CLI reads from here unless the user passes
# --path. Committed to the repo so the daily push shows up as a small
# diff the user can review.
DEFAULT_SNAPSHOT_DIR = Path("data/odds_snapshots")


def _utc_today_iso() -> str:
    """UTC day used for the JSONL filename. Matches what a GH-hosted
    runner (UTC by default) sees."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _game_date_from_game(game: dict) -> str:
    """Extract the game's UTC-day ISO date string.

    ``fetch_todays_games`` / ``fetch_upcoming_games`` return
    ``game_time_utc`` as an ISO8601 ``YYYY-MM-DDTHH:MM:SSZ`` string.
    We slice the first 10 chars — safer than ``datetime.fromisoformat``
    which chokes on the trailing ``Z`` in some Python versions.
    """
    gtu = (game.get("game_time_utc") or "")[:10]
    if len(gtu) == 10 and gtu[4] == "-" and gtu[7] == "-":
        return gtu
    # Fallback: whatever UTC date the runner sees.
    return _utc_today_iso()


def _fetch_games_via_espn(days_ahead: int = 2) -> list[dict]:
    """Fallback game-list source using ESPN's scoreboard endpoint.

    The NBA's stats.nba.com (what ``nba_api`` hits) silently returns
    empty payloads to datacenter IPs — GitHub Actions, AWS, Azure, etc.
    ESPN's ``site.api.espn.com`` does not enforce that block, so it's
    a reliable fallback for the GH runner path. When the runner is in
    use ``fetch_todays_games`` returns ``[]`` and we fall through to
    this function.

    Behavior matches ``fetch_todays_games`` + ``fetch_upcoming_games``:
    return ET-today's scheduled games; if there are none, walk forward
    up to ``days_ahead`` days and return the first non-empty day.

    Returns a list of game dicts in the same shape produced by
    ``_game_dict_from_v3`` so ``capture_snapshot_to_jsonl`` doesn't have
    to branch on the source.
    """
    from nba_betting.data.espn import fetch_scoreboard

    def _fetch_day(target: date) -> list[dict]:
        date_str = target.strftime("%Y%m%d")
        try:
            events = fetch_scoreboard(date_str)
        except Exception:
            return []
        games: list[dict] = []
        for ev in events:
            # Only pre-tipoff games — matches the nba_api path, which
            # filters to gameStatus==1. Live/final games would pollute
            # the closing-line series with post-tipoff odds movement.
            if ev.get("status") != "STATUS_SCHEDULED":
                continue
            home = ev.get("home_team") or {}
            away = ev.get("away_team") or {}
            if not home.get("abbr") or not away.get("abbr"):
                continue
            # ESPN's ``date`` field is ISO8601 UTC
            # (``2026-04-19T23:00Z``). Normalize trailing ``Z`` so the
            # downstream slice in ``_game_date_from_game`` sees a value
            # it recognizes.
            game_time_utc = ev.get("date") or ""
            if game_time_utc.endswith("Z") and "+" not in game_time_utc:
                # keep trailing Z — ``_game_date_from_game`` only reads
                # the first 10 chars
                pass
            games.append({
                # ESPN event ID is NOT a valid ``games.id`` (NBA API
                # format). Emit "" so capture_snapshot_to_jsonl's
                # ``game.get("game_id") or None`` coalesces to NULL;
                # the idempotence key uses (date, home, away, source,
                # ts), so we don't need a game_id for dedupe.
                "game_id": "",
                "home_team_id": int(home.get("espn_id") or 0),
                "home_team_abbr": home.get("abbr", ""),
                "home_team_name": home.get("name", ""),
                "away_team_id": int(away.get("espn_id") or 0),
                "away_team_abbr": away.get("abbr", ""),
                "away_team_name": away.get("name", ""),
                "status": ev.get("status", ""),
                "status_code": 1,
                "game_time_utc": game_time_utc,
                "home_score": 0,
                "away_score": 0,
            })
        return games

    today_et = datetime.now(_NBA_TZ).date()
    games = _fetch_day(today_et)
    if games:
        return games
    for day_offset in range(1, days_ahead + 1):
        games = _fetch_day(today_et + timedelta(days=day_offset))
        if games:
            return games
    return []


def capture_snapshot_to_jsonl(
    out_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
    *,
    timestamp: datetime | None = None,
) -> dict:
    """Fetch today + near-future games and current Polymarket/ESPN odds,
    then append one JSONL record per (game, source) to today's file.

    This intentionally mirrors ``capture_snapshot()`` but skips every
    database call — the GH Actions runner has no SQLite state.

    Args:
        out_dir: Directory to write the JSONL file into. Defaults to
            ``data/odds_snapshots/`` under the repo root. Will be
            created if missing.
        timestamp: UTC datetime to stamp every record with. Defaults to
            ``datetime.now(timezone.utc)``. Exposed for tests.

    Returns a status dict:
        games: int — games fetched
        written: int — JSONL lines appended
        polymarket_lines: int
        espn_lines: int
        warnings: list[str] — real problems (network failures, no
            games at all). Triggers a yellow "warn" status in the CLI.
        notes: list[str] — informational messages (e.g. "used ESPN
            fallback"). Does NOT trigger a warn status — these are
            expected/normal events the operator might want to see.
        path: str — absolute path to the JSONL file
        source: str — which fetch path produced the games
    """
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds

    # warnings = actual problems (network failures, parsing errors) that
    # the operator should investigate.
    # notes    = informational / expected events (e.g. "used ESPN
    # fallback") that don't indicate anything is wrong. Separating the
    # two lets the CLI show a green "ok" status when the only thing
    # that happened was the expected GH-Actions fallback path — the
    # previous "warn" label was misleading because it looked like a
    # real failure.
    warnings: list[str] = []
    notes: list[str] = []
    now = timestamp or datetime.now(timezone.utc)
    # Strip tzinfo for consistency with the rest of the pipeline, which
    # stores naive UTC datetimes (SQLite doesn't carry tzinfo).
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now

    # Prefer nba_api (richer metadata, authoritative team IDs). Falls
    # through to ESPN only when nba_api returns nothing, which is the
    # signature of the stats.nba.com datacenter-IP block. We log which
    # source was used so the workflow output makes the failure mode
    # obvious — "games=0 nba-api empty; ESPN returned N" is the tell.
    source = "nba-api"
    games = fetch_todays_games()
    if not games:
        games = fetch_upcoming_games(days_ahead=2)
    if not games:
        espn_games = _fetch_games_via_espn(days_ahead=2)
        if espn_games:
            games = espn_games
            source = "espn-fallback"
            notes.append(
                "nba-api returned 0 games; using ESPN fallback "
                "(normal on GitHub Actions — stats.nba.com blocks "
                "datacenter IPs)"
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{now_naive.strftime('%Y-%m-%d')}.jsonl"

    if not games:
        # Still touch the file so the workflow has something to commit
        # (or explicitly detects "no games" in the log). Avoid writing a
        # record — import step would reject it anyway.
        warnings.append("no games scheduled")
        return {
            "games": 0,
            "written": 0,
            "polymarket_lines": 0,
            "espn_lines": 0,
            "warnings": warnings,
            "notes": notes,
            "path": str(path),
            "source": source,
        }

    try:
        polymarket_odds = get_nba_odds()
    except Exception as e:  # pragma: no cover - network failure path
        polymarket_odds = []
        warnings.append(f"polymarket fetch failed: {e}")

    try:
        from nba_betting.data.espn_odds import get_espn_odds
        espn_odds = get_espn_odds()
    except Exception as e:  # pragma: no cover - network failure path
        espn_odds = []
        warnings.append(f"espn fetch failed: {e}")

    # Build team-pair indices mirroring snapshot_current_odds
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

    ts_iso = now_naive.isoformat(timespec="seconds")
    records: list[dict] = []

    for game in games:
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        if not home_abbr or not away_abbr:
            continue
        game_id = game.get("game_id") or None
        game_date_iso = _game_date_from_game(game)
        key = frozenset([home_abbr, away_abbr])

        poly = poly_by_teams.get(key)
        if poly:
            t = poly.get("teams", {})
            records.append({
                "game_date": game_date_iso,
                "home_team_abbr": home_abbr,
                "away_team_abbr": away_abbr,
                "source": "polymarket",
                "timestamp": ts_iso,
                "home_prob": t.get(home_abbr),
                "spread": None,
                "over_under": None,
                "game_id": game_id,
            })

        espn = espn_by_teams.get(key)
        if espn:
            t = espn.get("teams", {})
            records.append({
                "game_date": game_date_iso,
                "home_team_abbr": home_abbr,
                "away_team_abbr": away_abbr,
                "source": "espn",
                "timestamp": ts_iso,
                "home_prob": t.get(home_abbr),
                "spread": espn.get("spread"),
                "over_under": espn.get("over_under"),
                "game_id": game_id,
            })

    # Append-only write. Each run adds a block of lines; never rewrites.
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    return {
        "games": len(games),
        "written": len(records),
        "polymarket_lines": len(polymarket_odds),
        "espn_lines": len(espn_odds),
        "warnings": warnings,
        "notes": notes,
        "path": str(path),
        "source": source,
    }


def _iter_jsonl_files(path: Path) -> Iterable[Path]:
    """Yield every .jsonl file under ``path``.

    If ``path`` is a file, yields just that file. If it's a directory,
    yields every matching file in lexicographic order (so older days
    import before newer ones — tidier log output).
    """
    if path.is_file():
        yield path
        return
    if path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
        return
    # Non-existent path — caller handles empty result.
    return


def _parse_timestamp(raw: str) -> datetime:
    """Parse an ISO8601 timestamp as a naive UTC datetime.

    We intentionally drop tzinfo for parity with ``snapshot_current_odds``
    which writes ``datetime.utcnow()`` (naive UTC). Mixing naive and
    aware datetimes in the same column would break the ordering used by
    ``get_closing_line``.
    """
    # Accept trailing 'Z'
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_game_date(raw: str) -> date:
    return date.fromisoformat(raw)


def import_snapshots_jsonl(
    path: Path | str = DEFAULT_SNAPSHOT_DIR,
) -> dict:
    """Import JSONL snapshot records into the local OddsSnapshot table.

    Idempotent: a SELECT on the natural key
    ``(game_date, home_team_id, away_team_id, source, timestamp)``
    skips any record that's already present. Safe to rerun after a
    failed commit-back or to re-import old files after a DB rebuild.

    Args:
        path: File or directory containing ``*.jsonl`` records.

    Returns dict with counts: {files, records, imported, skipped, errors}.
    """
    p = Path(path)
    session = get_session()
    try:
        teams = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}

        files = list(_iter_jsonl_files(p))
        records_total = 0
        imported = 0
        skipped = 0
        errors: list[str] = []

        for fpath in files:
            with fpath.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    records_total += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"{fpath.name}:{line_no} bad JSON: {e}")
                        continue

                    home_id = teams.get(rec.get("home_team_abbr"))
                    away_id = teams.get(rec.get("away_team_abbr"))
                    if not home_id or not away_id:
                        errors.append(
                            f"{fpath.name}:{line_no} unknown team "
                            f"{rec.get('home_team_abbr')}/{rec.get('away_team_abbr')}"
                        )
                        continue

                    try:
                        game_date = _parse_game_date(rec["game_date"])
                        ts = _parse_timestamp(rec["timestamp"])
                    except (KeyError, ValueError) as e:
                        errors.append(f"{fpath.name}:{line_no} bad date/ts: {e}")
                        continue

                    source = rec.get("source")
                    if source not in ("polymarket", "espn"):
                        errors.append(f"{fpath.name}:{line_no} unknown source {source!r}")
                        continue

                    # Idempotence check on the natural key. If present,
                    # skip — never update in place (fields should be
                    # immutable for a (game, source, timestamp) tuple).
                    existing = session.execute(
                        select(OddsSnapshot.id)
                        .where(
                            OddsSnapshot.game_date == game_date,
                            OddsSnapshot.home_team_id == home_id,
                            OddsSnapshot.away_team_id == away_id,
                            OddsSnapshot.source == source,
                            OddsSnapshot.timestamp == ts,
                        )
                        .limit(1)
                    ).first()
                    if existing:
                        skipped += 1
                        continue

                    session.add(OddsSnapshot(
                        game_id=rec.get("game_id") or None,
                        game_date=game_date,
                        home_team_id=home_id,
                        away_team_id=away_id,
                        source=source,
                        timestamp=ts,
                        home_prob=rec.get("home_prob"),
                        spread=rec.get("spread"),
                        over_under=rec.get("over_under"),
                    ))
                    imported += 1

        session.commit()
        return {
            "files": len(files),
            "records": records_total,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
