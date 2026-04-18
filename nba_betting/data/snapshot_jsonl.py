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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from nba_betting.db.models import OddsSnapshot, Team
from nba_betting.db.session import get_session


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
        warnings: list[str]
        path: str — absolute path to the JSONL file
    """
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds

    warnings: list[str] = []
    now = timestamp or datetime.now(timezone.utc)
    # Strip tzinfo for consistency with the rest of the pipeline, which
    # stores naive UTC datetimes (SQLite doesn't carry tzinfo).
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now

    games = fetch_todays_games()
    if not games:
        games = fetch_upcoming_games(days_ahead=2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{now_naive.strftime('%Y-%m-%d')}.jsonl"

    if not games:
        # Still touch the file so the workflow has something to commit
        # (or explicitly detects "no games" in the log). Avoid writing a
        # record — import step would reject it anyway.
        return {
            "games": 0,
            "written": 0,
            "polymarket_lines": 0,
            "espn_lines": 0,
            "warnings": ["no games scheduled"],
            "path": str(path),
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
        "path": str(path),
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
