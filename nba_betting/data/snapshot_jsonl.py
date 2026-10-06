"""DB-free JSONL capture + import for odds snapshots.

Used by the GitHub Actions snapshot loop (see
``.github/workflows/snapshot-odds.yml`` and ``snapshot_loop.py``) which runs
while the user — who lives in Europe and is asleep during the NBA overnight
window — cannot hit `snapshot-odds` directly. The runner writes one JSONL
line per `(game, source)` to `data/odds_snapshots/YYYY-MM-DD.jsonl` (UTC
day of the capture), commits and pushes. On the user's machine
``nba-betting import-snapshots`` reads those files back and inserts rows
into the local SQLite DB.

Design notes:

* **No DB init on the runner.** We deliberately bypass
  ``capture_snapshot()`` / ``snapshot_current_odds()`` which call
  ``get_session()`` / ``init_db()``. GitHub Actions has no persistent
  SQLite, so the runner only needs network egress + JSONL append.
* **Per-day file, UTC boundary.** ``YYYY-MM-DD.jsonl`` uses the UTC date
  of the capture. Each record's own ``game_date`` is the game's ET date
  (``Game.date``), which is what import matches on.
* **Idempotent import.** A record's natural key is
  ``(home_team_id, away_team_id, source, timestamp)`` — one capture of one
  matchup from one source. ``game_date`` is deliberately NOT part of it:
  it is derived at import time from the DB, so it can differ between
  imports, and keying on it is what produced duplicate rows (2026-10 audit).
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from sqlalchemy import select

from nba_betting.data.espn_odds import is_spread_heuristic, market_home_prob
from nba_betting.data.polymarket import (
    NBA_TZ as _NBA_TZ,
    game_date_et,
    index_odds_by_pair,
    match_odds_for_game,
)
from nba_betting.db.models import Game, OddsSnapshot, Team
from nba_betting.db.session import get_session


def _et_date(ts: datetime) -> date:
    """ET calendar date of a naive-UTC (or aware) timestamp."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_NBA_TZ).date()


class _GameIndex:
    """In-memory ``games`` lookup for matching snapshots to the game they
    were tracking (one query instead of one per record)."""

    def __init__(self, session):
        rows = session.execute(
            select(Game.id, Game.home_team_id, Game.away_team_id, Game.date)
        ).all()
        self._by_pair_date = {(h, a, d): (gid, d) for gid, h, a, d in rows}
        self._by_id = {gid: (gid, d) for gid, h, a, d in rows}

    def game(self, game_id: str) -> tuple[str, date] | None:
        """``(game_id, game_date)`` for a stored game id, or None."""
        return self._by_id.get(game_id)

    def resolve(
        self,
        home_id: int,
        away_id: int,
        game_date_hint: date,
        captured_at: datetime,
        game_id_hint: str | None = None,
    ) -> tuple[str, date] | None:
        """``(game_id, game_date)`` of the game a snapshot belongs to, or None.

        The record's own ``game_date`` is the authority: current records
        carry the game's ET date; records written before 2026-05-29 carry
        the game's UTC tip date, which is the ET date + 1 for 8 PM+ ET
        tips. So try the hint, then the day before — never a day before the
        capture's ET date, because snapshots are only taken pre-tipoff.

        The capture timestamp is NOT used as the anchor. The old resolver
        did that with the UTC date, so an 8 PM ET capture (00:xx UTC next
        day) skipped tonight's game and snapped to the next meeting of the
        same home/away pair — e.g. playoff Game 1's closing line was filed
        under Game 2 (36 rows in the 2026 playoffs). There is deliberately
        no "nearest game in a window" fallback either: it would attach late
        preseason captures to an opening-week rematch.
        """
        if game_id_hint and game_id_hint in self._by_id:
            return self._by_id[game_id_hint]
        earliest = _et_date(captured_at)
        for d in (game_date_hint, game_date_hint - timedelta(days=1)):
            if d < earliest:
                continue
            hit = self._by_pair_date.get((home_id, away_id, d))
            if hit is not None:
                return hit
        return None


def reresolve_existing_snapshots(only_unmatched: bool = False) -> dict:
    """Re-derive ``game_date`` + ``game_id`` for stored snapshots via
    :meth:`_GameIndex.resolve`, using each row's stored ``game_date`` as the
    hint.

    Needed on an ongoing basis: snapshots are captured *before* their game
    exists in the ``games`` table (``sync`` only stores completed games), so
    the importer leaves ``game_id`` NULL and keeps the record's own
    ``game_date``. ``sync`` calls this with ``only_unmatched=True`` after
    adding new games, which links exactly those rows. Idempotent.

    A row that an older importer already attached to the WRONG game has its
    original date overwritten, so it cannot be fixed from the DB alone —
    :func:`repair_snapshots` re-matches those from the JSONL files.
    Returns ``{total, updated, unmatched}``.
    """
    session = get_session()
    try:
        query = select(OddsSnapshot)
        if only_unmatched:
            query = query.where(OddsSnapshot.game_id.is_(None))
        rows = session.execute(query).scalars().all()
        index = _GameIndex(session)
        updated = 0
        unmatched = 0
        for r in rows:
            if r.game_date is None or r.timestamp is None:
                unmatched += 1
                continue
            matched = index.resolve(r.home_team_id, r.away_team_id, r.game_date, r.timestamp)
            if matched is None:
                unmatched += 1
                continue
            game_id, game_date = matched
            if r.game_date != game_date or r.game_id != game_id:
                r.game_date = game_date
                r.game_id = game_id
                updated += 1
        session.commit()
        return {"total": len(rows), "updated": updated, "unmatched": unmatched}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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
    """The game's ET calendar date (``Game.date``) as ``YYYY-MM-DD``.

    Falls back to the UTC date of ``game_time_utc``, then to the runner's
    UTC day, when the tip time is missing or unparseable.
    """
    et = game_date_et(game)
    if et:
        return et
    gtu = (game.get("game_time_utc") or "")[:10]
    if len(gtu) == 10 and gtu[4] == "-" and gtu[7] == "-":
        return gtu
    return _utc_today_iso()


def _tip_time_utc(game: dict) -> datetime | None:
    """Aware UTC tip-off time of a game dict, or None."""
    raw = game.get("game_time_utc") or ""
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_games_via_espn(days_ahead: int = 2, errors: list[str] | None = None) -> list[dict]:
    """Game-list source using ESPN's scoreboard endpoint.

    stats.nba.com (what ``nba_api`` hits) does not answer datacenter IPs —
    from a GitHub runner every ScoreboardV3 call hangs until its 30 s read
    timeout, three retries per date (~4.6 min per capture before the
    2026-10 fix). ESPN's ``site.api.espn.com`` has no such block, so it is
    the runner's only game source (``skip_nba_api``) and the local
    fallback when nba_api returns nothing.

    Returns ET-today's scheduled games; if there are none, walks forward up
    to ``days_ahead`` days and returns the first non-empty day. Scoreboard
    exceptions are appended to ``errors`` (when given) so callers can tell
    "no games" from "couldn't ask".

    Game dicts have the same shape as ``_game_dict_from_v3`` so
    ``capture_snapshot_to_jsonl`` doesn't have to branch on the source.
    """
    from nba_betting.data.espn import fetch_scoreboard

    def _fetch_day(target: date) -> list[dict]:
        date_str = target.strftime("%Y%m%d")
        try:
            events = fetch_scoreboard(date_str)
        except Exception as e:  # noqa: BLE001 — network failure
            if errors is not None:
                errors.append(f"espn scoreboard {date_str}: {e}")
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
            games.append({
                # ESPN event ID is NOT a valid ``games.id`` (NBA API
                # format). Emit "" so the record's game_id is NULL; import
                # matches on (pair, game_date) instead.
                "game_id": "",
                "home_team_id": int(home.get("espn_id") or 0),
                "home_team_abbr": home.get("abbr", ""),
                "home_team_name": home.get("name", ""),
                "away_team_id": int(away.get("espn_id") or 0),
                "away_team_abbr": away.get("abbr", ""),
                "away_team_name": away.get("name", ""),
                "status": ev.get("status", ""),
                "status_code": 1,
                # ISO8601 UTC, e.g. ``2026-04-19T23:00Z``.
                "game_time_utc": ev.get("date") or "",
                "home_score": 0,
                "away_score": 0,
            })
        return games

    today_et = datetime.now(_NBA_TZ).date()
    for day_offset in range(0, days_ahead + 1):
        games = _fetch_day(today_et + timedelta(days=day_offset))
        if games:
            return games
    return []


def _espn_odds_by_game(
    games: list[dict], warnings: list[str],
) -> tuple[dict[tuple, dict], int]:
    """ESPN odds for the slate, keyed ``(frozenset(pair), et_date)``.

    ``get_espn_odds()`` with no date returns ESPN's *current* scoreboard —
    once tonight's games have tipped that is still tonight, so tomorrow's
    games (the ones actually being captured) never got an ESPN line. Fetch
    the scoreboard of each date in the slate instead.
    """
    from nba_betting.data.espn_odds import get_espn_odds

    dates = sorted({game_date_et(g) for g in games if game_date_et(g)}) or [None]
    out: dict[tuple, dict] = {}
    n = 0
    for d in dates:
        try:
            rows = get_espn_odds(d.replace("-", "") if d else None)
        except Exception as e:  # pragma: no cover - network failure path
            warnings.append(f"espn fetch failed ({d}): {e}")
            continue
        n += len(rows)
        for o in rows:
            t = o.get("teams", {})
            if len(t) == 2:
                out[(frozenset(t.keys()), d)] = o
    return out, n


def capture_snapshot_to_jsonl(
    out_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
    *,
    timestamp: datetime | None = None,
    skip_nba_api: bool = False,
    last_written: dict | None = None,
    heartbeat: timedelta | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict:
    """Fetch the upcoming slate and current Polymarket/ESPN odds, then
    append one JSONL record per (game, source) to the capture's UTC-day file.

    This intentionally mirrors ``capture_snapshot()`` but skips every
    database call — the GH Actions runner has no SQLite state.

    Args:
        out_dir: Directory to write the JSONL file into (created if missing).
        timestamp: UTC datetime to stamp every record with. Defaults to the
            clock reading taken *after* the odds are fetched, so the stamp
            is when the prices were observed, not when the run started.
        skip_nba_api: Use ESPN as the only game source. Set on GitHub
            runners, where stats.nba.com never answers (see
            ``_fetch_games_via_espn``).
        last_written: Optional mutable dict carried across captures by the
            snapshot loop. A record whose values match the last one written
            for the same (game, source) is skipped unless ``heartbeat`` has
            elapsed — line values are lossless, the file stays small.
        heartbeat: See ``last_written``.
        clock: Returns the current aware UTC time. For tests.

    Returns a status dict:
        games, written, deduped, polymarket_lines, espn_lines: ints
        next_tip_utc: aware datetime of the earliest scheduled tip, or None
        games_fetch_failed: True when no games came back AND a game source
            raised — "couldn't ask", as opposed to "nothing scheduled"
        warnings: list[str] — real problems (network failures, no games)
        notes: list[str] — informational (e.g. "used ESPN fallback")
        path: str — the JSONL file
        source: str — which fetch path produced the games
    """
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds

    clock = clock or (lambda: datetime.now(timezone.utc))
    # warnings = actual problems the operator should investigate.
    # notes    = expected events (e.g. "used ESPN fallback") that keep the
    #            CLI status green.
    warnings: list[str] = []
    notes: list[str] = []
    fetch_errors: list[str] = []

    games: list[dict] = []
    source = "espn"
    if not skip_nba_api:
        # Prefer nba_api locally (authoritative team/game IDs); fall through
        # to ESPN only when it returns nothing.
        source = "nba-api"
        # Exhibitions included: capturing their odds is harmless (they
        # never join a stored game) and keeps the pipeline exercised.
        games = fetch_todays_games(include_exhibition=True)
        if not games:
            games = fetch_upcoming_games(days_ahead=2, include_exhibition=True)
    if not games:
        espn_games = _fetch_games_via_espn(days_ahead=2, errors=fetch_errors)
        if espn_games:
            games = espn_games
            if not skip_nba_api:
                source = "espn-fallback"
                notes.append(
                    "nba-api returned 0 games; using ESPN fallback "
                    "(use --skip-nba-api on GitHub Actions: stats.nba.com "
                    "does not answer datacenter IPs)"
                )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not games:
        now = timestamp or clock()
        warnings.extend(fetch_errors)
        warnings.append("no games scheduled")
        return {
            "games": 0, "written": 0, "deduped": 0,
            "polymarket_lines": 0, "espn_lines": 0,
            "next_tip_utc": None,
            "games_fetch_failed": bool(fetch_errors),
            "warnings": warnings, "notes": notes,
            "path": str(out_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"),
            "source": source,
        }

    try:
        polymarket_odds = get_nba_odds()
    except Exception as e:  # pragma: no cover - network failure path
        polymarket_odds = []
        warnings.append(f"polymarket fetch failed: {e}")
    espn_index, espn_lines = _espn_odds_by_game(games, warnings)
    poly_index = index_odds_by_pair(polymarket_odds)

    # Stamp AFTER the odds are in hand.
    now = timestamp or clock()
    # Naive UTC, matching the rest of the pipeline (SQLite has no tzinfo).
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo else now
    ts_iso = now_naive.isoformat(timespec="seconds")
    path = out_dir / f"{now_naive.strftime('%Y-%m-%d')}.jsonl"

    records: list[dict] = []
    deduped = 0

    def _emit(rec: dict, values: tuple) -> None:
        nonlocal deduped
        if last_written is not None:
            k = (rec["game_date"], rec["home_team_abbr"], rec["away_team_abbr"], rec["source"])
            prev = last_written.get(k)
            if (
                prev is not None
                and prev[0] == values
                and heartbeat is not None
                and now_naive - prev[1] < heartbeat
            ):
                deduped += 1
                return
            last_written[k] = (values, now_naive)
        records.append(rec)

    tips = [t for t in (_tip_time_utc(g) for g in games) if t is not None]
    for game in games:
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        if not home_abbr or not away_abbr:
            continue
        game_id = game.get("game_id") or None
        game_date_iso = _game_date_from_game(game)
        et_date = game_date_et(game)
        key = frozenset([home_abbr, away_abbr])

        poly = match_odds_for_game(poly_index, key, et_date)
        if poly:
            home_prob = poly.get("teams", {}).get(home_abbr)
            _emit({
                "game_date": game_date_iso,
                "home_team_abbr": home_abbr,
                "away_team_abbr": away_abbr,
                "source": "polymarket",
                "timestamp": ts_iso,
                "home_prob": home_prob,
                "spread": None,
                "over_under": None,
                "game_id": game_id,
            }, (home_prob,))

        espn = espn_index.get((key, et_date)) or espn_index.get((key, None))
        if espn:
            # Only a moneyline-derived prob is a market price; when ESPN
            # lists just a spread, store the spread and leave home_prob
            # NULL. Raw moneylines ride along (their presence also marks
            # the record as post-fix for the importer).
            home_prob = market_home_prob(espn, home_abbr)
            values = (home_prob, espn.get("spread"), espn.get("over_under"),
                      espn.get("home_moneyline"), espn.get("away_moneyline"))
            if any(v is not None for v in values):
                _emit({
                    "game_date": game_date_iso,
                    "home_team_abbr": home_abbr,
                    "away_team_abbr": away_abbr,
                    "source": "espn",
                    "timestamp": ts_iso,
                    "home_prob": home_prob,
                    "spread": espn.get("spread"),
                    "over_under": espn.get("over_under"),
                    "home_moneyline": espn.get("home_moneyline"),
                    "away_moneyline": espn.get("away_moneyline"),
                    "game_id": game_id,
                }, values)

    # Append-only write. Each run adds a block of lines; never rewrites.
    if records:
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    return {
        "games": len(games),
        "written": len(records),
        "deduped": deduped,
        "polymarket_lines": len(polymarket_odds),
        "espn_lines": espn_lines,
        "next_tip_utc": min(tips) if tips else None,
        "games_fetch_failed": False,
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
    which writes naive UTC. Mixing naive and aware datetimes in the same
    column would break the ordering used by ``get_closing_line``.
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


def _iter_parsed_records(
    files: list[Path], teams: dict[str, int], errors: list[str],
) -> Iterator[dict]:
    """Parse + validate JSONL records. Bad lines go to ``errors``; good ones
    are yielded with ``home_id``, ``away_id``, ``ts``, ``hint`` (the record's
    game_date), ``key`` (the natural key) and ``home_prob`` (legacy ESPN
    spread-proxy values already set to None)."""
    for fpath in files:
        with fpath.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                where = f"{fpath.name}:{line_no}"
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{where} bad JSON: {e}")
                    continue

                home_id = teams.get(rec.get("home_team_abbr"))
                away_id = teams.get(rec.get("away_team_abbr"))
                if not home_id or not away_id:
                    errors.append(
                        f"{where} unknown team "
                        f"{rec.get('home_team_abbr')}/{rec.get('away_team_abbr')}"
                    )
                    continue
                try:
                    hint = _parse_game_date(rec["game_date"])
                    ts = _parse_timestamp(rec["timestamp"])
                except (KeyError, ValueError) as e:
                    errors.append(f"{where} bad date/ts: {e}")
                    continue
                source = rec.get("source")
                if source not in ("polymarket", "espn"):
                    errors.append(f"{where} unknown source {source!r}")
                    continue

                home_prob = rec.get("home_prob")
                # Pre-fix ESPN records (no moneyline fields) stored the
                # 2.5%/pt spread proxy as home_prob — not a market price.
                if (
                    source == "espn"
                    and "home_moneyline" not in rec
                    and is_spread_heuristic(home_prob, rec.get("spread"))
                ):
                    home_prob = None

                yield {
                    "rec": rec,
                    "home_id": home_id,
                    "away_id": away_id,
                    "source": source,
                    "ts": ts,
                    "hint": hint,
                    "key": (home_id, away_id, source, ts),
                    "home_prob": home_prob,
                }


def _target_game(index: _GameIndex, p: dict) -> tuple[date, str | None]:
    """``(game_date, game_id)`` a parsed record should be stored under."""
    rec_game_id = p["rec"].get("game_id") or None
    matched = index.resolve(p["home_id"], p["away_id"], p["hint"], p["ts"], rec_game_id)
    if matched is not None:
        game_id, game_date = matched
        return game_date, game_id
    # Not in the DB yet (future/preseason game): keep the record's own ET
    # date; `sync` links it once the game is stored.
    return p["hint"], rec_game_id


def import_snapshots_jsonl(
    path: Path | str = DEFAULT_SNAPSHOT_DIR,
) -> dict:
    """Import JSONL snapshot records into the local OddsSnapshot table.

    Idempotent on the natural key ``(home_team_id, away_team_id, source,
    timestamp)``: records already present are skipped, so re-importing the
    whole directory every day is safe and cheap (existing keys are loaded
    once, not queried per record).

    Args:
        path: File or directory containing ``*.jsonl`` records.

    Returns dict with counts: {files, records, imported, skipped, errors}.
    """
    p = Path(path)
    session = get_session()
    try:
        teams = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}
        existing = {tuple(row) for row in session.execute(select(
            OddsSnapshot.home_team_id, OddsSnapshot.away_team_id,
            OddsSnapshot.source, OddsSnapshot.timestamp,
        )).all()}
        index = _GameIndex(session)

        files = list(_iter_jsonl_files(p))
        errors: list[str] = []
        records_total = imported = skipped = 0
        for parsed in _iter_parsed_records(files, teams, errors):
            records_total += 1
            if parsed["key"] in existing:
                skipped += 1
                continue
            game_date, game_id = _target_game(index, parsed)
            rec = parsed["rec"]
            session.add(OddsSnapshot(
                game_id=game_id,
                game_date=game_date,
                home_team_id=parsed["home_id"],
                away_team_id=parsed["away_id"],
                source=parsed["source"],
                timestamp=parsed["ts"],
                home_prob=parsed["home_prob"],
                spread=rec.get("spread"),
                over_under=rec.get("over_under"),
            ))
            existing.add(parsed["key"])
            imported += 1

        session.commit()
        return {
            "files": len(files),
            "records": records_total + len(errors),
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# When the Polymarket multi-event collision fix reached main (PR #11).
POLYMARKET_COLLISION_FIX_UTC = datetime(2026, 4, 22, 21, 13, 25)
_COLLISION_TOLERANCE = 0.12
_MARGIN_SD = 13.0   # NBA final-margin SD around the spread, points


def _spread_home_prob(spread: float) -> float:
    """Home win probability implied by the home spread, assuming the final
    margin is normal around -spread with SD ``_MARGIN_SD``. A sanity
    reference only (not stored anywhere)."""
    return 0.5 * (1.0 + math.erf((-spread / _MARGIN_SD) / math.sqrt(2.0)))


def _collision_rows(by_key: dict[tuple, OddsSnapshot]) -> list[tuple]:
    """Natural keys of pre-fix Polymarket rows belonging to games where at
    least one pre-fix price is off by more than ``_COLLISION_TOLERANCE``
    from ESPN's spread captured at the same moment (see repair step 5)."""
    espn_spread = {
        (r.home_team_id, r.away_team_id, r.timestamp): r.spread
        for r in by_key.values() if r.source == "espn" and r.spread is not None
    }
    pre_fix: dict[tuple, list[tuple]] = {}
    flagged: set[tuple] = set()
    for k, r in by_key.items():
        if (r.source != "polymarket" or r.home_prob is None or r.timestamp is None
                or r.timestamp >= POLYMARKET_COLLISION_FIX_UTC):
            continue
        game = (r.game_date, r.home_team_id, r.away_team_id)
        pre_fix.setdefault(game, []).append(k)
        spread = espn_spread.get((r.home_team_id, r.away_team_id, r.timestamp))
        if spread is not None and abs(r.home_prob - _spread_home_prob(spread)) > _COLLISION_TOLERANCE:
            flagged.add(game)
    return [k for game in flagged for k in pre_fix[game]]


def repair_snapshots(
    path: Path | str = DEFAULT_SNAPSHOT_DIR,
    *,
    dry_run: bool = False,
) -> dict:
    """One-off repair of an ``odds_snapshots`` table filled by pre-2026-10
    code. Idempotent; a fresh import with current code needs none of it.

    1. **Duplicates** — rows sharing the natural key (home, away, source,
       timestamp) are collapsed to the lowest id. They came from the old
       import key, which included the derived ``game_date``.
    2. **Wrong game** — every row that came from a JSONL file is re-matched
       from that record's own ``game_date`` (see ``_GameIndex.resolve``),
       moving closing lines that the old UTC-anchored resolver had filed
       under the next game of a series back to their real game. Rows not in
       any file (local ``predict`` captures) carry a real nba_api game_id,
       which wins: their ``game_date`` is set to that game's date (old
       local captures were filed under the UTC date).
    3. **Fake ESPN probabilities** — ESPN ``home_prob`` values that are
       exactly the spread proxy are set to NULL (all ESPN probs captured
       before the moneyline fix; spreads/totals are kept).
    4. **Post-game prices** — rows captured on a later ET day than their
       game, or priced at a resolved 0/1 (>= 0.99 / <= 0.01), are deleted.
       Early local captures stored the settled market (1.0) as the
       "closing line" of two 2026-04-09 games.
    5. **Wrong Polymarket event** — before the multi-event collision fix
       (``POLYMARKET_COLLISION_FIX_UTC``) a pair with several open events
       (playoff Games 1 and 2 at the same arena) could be priced from the
       wrong one. For each game, if any pre-fix Polymarket price disagrees
       by more than ``_COLLISION_TOLERANCE`` with ESPN's spread captured at
       the same moment, all of that game's pre-fix Polymarket rows are
       deleted. After the fix no capture exceeds that gap (0 of 519); before
       it 13 of 100 did, in 4 games.

    Returns ``{duplicates_removed, rematched, espn_probs_nulled,
    postgame_removed, collision_removed, errors, dry_run}``. With
    ``dry_run`` nothing is written.
    """
    session = get_session()
    try:
        rows = session.execute(
            select(OddsSnapshot).order_by(OddsSnapshot.id)
        ).scalars().all()
        by_key: dict[tuple, OddsSnapshot] = {}
        dupes = 0
        for r in rows:
            k = (r.home_team_id, r.away_team_id, r.source, r.timestamp)
            if k in by_key:
                session.delete(r)
                dupes += 1
            else:
                by_key[k] = r

        teams = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}
        index = _GameIndex(session)
        errors: list[str] = []
        rematched = 0
        from_files: set[tuple] = set()
        for parsed in _iter_parsed_records(list(_iter_jsonl_files(Path(path))), teams, errors):
            from_files.add(parsed["key"])
            r = by_key.get(parsed["key"])
            if r is None:
                continue
            game_date, game_id = _target_game(index, parsed)
            if (r.game_date, r.game_id) != (game_date, game_id):
                r.game_date, r.game_id = game_date, game_id
                rematched += 1

        for k, r in by_key.items():
            if k in from_files or r.game_id is None:
                continue
            hit = index.game(r.game_id)
            if hit is not None and r.game_date != hit[1]:
                r.game_date = hit[1]
                rematched += 1

        nulled = 0
        for r in by_key.values():
            if r.source == "espn" and is_spread_heuristic(r.home_prob, r.spread):
                r.home_prob = None
                nulled += 1

        postgame = []
        for k, r in list(by_key.items()):
            if (
                (r.timestamp is not None and r.game_date is not None
                 and _et_date(r.timestamp) > r.game_date)
                or (r.home_prob is not None and (r.home_prob >= 0.99 or r.home_prob <= 0.01))
            ):
                postgame.append(k)
        for k in postgame:
            session.delete(by_key.pop(k))

        collision = _collision_rows(by_key)
        for k in collision:
            session.delete(by_key.pop(k))

        if dry_run:
            session.rollback()
        else:
            session.commit()
        return {
            "duplicates_removed": dupes,
            "rematched": rematched,
            "espn_probs_nulled": nulled,
            "postgame_removed": len(postgame),
            "collision_removed": len(collision),
            "errors": errors,
            "dry_run": dry_run,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
