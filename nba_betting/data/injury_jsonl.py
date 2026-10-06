"""DB-free JSONL capture + import for DAILY INJURY snapshots.

Companion to ``snapshot_jsonl.py`` (odds). The ``historical_injuries``
table is what turns the injury features from constant zeros into signal
(``builder._attach_injury_features`` joins it on the game's ET date), but
until 2026-09 it was only written when the user personally ran
``predict`` — 33 days of coverage in five seasons. The GitHub Actions
cron (``.github/workflows/snapshot-odds.yml``) now also runs
``snapshot-injuries --jsonl`` on every firing, so the archive grows on
its own during the season, and ``import-snapshots`` loads it locally.

Design:

* **One file per ET day, each team frozen at its tip-off.**
  ``data/injury_snapshots/YYYY-MM-DD.jsonl`` holds the full league injury
  list for that NBA day (the ET date — the key
  ``historical_injuries.snapshot_date`` and ``Game.date`` use). A team's
  lines keep updating until its game tips, then stay as they were at the
  last pre-tip capture. Training joins this file to the same day's games,
  so a list refreshed after tip-off leaked that night's in-game injuries
  into the "pre-game" features (2026-10-05: an in-game injury was added at
  the 02:14Z capture). Tip times come from ESPN's scoreboard; if it can't
  be read, an existing day file is left untouched.
* **No-change runs leave the file alone.** If nothing but ``captured_at``
  would change, the file is not rewritten, so the workflow's "anything
  to commit?" check stays quiet.
* **Import replaces the day.** ``import_injuries_jsonl`` groups records
  by ``snapshot_date`` and upserts each day through
  ``persist_historical_injuries`` (delete-day + insert), so re-importing
  the same files is idempotent and a newer file for a day supersedes
  whatever a local ``predict`` wrote earlier that day.
* **No DB on the runner.** Capture only needs ESPN egress.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from nba_betting.data.polymarket import NBA_TZ

DEFAULT_INJURY_SNAPSHOT_DIR = Path("data/injury_snapshots")

_FIELDS = ("snapshot_date", "captured_at", "player_name", "player_id",
           "team_abbr", "status", "reason", "impact_rating")


def _record(inj, snapshot_date: date, captured_at: str) -> dict:
    return {
        "snapshot_date": snapshot_date.isoformat(),
        "captured_at": captured_at,
        "player_name": inj.player_name,
        "player_id": inj.player_id or "",
        "team_abbr": (inj.team_abbr or "").upper(),
        "status": inj.status,
        "reason": (inj.reason or "")[:200],   # column width; keeps files small
        "impact_rating": float(inj.impact_rating or 0.0),
    }


def _content_key(lines: list[str]) -> list[str]:
    """Lines with the volatile ``captured_at`` removed — equality means
    the injury list itself is unchanged."""
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        rec.pop("captured_at", None)
        out.append(json.dumps(rec, sort_keys=True))
    return out


_FETCH = object()   # sentinel: look tip-off status up on ESPN


def _started_teams(now: datetime) -> set[str] | None:
    """Abbreviations of teams whose ET-today game has tipped (or is past its
    scheduled tip), from ESPN's scoreboard. ``None`` if it can't be read."""
    from nba_betting.data.espn import fetch_scoreboard

    day = now.astimezone(NBA_TZ).date()
    try:
        events = fetch_scoreboard(day.strftime("%Y%m%d"))
    except Exception:  # noqa: BLE001 — network failure
        return None
    started: set[str] = set()
    for ev in events:
        status = ev.get("status") or ""
        if status in ("STATUS_POSTPONED", "STATUS_CANCELED"):
            continue
        tip = None
        raw = ev.get("date") or ""
        try:
            tip = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
        except ValueError:
            pass
        if tip is not None and tip.tzinfo is None:
            tip = tip.replace(tzinfo=timezone.utc)
        if status != "STATUS_SCHEDULED" or (tip is not None and tip <= now):
            for side in ("home_team", "away_team"):
                abbr = ((ev.get(side) or {}).get("abbr") or "").upper()
                if abbr:
                    started.add(abbr)
    return started


def _read_records(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def capture_injuries_to_jsonl(
    out_dir: Path | str = DEFAULT_INJURY_SNAPSHOT_DIR,
    *,
    timestamp: datetime | None = None,
    injuries: list | None = None,
    started_teams=_FETCH,
) -> dict:
    """Fetch the ESPN injury report (with depth-chart impact ratings) and
    write today's (ET) list to ``<out_dir>/<YYYY-MM-DD>.jsonl``, keeping
    each already-tipped team's lines as they were before its tip-off.

    Args:
        out_dir: Directory for the per-day files (created if missing).
        timestamp: UTC capture time; defaults to now. Exposed for tests.
        injuries: Pre-built ``PlayerInjury`` list (skips ESPN). For tests.
        started_teams: Abbreviations of teams whose game today has tipped;
            ``None`` = unknown. Default: read from ESPN's scoreboard.

    For a tipped team the lines come from today's file if it exists, else
    from the previous day's file (last night's list — stale but pre-tip);
    only if neither exists is the current list used, with a warning.

    Returns ``{snapshot_date, players, written, unchanged, frozen_teams,
    path, warnings}`` — ``written`` is the number of lines written (0 when
    the file was left untouched).
    """
    now = timestamp or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    warnings: list[str] = []

    if injuries is None:
        from nba_betting.data.injuries import build_injury_list_from_espn
        try:
            injuries = build_injury_list_from_espn()
        except Exception as e:  # network / parse failure: never crash the cron
            injuries = []
            warnings.append(f"espn injuries fetch failed: {e}")

    snapshot_date = now.astimezone(NBA_TZ).date()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{snapshot_date.isoformat()}.jsonl"
    result = {
        "snapshot_date": snapshot_date.isoformat(),
        "players": len(injuries),
        "written": 0,
        "unchanged": False,
        "frozen_teams": [],
        "path": str(path.resolve()),
        "warnings": warnings,
    }
    if not injuries:
        warnings.append("no injuries returned; file left untouched")
        return result

    existing = _read_records(path)
    if started_teams is _FETCH:
        started_teams = _started_teams(now)
    if started_teams is None and existing is not None:
        warnings.append(
            "tip-off times unavailable (ESPN scoreboard failed); kept the "
            "existing file so post-tip news can't leak into it"
        )
        result["unchanged"] = True
        return result
    frozen = {t.upper() for t in (started_teams or ())}

    captured_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh = [_record(i, snapshot_date, captured_at) for i in injuries]
    merged = fresh
    if frozen:
        base = existing
        if base is None:
            prev = _read_records(out / f"{(snapshot_date - timedelta(days=1)).isoformat()}.jsonl")
            base = [dict(r, snapshot_date=snapshot_date.isoformat()) for r in prev] if prev else None
        if base is None:
            warnings.append(
                f"no pre-tip injury capture for {', '.join(sorted(frozen))}; "
                "using the current list"
            )
        else:
            merged = (
                [r for r in fresh if r["team_abbr"] not in frozen]
                + [r for r in base if (r.get("team_abbr") or "").upper() in frozen]
            )
            result["frozen_teams"] = sorted(frozen)

    merged.sort(key=lambda r: ((r.get("team_abbr") or ""), (r.get("player_name") or "").lower()))
    lines = [json.dumps(r, sort_keys=True) for r in merged]

    if existing is not None:
        old_lines = [json.dumps(r, sort_keys=True) for r in existing]
        if _content_key(old_lines) == _content_key(lines):
            result["unchanged"] = True
            return result

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["written"] = len(lines)
    return result


def import_injuries_jsonl(path: Path | str = DEFAULT_INJURY_SNAPSHOT_DIR) -> dict:
    """Load per-day injury JSONL files into ``historical_injuries``.

    Each file's records are grouped by ``snapshot_date`` (falling back to
    the file stem) and written through ``persist_historical_injuries``,
    which replaces that day's rows — idempotent, and a newer capture of a
    day supersedes an older one. Returns ``{files, days, rows, errors}``.
    """
    from nba_betting.data.injuries import PlayerInjury, persist_historical_injuries
    from nba_betting.data.snapshot_jsonl import _iter_jsonl_files

    files = list(_iter_jsonl_files(Path(path)))
    days = 0
    rows = 0
    errors: list[str] = []
    for fpath in files:
        by_day: dict[date, list[PlayerInjury]] = {}
        try:
            stem_date = date.fromisoformat(fpath.stem)
        except ValueError:
            stem_date = None
        with fpath.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{fpath.name}:{line_no} bad JSON: {e}")
                    continue
                raw_date = rec.get("snapshot_date")
                try:
                    sd = date.fromisoformat(raw_date) if raw_date else stem_date
                except ValueError:
                    sd = stem_date
                if sd is None:
                    errors.append(f"{fpath.name}:{line_no} no snapshot_date")
                    continue
                name = (rec.get("player_name") or "").strip()
                team = (rec.get("team_abbr") or "").strip().upper()
                if not name or not team:
                    errors.append(f"{fpath.name}:{line_no} missing player/team")
                    continue
                try:
                    impact = float(rec.get("impact_rating") or 0.0)
                except (TypeError, ValueError):
                    impact = 0.0
                by_day.setdefault(sd, []).append(PlayerInjury(
                    player_name=name,
                    team_abbr=team,
                    status=rec.get("status") or "Unknown",
                    reason=rec.get("reason") or "",
                    impact_rating=impact,
                    date_reported=(rec.get("captured_at") or "")[:10],
                    player_id=str(rec.get("player_id") or ""),
                ))
        for sd, injs in sorted(by_day.items()):
            rows += persist_historical_injuries(injs, snapshot_date=sd)
            days += 1
    return {"files": len(files), "days": days, "rows": rows, "errors": errors}
