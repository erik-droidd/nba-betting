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

* **One file per ET day, latest capture wins.** ``data/injury_snapshots/
  YYYY-MM-DD.jsonl`` holds the FULL league injury list as of the most
  recent capture on that NBA day (~150 lines). The day is the ET date —
  the same key ``historical_injuries.snapshot_date`` and ``Game.date``
  use — so a 10 PM ET capture still files under the right day. The last
  capture of a day is the one closest to tip-off, i.e. the best proxy
  for "who was actually out", which is what training wants.
* **No-change runs leave the file alone.** The cron fires ~20×/day;
  if nothing but ``captured_at`` would change, the file is not
  rewritten, so the workflow's "anything to commit?" check stays quiet.
* **Import replaces the day.** ``import_injuries_jsonl`` groups records
  by ``snapshot_date`` and upserts each day through
  ``persist_historical_injuries`` (delete-day + insert), so re-importing
  the same files is idempotent and a newer file for a day supersedes
  whatever a local ``predict`` wrote earlier that day.
* **No DB on the runner.** Capture only needs ESPN egress.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
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


def capture_injuries_to_jsonl(
    out_dir: Path | str = DEFAULT_INJURY_SNAPSHOT_DIR,
    *,
    timestamp: datetime | None = None,
    injuries: list | None = None,
) -> dict:
    """Fetch the ESPN injury report (with depth-chart impact ratings) and
    write today's (ET) full list to ``<out_dir>/<YYYY-MM-DD>.jsonl``.

    Args:
        out_dir: Directory for the per-day files (created if missing).
        timestamp: UTC capture time; defaults to now. Exposed for tests.
        injuries: Pre-built ``PlayerInjury`` list (skips ESPN). For tests.

    Returns ``{snapshot_date, players, written, unchanged, path, warnings}``
    — ``written`` is the number of lines written (0 when the file was
    left untouched because nothing changed or nothing was fetched).
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
        "path": str(path.resolve()),
        "warnings": warnings,
    }
    if not injuries:
        warnings.append("no injuries returned; file left untouched")
        return result

    captured_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ordered = sorted(injuries, key=lambda i: ((i.team_abbr or ""), i.player_name.lower()))
    lines = [json.dumps(_record(i, snapshot_date, captured_at), sort_keys=True) for i in ordered]

    if path.exists():
        existing = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if _content_key(existing) == _content_key(lines):
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
