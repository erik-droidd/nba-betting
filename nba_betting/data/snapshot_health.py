"""How close to the ACTUAL tip-off each game's odds were last observed.

The question the snapshot pipeline exists to answer is "is the stored
closing line really the closing line?". Two traps, both hit in 2026-10:

* **Listed vs actual tip.** NBA games tip ~10 min after the listed time,
  and the capture keeps a game until ESPN marks it started — so captures
  after the *listed* time are normal and correct. Judge against the actual
  tip, estimated from ESPN play-by-play: each first-quarter play gives
  ``wallclock - elapsed game time``; stoppages only push that later, and
  ESPN pre-logs a few plays with bogus early stamps, so the tip is the
  start of the first dense cluster (>= 8 estimates within 4 min), not the
  minimum or the median (the median drifts ~8 min late).
* **Written vs observed.** The loop doesn't rewrite an unchanged line for up
  to 30 min (heartbeat), so a game's last *written* line can be older than
  its last *observation*. Every capture covers every scheduled game, so the
  last observation is the latest capture time (of any game) before the tip.

Needs ESPN (blocked from some cloud sandboxes); runs from the user's machine.
"""
from __future__ import annotations

import json
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

UTC = timezone.utc
CLUSTER_SIZE = 8
CLUSTER_SPAN = timedelta(minutes=4)
# A capture checks game status, then downloads odds and stamps the rows, so a
# row can be stamped seconds after the jump ball (2026-10-25 IND@BKN: +2 s,
# score 0-0). Real post-tip leakage would show up minutes late (5-min cadence).
POST_TIP_TOLERANCE = timedelta(seconds=60)


def _parse_utc(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def estimate_tip(plays: Iterable[dict]) -> datetime | None:
    """Actual tip-off from ESPN play-by-play (see module docstring)."""
    est: list[datetime] = []
    for p in plays:
        if (p.get("period") or {}).get("number") != 1 or not p.get("wallclock"):
            continue
        clock = (p.get("clock") or {}).get("displayValue", "")
        try:
            if ":" in clock:
                m, s = clock.split(":", 1)
                remaining = int(m) * 60 + float(s)
            else:
                remaining = float(clock)
        except ValueError:
            continue
        est.append(_parse_utc(p["wallclock"]) - timedelta(seconds=720 - remaining))
    est.sort()
    for e in est:
        if sum(1 for x in est if e <= x <= e + CLUSTER_SPAN) >= CLUSTER_SIZE:
            return e
    return None


def _load_jsonl(directory: Path, pattern: str = "*.jsonl") -> list[dict]:
    out: list[dict] = []
    for f in sorted(Path(directory).glob(pattern)):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _ts(rec: dict) -> datetime:
    return datetime.fromisoformat(rec["timestamp"]).replace(tzinfo=UTC)


def game_health(
    *,
    tip: datetime,
    records: list[dict],
    capture_times: list[datetime],
    injury_rows: list[dict] | None,
    teams: tuple[str, str],
) -> dict:
    """Freshness + sanity numbers for one game (records = its odds lines)."""
    ts = sorted({_ts(r) for r in records})
    pre = [t for t in ts if t <= tip + POST_TIP_TOLERANCE]
    # Other games' captures only prove this game was seen if they precede its
    # tip; the race tolerance applies to this game's own rows alone (a capture
    # 30 s after its tip may already be the next slate, without this game).
    observed = [t for t in capture_times if ts[0] <= t <= tip] + [t for t in pre if t > tip]
    espn = [r for r in records if r.get("source") == "espn"]
    freeze: list[str] | None = None
    if injury_rows is not None:
        freeze = [t for t in teams
                  if any(_parse_utc(r["captured_at"]) > tip for r in injury_rows if r.get("team_abbr") == t)]
    return {
        "captures": len(ts),
        "first_capture_h": (tip - ts[0]).total_seconds() / 3600,
        "last_written_min": (tip - pre[-1]).total_seconds() / 60 if pre else None,
        "last_observed_min": (tip - max(observed)).total_seconds() / 60 if observed else None,
        "post_tip_rows": sum(1 for t in ts if t > tip + POST_TIP_TOLERANCE),
        "espn_rows": len(espn),
        "espn_ml_rows": sum(1 for r in espn if r.get("home_moneyline") is not None),
        "freeze_violations": freeze,
    }


def health_report(
    start: date,
    end: date,
    *,
    season: str | None = "regular-season",
    odds_dir: Path | str = "data/odds_snapshots",
    injury_dir: Path | str = "data/injury_snapshots",
    scoreboard: Callable[[str], list[dict]] | None = None,
    plays_for: Callable[[str], list[dict]] | None = None,
) -> dict:
    """Per-game health for finished games with ET dates in [start, end].

    ``season``: ESPN season slug to keep ("regular-season", "preseason",
    "post-season"), or None for all. ``scoreboard`` / ``plays_for`` are
    injectable for tests; by default they call ESPN.
    """
    from nba_betting.data import espn

    scoreboard = scoreboard or espn.fetch_scoreboard
    plays_for = plays_for or (lambda eid: espn._get("summary", {"event": eid}).get("plays", []))

    records = _load_jsonl(Path(odds_dir))
    capture_times = sorted({_ts(r) for r in records})
    rows: list[dict] = []
    day = start
    while day <= end:
        gd = day.isoformat()
        inj_path = Path(injury_dir) / f"{gd}.jsonl"
        injury_rows = _load_jsonl(Path(injury_dir), f"{gd}.jsonl") if inj_path.exists() else None
        for ev in scoreboard(day.strftime("%Y%m%d")):
            if ev.get("status") != "STATUS_FINAL":
                continue
            if season and ev.get("season_slug") and ev["season_slug"] != season:
                continue
            home = (ev.get("home_team") or {}).get("abbr")
            away = (ev.get("away_team") or {}).get("abbr")
            row = {"date": gd, "game": f"{away}@{home}", "listed_tip": ev.get("date")}
            mine = [r for r in records if r.get("home_team_abbr") == home
                    and r.get("away_team_abbr") == away and r.get("game_date") == gd]
            tip = estimate_tip(plays_for(ev.get("espn_event_id"))) if mine else None
            if not mine:
                row["note"] = "NO DATA"
            elif tip is None:
                row["note"] = "NO TIP"
            else:
                row.update(game_health(tip=tip, records=mine, capture_times=capture_times,
                                       injury_rows=injury_rows, teams=(home, away)))
                row["tip"] = tip
                if injury_rows is None:
                    row["note"] = "no injury file"
                elif row["freeze_violations"]:
                    row["note"] = "FREEZE VIOLATED " + ",".join(row["freeze_violations"])
            rows.append(row)
        day += timedelta(days=1)

    measured = [r["last_observed_min"] for r in rows if r.get("last_observed_min") is not None]
    espn_rows = sum(r.get("espn_rows", 0) for r in rows)
    summary = {
        "games": len(rows),
        "measured": len(measured),
        "median_min": statistics.median(measured) if measured else None,
        "max_min": max(measured) if measured else None,
        "within_10": sum(1 for m in measured if m <= 10),
        "post_tip_rows": sum(r.get("post_tip_rows", 0) for r in rows),
        "missing": sum(1 for r in rows if r.get("note") in ("NO DATA", "NO TIP")),
        "freeze_violations": sum(1 for r in rows if r.get("freeze_violations")),
        "espn_ml_share": (sum(r.get("espn_ml_rows", 0) for r in rows) / espn_rows) if espn_rows else None,
    }
    return {"rows": rows, "summary": summary}
