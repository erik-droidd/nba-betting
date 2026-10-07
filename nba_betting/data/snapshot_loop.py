"""Long-running snapshot loop for the GitHub Actions runner.

Why a loop: GitHub's scheduler delivered ~6 of the 33 cron slots per day,
often hours late (2026-09/10: zero runs 09-15 UTC, 55 runs 04-09 UTC when
no slot was scheduled). The last snapshot before a 7 PM ET tip was a
median 97 minutes old, so "closing lines" weren't. One job that stays up
and paces itself does not depend on the scheduler firing on time.

Shape (see ``.github/workflows/snapshot-odds.yml``):

* Each run loops captures for up to ``budget`` (just under GitHub's 6-hour
  job limit), committing + pushing at most every ``push_every``.
* Cadence follows the next tip-off: 30 min when it is >3 h away, 15 min
  within 3 h, 5 min within the last hour. Injuries: every 15 min within
  2 h of a tip (late scratches), hourly otherwise.
* It stops as **idle** when nothing tips within ``idle_after`` (overnight,
  off days, offseason) and as **budget** when the next capture would run
  past the budget. Only a budget stop asks the workflow to dispatch a
  successor run, so the chain carries a game day end to end; idle stops
  end it and an hourly cron restarts it.
* ``idle_after`` is 18 h, not 12: GitHub delivered the hourly cron only
  ~4 times a day (2026-10-07: 01:32, 08:39, 16:28 UTC), so with 12 h the
  08:39 run stopped at once and the chain only restarted at 16:28, 5.5 h
  late. 18 h lets morning deliveries start it for a 23:00 UTC tip, while
  the chain still stops overnight (after the last tip the next is usually
  20+ h away) — except before weekend matinees, which it then covers.
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("nba_betting.snapshot_loop")

IDLE_AFTER = timedelta(hours=18)
PUSH_EVERY = timedelta(minutes=30)
HEARTBEAT = timedelta(minutes=30)
FETCH_RETRY = timedelta(minutes=10)
MAX_FETCH_RETRIES = 2


def odds_interval(now: datetime, next_tip: datetime | None,
                  idle_after: timedelta = IDLE_AFTER) -> timedelta | None:
    """Wait before the next odds capture, or None to stop (nothing tips
    within ``idle_after``)."""
    if next_tip is None:
        return None
    until = next_tip - now
    if until > idle_after:
        return None
    if until > timedelta(hours=3):
        return timedelta(minutes=30)
    if until > timedelta(hours=1):
        return timedelta(minutes=15)
    return timedelta(minutes=5)


def injury_interval(now: datetime, next_tip: datetime | None) -> timedelta:
    """How often to refresh the injury list."""
    if next_tip is not None and next_tip - now <= timedelta(hours=2):
        return timedelta(minutes=15)
    return timedelta(hours=1)


@dataclass
class LoopResult:
    reason: str                 # "idle" | "budget"
    captures: int = 0
    injury_captures: int = 0
    commit_runs: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def should_continue(self) -> bool:
        """Whether the workflow should dispatch a successor run."""
        return self.reason == "budget"


def run_loop(
    *,
    capture_odds: Callable[[dict], dict],
    capture_injuries: Callable[[], dict],
    commit: Callable[[], bool],
    budget: timedelta,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    push_every: timedelta = PUSH_EVERY,
    idle_after: timedelta = IDLE_AFTER,
) -> LoopResult:
    """Capture until idle or out of budget. Collaborators are injected so
    the pacing is testable without network, git, or real time.

    ``capture_odds(last_written)`` returns the dict from
    ``capture_snapshot_to_jsonl``; ``commit()`` commits + pushes whatever
    changed and returns success. A failing capture or commit is logged and
    retried next round — it never ends the loop on its own.
    """
    deadline = clock() + budget
    last_written: dict = {}
    res = LoopResult(reason="idle")
    next_injuries: datetime | None = None
    last_push = clock()
    fetch_failures = 0

    while True:
        now = clock()
        next_tip = None
        fetch_failed = False
        try:
            out = capture_odds(last_written)
            res.captures += 1
            next_tip = out.get("next_tip_utc")
            fetch_failed = bool(out.get("games_fetch_failed"))
        except Exception as e:  # noqa: BLE001 — keep the loop alive
            fetch_failed = True
            res.errors.append(f"odds capture: {e}")
            logger.warning("odds capture failed: %s", e)

        if next_injuries is None or now >= next_injuries:
            try:
                capture_injuries()
                res.injury_captures += 1
            except Exception as e:  # noqa: BLE001
                res.errors.append(f"injury capture: {e}")
                logger.warning("injury capture failed: %s", e)
            next_injuries = now + injury_interval(now, next_tip)

        if clock() - last_push >= push_every:
            if commit():
                res.commit_runs += 1
            last_push = clock()

        wait = odds_interval(now, next_tip, idle_after)
        if wait is None and fetch_failed and fetch_failures < MAX_FETCH_RETRIES:
            # "Couldn't ask" is not "nothing scheduled": retry a couple of
            # times before letting a network blip end the chain.
            fetch_failures += 1
            wait = FETCH_RETRY
        elif not fetch_failed:
            fetch_failures = 0
        if wait is None:
            res.reason = "idle"
            break
        if clock() + wait > deadline:
            res.reason = "budget"
            break
        sleep(wait.total_seconds())

    if commit():
        res.commit_runs += 1
    return res


def shell_commit(cmd: str, cwd: Path | None = None) -> Callable[[], bool]:
    """``commit`` callable that runs ``cmd`` through the shell (the
    workflow's commit-and-push script)."""
    def _commit() -> bool:
        try:
            proc = subprocess.run(cmd, shell=True, cwd=cwd, timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("commit command timed out: %s", cmd)
            return False
        return proc.returncode == 0
    return _commit
