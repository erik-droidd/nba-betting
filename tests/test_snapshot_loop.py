"""Self-paced snapshot loop for the GitHub runner (2026-10 audit: the cron
scheduler delivered ~6 of 33 slots/day, so closing lines were ~97 min old)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nba_betting.data.snapshot_loop import injury_interval, odds_interval, run_loop

T0 = datetime(2026, 10, 7, 16, 0, tzinfo=timezone.utc)
M = lambda n: timedelta(minutes=n)
H = lambda n: timedelta(hours=n)


@pytest.mark.parametrize("until,expected", [
    (H(19), None), (H(18) + M(1), None), (H(18), M(30)), (H(13), M(30)), (H(4), M(30)),
    (H(3), M(15)), (H(1) + M(1), M(15)), (H(1), M(5)), (M(3), M(5)), (-M(10), M(5)),
])
def test_odds_interval_tightens_toward_tip(until, expected):
    assert odds_interval(T0, T0 + until) == expected


def test_odds_interval_stops_without_games():
    assert odds_interval(T0, None) is None


def test_injury_interval():
    assert injury_interval(T0, T0 + H(2)) == M(15)
    assert injury_interval(T0, T0 + H(5)) == H(1)
    assert injury_interval(T0, None) == H(1)


class _Sim:
    """Fake clock + sleep; captures report a fixed slate of tip times."""

    def __init__(self, tips):
        self.now = T0
        self.tips = sorted(tips)
        self.odds_at: list[datetime] = []
        self.injuries_at: list[datetime] = []
        self.commits: list[datetime] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += timedelta(seconds=seconds)

    def odds(self, last_written):
        self.odds_at.append(self.now)
        upcoming = [t for t in self.tips if t > self.now]
        return {"next_tip_utc": upcoming[0] if upcoming else None, "games_fetch_failed": False}

    def injuries(self):
        self.injuries_at.append(self.now)
        return {}

    def commit(self):
        self.commits.append(self.now)
        return True


def test_loop_covers_tips_closely_and_stops_idle_after_the_last():
    tips = [T0 + H(7), T0 + H(7) + M(30), T0 + H(10)]          # 23:00, 23:30, 02:00Z
    sim = _Sim(tips)
    res = run_loop(capture_odds=sim.odds, capture_injuries=sim.injuries, commit=sim.commit,
                   budget=H(20), clock=sim.clock, sleep=sim.sleep)
    assert res.reason == "idle" and not res.should_continue
    for tip in tips:   # a capture within 5 min before every tip
        before = [t for t in sim.odds_at if t <= tip]
        assert tip - before[-1] <= M(5)
    assert sim.odds_at[-1] >= tips[-1]                # stopped only once the last tip came
    assert sim.commits[-1] == sim.now                  # final commit
    gaps = [b - a for a, b in zip(sim.commits, sim.commits[1:])]
    assert all(g >= M(30) for g in gaps[:-1])          # throttled pushes


def test_loop_stops_on_budget_and_asks_for_a_successor():
    sim = _Sim([T0 + H(8)])
    res = run_loop(capture_odds=sim.odds, capture_injuries=sim.injuries, commit=sim.commit,
                   budget=H(5) + M(30), clock=sim.clock, sleep=sim.sleep)
    assert res.reason == "budget" and res.should_continue
    assert sim.now - T0 <= H(5) + M(30)


def test_loop_retries_a_failed_fetch_before_going_idle():
    sim = _Sim([])
    calls = []

    def odds(last_written):
        calls.append(sim.now)
        return {"next_tip_utc": None, "games_fetch_failed": True}
    res = run_loop(capture_odds=odds, capture_injuries=sim.injuries, commit=sim.commit,
                   budget=H(5), clock=sim.clock, sleep=sim.sleep)
    assert res.reason == "idle"
    assert [c - T0 for c in calls] == [M(0), M(10), M(20)]


def test_loop_survives_capture_exceptions():
    sim = _Sim([T0 + H(2)])
    n = {"odds": 0}

    def flaky_odds(last_written):
        n["odds"] += 1
        if n["odds"] == 2:
            raise RuntimeError("polymarket 502")
        return sim.odds(last_written)

    def broken_injuries():
        raise RuntimeError("espn 500")
    res = run_loop(capture_odds=flaky_odds, capture_injuries=broken_injuries, commit=sim.commit,
                   budget=H(6), clock=sim.clock, sleep=sim.sleep)
    assert res.reason == "idle" and res.captures >= 5
    assert any("polymarket 502" in e for e in res.errors)
    assert any("espn 500" in e for e in res.errors)


def test_loop_shares_dedupe_state_across_captures():
    sim = _Sim([T0 + H(1)])
    seen = []

    def odds(last_written):
        seen.append(id(last_written))
        last_written["k"] = 1
        return sim.odds(last_written)
    run_loop(capture_odds=odds, capture_injuries=sim.injuries, commit=sim.commit,
             budget=H(3), clock=sim.clock, sleep=sim.sleep)
    assert len(set(seen)) == 1 and len(seen) > 1


def test_morning_cron_delivery_starts_the_loop_for_an_evening_slate():
    """2026-10-07: GitHub delivered the hourly cron at 08:39 UTC; the first
    tip was 23:00 (14.3 h later). With a 12 h idle window that run stopped
    at once and the chain only restarted at 16:28. With 18 h it loops."""
    sim = _Sim([T0.replace(hour=23, minute=0)])
    sim.now = T0.replace(hour=8, minute=39)
    res = run_loop(capture_odds=sim.odds, capture_injuries=sim.injuries, commit=sim.commit,
                   budget=H(5) + M(30), clock=sim.clock, sleep=sim.sleep)
    assert res.reason == "budget" and res.should_continue
    assert res.captures >= 10


def test_chain_still_stops_overnight_before_an_evening_slate():
    """After the last West Coast tip (~02:15 UTC) the next first tip is
    ~20.75 h away: stop, don't keep a runner up all night."""
    sim = _Sim([T0.replace(hour=23, minute=0)])
    sim.now = T0.replace(hour=2, minute=15)
    res = run_loop(capture_odds=sim.odds, capture_injuries=sim.injuries, commit=sim.commit,
                   budget=H(5) + M(30), clock=sim.clock, sleep=sim.sleep)
    assert res.reason == "idle" and res.captures == 1
