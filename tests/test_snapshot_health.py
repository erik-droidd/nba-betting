"""`snapshot-health`: closing-line freshness against the ACTUAL tip."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from nba_betting.data.snapshot_health import estimate_tip, game_health, health_report

UTC = timezone.utc
TIP = datetime(2026, 10, 7, 2, 14, 0, tzinfo=UTC)


def _play(at: datetime, clock: str, period: int = 1) -> dict:
    return {"wallclock": at.strftime("%Y-%m-%dT%H:%M:%SZ"), "period": {"number": period},
            "clock": {"displayValue": clock}}


def _plays_with_bogus_start() -> list[dict]:
    """2026-10-06 LAL@GSW shape: 3 entries pre-logged 20 min early, then the
    real game; stoppages make later plays lag further behind game time."""
    bogus = [_play(TIP - timedelta(minutes=20, seconds=40), "12:00"),
             _play(TIP - timedelta(minutes=20, seconds=17), "11:37"),
             _play(TIP - timedelta(minutes=20, seconds=17), "11:37")]
    real, wall = [], TIP
    for i in range(1, 40):
        wall += timedelta(seconds=20 + (60 if i % 7 == 0 else 0))   # occasional stoppage
        game_left = 720 - 12 * i
        real.append(_play(wall, f"{game_left // 60}:{game_left % 60:02d}"))
    return bogus + real


def test_estimate_tip_skips_bogus_early_stamps_and_ignores_stoppage_drift():
    tip = estimate_tip(_plays_with_bogus_start())
    assert abs((tip - TIP).total_seconds()) <= 10


def test_estimate_tip_needs_enough_plays():
    assert estimate_tip([_play(TIP, "11:50")] * 3) is None


def _rec(t: datetime, source="polymarket", **kw) -> dict:
    return {"game_date": "2026-10-06", "home_team_abbr": "GSW", "away_team_abbr": "LAL", "source": source,
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"), "home_prob": 0.6, **kw}


def test_game_health_uses_union_of_captures_and_tolerates_the_status_race():
    mine = [_rec(TIP - timedelta(hours=3)), _rec(TIP - timedelta(minutes=12), "espn", home_moneyline=-150.0),
            _rec(TIP + timedelta(seconds=2))]                    # stamped 2 s after the jump ball: OK
    other_game_capture = TIP - timedelta(minutes=2)             # this game unchanged then, not rewritten
    h = game_health(tip=TIP, records=mine, capture_times=sorted({*map(lambda r: datetime.fromisoformat(r["timestamp"]).replace(tzinfo=UTC), mine), other_game_capture}),
                    injury_rows=[{"team_abbr": "GSW", "captured_at": "2026-10-07T01:21:00Z"},
                                 {"team_abbr": "LAL", "captured_at": "2026-10-07T02:30:00Z"}],
                    teams=("GSW", "LAL"))
    assert h["post_tip_rows"] == 0
    assert round(h["last_observed_min"], 2) == round(-2 / 60, 2)
    assert (h["espn_rows"], h["espn_ml_rows"]) == (1, 1)
    assert h["freeze_violations"] == ["LAL"]


def test_health_report_end_to_end(tmp_path):
    odds = tmp_path / "odds"; inj = tmp_path / "inj"
    odds.mkdir(); inj.mkdir()
    lines = [_rec(TIP - timedelta(minutes=m)) for m in (180, 30, 4)] + [_rec(TIP + timedelta(minutes=6))]
    (odds / "2026-10-07.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    (inj / "2026-10-06.jsonl").write_text(json.dumps({"team_abbr": "GSW", "captured_at": "2026-10-07T01:00:00Z"}) + "\n")
    board = [
        {"status": "STATUS_FINAL", "season_slug": "regular-season", "espn_event_id": "1", "date": "2026-10-07T02:00Z",
         "home_team": {"abbr": "GSW"}, "away_team": {"abbr": "LAL"}},
        {"status": "STATUS_FINAL", "season_slug": "regular-season", "espn_event_id": "2", "date": "2026-10-07T02:30Z",
         "home_team": {"abbr": "LAC"}, "away_team": {"abbr": "SAC"}},      # never captured
        {"status": "STATUS_FINAL", "season_slug": "preseason", "espn_event_id": "3", "date": "2026-10-06T23:00Z",
         "home_team": {"abbr": "CHA"}, "away_team": {"abbr": "BKN"}},      # filtered out
    ]
    rep = health_report(date(2026, 10, 6), date(2026, 10, 6), odds_dir=odds, injury_dir=inj,
                        scoreboard=lambda d: board if d == "20261006" else [],
                        plays_for=lambda eid: _plays_with_bogus_start())
    s = rep["summary"]
    assert (s["games"], s["measured"], s["missing"]) == (2, 1, 1)
    assert round(s["max_min"]) == 4 and s["post_tip_rows"] == 1 and s["freeze_violations"] == 0


def test_a_later_capture_of_other_games_does_not_count_as_seeing_this_one():
    """2026-10-21 SAC@LAC: a capture 30 s after its tip (of the next day's
    slate) must not make its last observation look like 'at tip'."""
    mine = [_rec(TIP - timedelta(minutes=25))]
    h = game_health(tip=TIP, records=mine, injury_rows=None, teams=("GSW", "LAL"),
                    capture_times=[TIP - timedelta(minutes=25), TIP + timedelta(seconds=30)])
    assert round(h["last_observed_min"]) == 25
