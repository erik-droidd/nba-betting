"""Season string derivation and ET-date keying of odds snapshots (2026-09).

- ``config.season_for_date``: CURRENT_SEASON used to be a hardcoded
  "2025-26", so `sync` would have silently stopped following the league at
  the 2026-27 tip-off. The season rolls over on July 1.
- ``odds_tracker.snapshot_game_date``: predict-time snapshots were filed
  under the UTC date prefix of ``game_time_utc`` while every consumer
  (Game.date, get_closing_line, the tracker's per-game date) keys on the ET
  date — late tip-offs landed a day late and never joined.
"""
from __future__ import annotations

from datetime import date

from nba_betting.config import season_for_date
from nba_betting.data.odds_tracker import snapshot_game_date


def test_season_for_date_rolls_over_on_july_1():
    assert season_for_date(date(2025, 10, 21)) == "2025-26"   # opening night
    assert season_for_date(date(2026, 1, 15)) == "2025-26"    # mid-season
    assert season_for_date(date(2026, 6, 30)) == "2025-26"    # finals / last day
    assert season_for_date(date(2026, 7, 1)) == "2026-27"     # off-season -> upcoming
    assert season_for_date(date(2026, 9, 2)) == "2026-27"
    assert season_for_date(date(2099, 12, 31)) == "2099-00"   # two-digit suffix


def test_snapshot_game_date_uses_et_not_utc():
    fallback = date(2000, 1, 1)
    # 9:30 PM ET tip-off on 2026-04-18 == 2026-04-19T01:30Z: the UTC prefix
    # says the 19th, the NBA game date is the 18th.
    late = {"game_time_utc": "2026-04-19T01:30:00Z"}
    assert snapshot_game_date(late, fallback) == date(2026, 4, 18)
    early = {"game_time_utc": "2026-04-18T23:00:00Z"}   # 7 PM ET
    assert snapshot_game_date(early, fallback) == date(2026, 4, 18)


def test_snapshot_game_date_explicit_and_fallback():
    fallback = date(2000, 1, 1)
    assert snapshot_game_date({"game_date": date(2026, 4, 20)}, fallback) == date(2026, 4, 20)
    assert snapshot_game_date({"game_date": "2026-04-20"}, fallback) == date(2026, 4, 20)
    # Explicit wins over the timestamp.
    assert snapshot_game_date(
        {"game_date": "2026-04-20", "game_time_utc": "2026-04-19T01:30:00Z"}, fallback,
    ) == date(2026, 4, 20)
    assert snapshot_game_date({}, fallback) == fallback
    assert snapshot_game_date({"game_time_utc": "garbage"}, fallback) == fallback
