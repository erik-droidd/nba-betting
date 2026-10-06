"""Preseason / All-Star games are not predicted (2026-10 audit: `predict`
staked $10-50 on 9 preseason games at 17-34% fake edges)."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("gid,expected", [
    ("0012600025", True),    # preseason
    ("0032500001", True),    # All-Star
    ("0022600001", False),   # regular season
    ("0042500405", False),   # playoffs
    ("0052500101", False),   # play-in
    ("0062500001", False),   # NBA Cup final (competitive)
    ("", False), (None, False),
])
def test_is_exhibition_game(gid, expected):
    from nba_betting.data.nba_stats import is_exhibition_game
    assert is_exhibition_game(gid) is expected


def _v3(gid, status=1):
    return {"gameId": gid, "gameStatus": status, "gameTimeUTC": "2026-10-06T23:00:00Z",
            "homeTeam": {"teamTricode": "CHA"}, "awayTeam": {"teamTricode": "BKN"}}


def test_fetchers_skip_exhibitions_unless_asked(monkeypatch):
    from nba_betting.data import nba_stats
    slate = {0: [_v3("0012600025"), _v3("0022600001")], 1: [_v3("0012600030")], 2: [_v3("0022600009")]}
    today = nba_stats._today_et()
    monkeypatch.setattr(nba_stats, "_fetch_v3_games_for_date", lambda d: slate.get((d - today).days, []))

    assert [g["game_id"] for g in nba_stats.fetch_todays_games()] == ["0022600001"]
    assert len(nba_stats.fetch_todays_games(include_exhibition=True)) == 2
    # An exhibition-only day is not "the next game day".
    assert [g["game_id"] for g in nba_stats.fetch_upcoming_games(days_ahead=3)] == ["0022600009"]
    assert [g["game_id"] for g in nba_stats.fetch_upcoming_games(days_ahead=3, include_exhibition=True)] == ["0012600030"]
