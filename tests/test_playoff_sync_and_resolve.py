"""Tests for the post-regular-season sync gap + resolve fallback.

Two bugs led to ``performance`` showing the same numbers indefinitely:

1. ``fetch_season_games`` only fetched ``season_type_all_star="Regular Season"``
   from NBA's LeagueGameLog. After the regular season ended (~mid-April), no
   new games entered the games table, so ``update_results`` could never match
   playoff predictions to a stored game and they sat unresolved forever.

2. Pre-PR-#13, ``record_predictions`` used ``date.today()`` (local system
   date), so for users east of US/Eastern an evening prediction could be
   filed under tomorrow-local while the game was stored under today-ET.
   ``update_results`` matched dates exactly and missed.

The fixes here:
- ``fetch_season_games`` now unions Regular Season + PlayIn + Playoffs.
- ``update_results`` falls back to a ±1-day window when the exact date
  match fails, but only resolves when EXACTLY ONE candidate game exists in
  that window — same matchup on consecutive calendar days is effectively
  unheard of in the NBA, so the single-candidate rule keeps the fallback
  safe.

These tests pin those two behaviors so we don't regress next time someone
"simplifies" the sync or the resolver.
"""
from __future__ import annotations

import importlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fix 1 — fetch_season_games unions all competitive segments
# ---------------------------------------------------------------------------


def _segment_frame(game_id: str, segment_label: str) -> pd.DataFrame:
    """Build a minimal LeagueGameLog-shaped frame for a single segment.

    The real endpoint returns dozens of columns; we only need GAME_ID and
    TEAM_ID for the dedupe path under test, plus a label column we can
    inspect to verify the right segment fed each row.
    """
    return pd.DataFrame({
        "GAME_ID": [game_id, game_id],  # both teams of one game
        "TEAM_ID": [1, 2],
        "SEGMENT_LABEL": [segment_label, segment_label],
    })


class _StubGameLog:
    """Capture (season, season_type) calls and return a per-segment frame."""

    calls: list[tuple[str, str]] = []
    payload: dict[str, pd.DataFrame] = {}

    def __init__(self, season, player_or_team_abbreviation, season_type_all_star):
        type(self).calls.append((season, season_type_all_star))
        self._df = type(self).payload.get(season_type_all_star, pd.DataFrame())

    def get_data_frames(self):
        return [self._df]


def test_fetch_season_games_unions_regular_play_in_and_playoffs(monkeypatch):
    """All three competitive segments are fetched and concatenated.

    This is the *primary* bug: pre-fix, only Regular Season was queried, so
    the games table never gained playoff or play-in rows and resolution
    silently broke for an entire post-season.
    """
    from nba_betting.data import nba_stats

    _StubGameLog.calls = []
    _StubGameLog.payload = {
        "Regular Season": _segment_frame("0022500001", "regular"),
        "PlayIn":         _segment_frame("0052500001", "playin"),
        "Playoffs":       _segment_frame("0042500001", "playoffs"),
    }

    # Stub `nba_api.stats.endpoints.leaguegamelog.LeagueGameLog`. The
    # function under test does `from nba_api.stats.endpoints import
    # leaguegamelog`, then `leaguegamelog.LeagueGameLog(...)`. We register a
    # fake submodule under sys.modules so the import resolves to it.
    import sys
    fake_submodule = type("FakeLGL", (), {"LeagueGameLog": _StubGameLog})
    monkeypatch.setitem(
        sys.modules, "nba_api.stats.endpoints.leaguegamelog", fake_submodule,
    )
    monkeypatch.setattr(nba_stats, "_rate_limit", lambda: None)

    df = nba_stats.fetch_season_games("2025-26")

    # All three segments queried, each with the expected season type.
    requested_types = [c[1] for c in _StubGameLog.calls]
    assert requested_types == ["Regular Season", "PlayIn", "Playoffs"], (
        f"expected the three competitive segments to be queried in order, "
        f"got {requested_types}"
    )
    # Output unions all three (3 game IDs × 2 teams = 6 rows).
    assert len(df) == 6
    assert set(df["SEGMENT_LABEL"]) == {"regular", "playin", "playoffs"}


def test_fetch_season_games_tolerates_empty_or_failing_segment(monkeypatch):
    """If a segment errors (e.g. playoffs not yet started), other segments
    still come through. Pre-season is treated as a hard failure scenario
    here — the pipeline shouldn't depend on it.
    """
    from nba_betting.data import nba_stats

    _StubGameLog.calls = []

    class _FailingPlayoffs:
        """Returns the regular/playin frames, raises on Playoffs."""
        def __init__(self, season, player_or_team_abbreviation, season_type_all_star):
            type(self).calls = getattr(type(self), "calls", []) + [season_type_all_star]
            if season_type_all_star == "Playoffs":
                raise RuntimeError("nba_api 500 / playoffs not started")
            self._df = {
                "Regular Season": _segment_frame("0022500001", "regular"),
                "PlayIn":         _segment_frame("0052500001", "playin"),
            }.get(season_type_all_star, pd.DataFrame())

        def get_data_frames(self):
            return [self._df]

    import sys
    fake_submodule = type("FakeLGL", (), {"LeagueGameLog": _FailingPlayoffs})
    monkeypatch.setitem(
        sys.modules, "nba_api.stats.endpoints.leaguegamelog", fake_submodule,
    )
    monkeypatch.setattr(nba_stats, "_rate_limit", lambda: None)

    df = nba_stats.fetch_season_games("2025-26")
    # Regular Season + PlayIn rows survive; Playoffs failure is swallowed.
    assert len(df) == 4
    assert set(df["SEGMENT_LABEL"]) == {"regular", "playin"}


def test_fetch_season_games_dedupes_when_segments_overlap(monkeypatch):
    """Defensive: if NBA's API ever double-classifies a game we don't want
    duplicate rows feeding the sync loop. Dedupe key is (GAME_ID, TEAM_ID).
    """
    from nba_betting.data import nba_stats

    _StubGameLog.calls = []
    overlapping = _segment_frame("0022500001", "shared")
    _StubGameLog.payload = {
        "Regular Season": overlapping,
        "PlayIn":         overlapping,  # same game id, both teams — should dedupe
        "Playoffs":       pd.DataFrame(),
    }

    import sys
    fake_submodule = type("FakeLGL", (), {"LeagueGameLog": _StubGameLog})
    monkeypatch.setitem(
        sys.modules, "nba_api.stats.endpoints.leaguegamelog", fake_submodule,
    )
    monkeypatch.setattr(nba_stats, "_rate_limit", lambda: None)

    df = nba_stats.fetch_season_games("2025-26")
    # Two teams × one (deduped) game = 2 rows, not 4.
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Fix 2 — update_results' lenient ±1-day fallback
# ---------------------------------------------------------------------------


def _setup_isolated_state(tmp_path, monkeypatch):
    """Point DB_PATH and HISTORY_FILE at tmp paths and reload the affected
    modules. Mirrors the pattern in test_snapshot_jsonl.

    Returns (session_module, tracker_module, history_path).
    """
    from nba_betting import config as _cfg

    test_db = tmp_path / "t.sqlite"
    test_data = tmp_path / "data"
    test_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_cfg, "DB_PATH", str(test_db))
    monkeypatch.setattr(_cfg, "DATA_DIR", test_data)

    from nba_betting.db import session as _session
    importlib.reload(_session)
    from nba_betting.betting import tracker as _tracker
    importlib.reload(_tracker)

    return _session, _tracker, test_data / "prediction_history.json"


def _seed_team_pair(session_module):
    """Two teams referenced by the fixture records below."""
    from nba_betting.db.models import Team
    s = session_module.get_session()
    try:
        s.add(Team(id=1, abbreviation="AAA", name="Alphas"))
        s.add(Team(id=2, abbreviation="BBB", name="Betas"))
        s.commit()
    finally:
        s.close()


def _seed_game(session_module, *, game_id: str, when: date, home_id: int, away_id: int,
               home_score: int, away_score: int):
    from nba_betting.db.models import Game
    s = session_module.get_session()
    try:
        s.add(Game(
            id=game_id,
            date=when,
            season="2025-26",
            home_team_id=home_id,
            away_team_id=away_id,
            home_score=home_score,
            away_score=away_score,
            home_win=home_score > away_score,
        ))
        s.commit()
    finally:
        s.close()


def _write_unresolved_record(history_path: Path, *, record_date: str, home: str, away: str,
                              market_home_prob: float = 0.5, bet_side: str = "HOME",
                              bet_size: float = 10.0):
    record = {
        "date": record_date,
        "home_team": home,
        "away_team": away,
        "model_home_prob": 0.6,
        "market_home_prob": market_home_prob,
        "bet_side": bet_side,
        "edge": 0.05,
        "bet_size": bet_size,
        "model_type": "ensemble",
    }
    history_path.write_text(json.dumps([record], indent=2))


def test_update_results_resolves_legacy_record_under_local_date(tmp_path, monkeypatch):
    """The five 2026-04-10 records in our real history actually correspond
    to 2026-04-09 games (Vienna-evening filed under tomorrow-local). The
    fallback finds them at date-1 and resolves correctly.
    """
    session_module, tracker, history_path = _setup_isolated_state(tmp_path, monkeypatch)

    _seed_team_pair(session_module)
    _seed_game(
        session_module, game_id="g1", when=date(2026, 4, 9),
        home_id=1, away_id=2, home_score=110, away_score=100,
    )
    _write_unresolved_record(
        history_path, record_date="2026-04-10", home="AAA", away="BBB",
        bet_side="HOME", market_home_prob=0.5, bet_size=10.0,
    )

    n = tracker.update_results()
    assert n == 1, "the single unresolved legacy record should have resolved via the ±1 day fallback"

    resolved = json.loads(history_path.read_text())[0]
    assert resolved["home_won"] is True
    # Bet HOME at 0.5 → decimal odds 2.0 → win = +bet_size profit.
    assert resolved["profit"] == pytest.approx(10.0)


def test_update_results_skips_when_window_has_two_candidates(tmp_path, monkeypatch):
    """Safety: if both date-1 AND date+1 contain a same-matchup completed
    game, we can't tell which the user actually bet, so we leave the record
    unresolved rather than silently misattribute. (NBA in practice never
    schedules the same matchup on consecutive calendar days, but we still
    enforce the constraint.)
    """
    session_module, tracker, history_path = _setup_isolated_state(tmp_path, monkeypatch)

    _seed_team_pair(session_module)
    _seed_game(
        session_module, game_id="g1", when=date(2026, 4, 9),
        home_id=1, away_id=2, home_score=110, away_score=100,
    )
    _seed_game(
        session_module, game_id="g2", when=date(2026, 4, 11),
        home_id=1, away_id=2, home_score=95, away_score=105,
    )
    _write_unresolved_record(history_path, record_date="2026-04-10", home="AAA", away="BBB")

    n = tracker.update_results()
    assert n == 0, "ambiguous ±1 day window must NOT resolve — that risks misattribution"

    rec = json.loads(history_path.read_text())[0]
    assert rec.get("home_won") is None


def test_update_results_exact_date_still_preferred_over_fallback(tmp_path, monkeypatch):
    """Sanity: when the exact date matches AND a fallback candidate exists,
    the exact date wins. We should never fire the fallback when the primary
    query already has a row.
    """
    session_module, tracker, history_path = _setup_isolated_state(tmp_path, monkeypatch)

    _seed_team_pair(session_module)
    # Game on the exact date the record claims.
    _seed_game(
        session_module, game_id="g_exact", when=date(2026, 4, 10),
        home_id=1, away_id=2, home_score=110, away_score=100,
    )
    # Also a game one day off — fallback would otherwise pick this up.
    _seed_game(
        session_module, game_id="g_fallback", when=date(2026, 4, 9),
        home_id=1, away_id=2, home_score=80, away_score=120,  # different result
    )
    _write_unresolved_record(history_path, record_date="2026-04-10", home="AAA", away="BBB")

    n = tracker.update_results()
    assert n == 1
    resolved = json.loads(history_path.read_text())[0]
    # If the exact-date game (home win) is what resolved this record, profit
    # is +bet_size. If the fallback (home loss) leaked through, profit would
    # be -bet_size. This pins the correct precedence.
    assert resolved["home_won"] is True
    assert resolved["profit"] == pytest.approx(10.0)
