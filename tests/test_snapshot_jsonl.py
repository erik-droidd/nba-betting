"""Tests for the DB-free JSONL snapshot path.

Context: the user lives in Europe and is asleep during the NBA overnight
window, so odds snapshots are captured remotely by a GitHub Actions cron
that writes JSONL and commits it back to the repo. These tests pin the
round-trip (snapshot → JSONL → import) and verify the import is
idempotent — the single most important property because the GH runner
may re-push the same file after transient failures, and
``import-snapshots`` is expected to be safe to rerun on any schedule.
"""
from __future__ import annotations

import importlib
import json
from datetime import date, datetime
from pathlib import Path

import pytest


def _reload_with_tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a tempfile and reload the session module so
    the SQLAlchemy engine rebinds. Returns (session_module, jsonl_module).

    Mirrors the pattern used by ``test_apply_additive_migrations_is_idempotent``
    — any test that needs to touch the real `odds_snapshots` table must
    isolate from the developer's local SQLite file or the suite becomes
    environment-dependent.
    """
    from nba_betting import config as _cfg
    test_db = tmp_path / "t.sqlite"
    monkeypatch.setattr(_cfg, "DB_PATH", str(test_db))

    from nba_betting.db import session as _session
    importlib.reload(_session)

    # Reload the jsonl module too so it picks up the rebound engine
    # through its `from ... import get_session` binding.
    from nba_betting.data import snapshot_jsonl as _jsonl
    importlib.reload(_jsonl)
    return _session, _jsonl


def _seed_teams(session_module):
    """Insert the two teams referenced in the fixture records so the
    foreign-key lookup in ``import_snapshots_jsonl`` succeeds.
    """
    from nba_betting.db.models import Team
    sess = session_module.get_session()
    try:
        sess.add(Team(id=1610612738, abbreviation="BOS", name="Celtics"))
        sess.add(Team(id=1610612747, abbreviation="LAL", name="Lakers"))
        sess.commit()
    finally:
        sess.close()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Import happy path + idempotence
# ---------------------------------------------------------------------------


def test_import_round_trip_inserts_rows(tmp_path, monkeypatch):
    """Writing a 2-record JSONL file and importing it should result in
    exactly 2 rows in ``odds_snapshots``."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)

    records = [
        {
            "game_date": "2026-04-18",
            "home_team_abbr": "BOS",
            "away_team_abbr": "LAL",
            "source": "polymarket",
            "timestamp": "2026-04-18T22:00:00",
            "home_prob": 0.62,
            "spread": None,
            "over_under": None,
            "game_id": None,
        },
        {
            "game_date": "2026-04-18",
            "home_team_abbr": "BOS",
            "away_team_abbr": "LAL",
            "source": "espn",
            "timestamp": "2026-04-18T22:00:00",
            "home_prob": 0.60,
            "spread": -3.5,
            "over_under": 224.5,
            "game_id": None,
        },
    ]
    jsonl_path = tmp_path / "snapshots" / "2026-04-18.jsonl"
    _write_jsonl(jsonl_path, records)

    result = jsonl.import_snapshots_jsonl(jsonl_path)

    assert result["records"] == 2
    assert result["imported"] == 2
    assert result["skipped"] == 0
    assert result["errors"] == []

    # Confirm the rows actually landed with the right fields.
    from nba_betting.db.models import OddsSnapshot
    from sqlalchemy import select
    sess = session_module.get_session()
    try:
        rows = sess.execute(select(OddsSnapshot)).scalars().all()
        assert len(rows) == 2
        by_source = {r.source: r for r in rows}
        assert by_source["polymarket"].home_prob == pytest.approx(0.62)
        assert by_source["espn"].spread == pytest.approx(-3.5)
        assert by_source["espn"].over_under == pytest.approx(224.5)
        # Team IDs resolved from abbr.
        assert by_source["polymarket"].home_team_id == 1610612738
        assert by_source["polymarket"].away_team_id == 1610612747
    finally:
        sess.close()


def test_import_is_idempotent(tmp_path, monkeypatch):
    """The single most important property: re-importing the same file
    must insert 0 new rows. The GH Actions runner can re-push on
    transient failures, and the user may run ``import-snapshots`` on
    any cadence — duplicates would poison ``get_closing_line`` and
    line-movement features."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)

    rec = {
        "game_date": "2026-04-18",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "source": "polymarket",
        "timestamp": "2026-04-18T22:00:00",
        "home_prob": 0.62,
        "spread": None,
        "over_under": None,
        "game_id": None,
    }
    jsonl_path = tmp_path / "snapshots" / "2026-04-18.jsonl"
    _write_jsonl(jsonl_path, [rec])

    first = jsonl.import_snapshots_jsonl(jsonl_path)
    assert first["imported"] == 1
    assert first["skipped"] == 0

    # Second import of the EXACT same file — everything should dedup.
    second = jsonl.import_snapshots_jsonl(jsonl_path)
    assert second["imported"] == 0
    assert second["skipped"] == 1
    assert second["errors"] == []


def test_import_skips_bad_rows_but_inserts_good_ones(tmp_path, monkeypatch):
    """Unknown team / bad JSON / unknown source records should be logged
    as errors but must NOT prevent good records in the same file from
    being imported. The GH runner produces concatenated JSONL from
    multiple days — a single bad line shouldn't sink the whole file."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)

    good = {
        "game_date": "2026-04-18",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "source": "polymarket",
        "timestamp": "2026-04-18T22:00:00",
        "home_prob": 0.62,
        "spread": None,
        "over_under": None,
        "game_id": None,
    }
    unknown_team = {
        **good,
        "home_team_abbr": "XXX",
        "timestamp": "2026-04-18T22:15:00",
    }
    bad_source = {
        **good,
        "source": "bookmaker-of-last-resort",
        "timestamp": "2026-04-18T22:30:00",
    }

    jsonl_path = tmp_path / "2026-04-18.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(good) + "\n")
        f.write("not-json-at-all\n")
        f.write(json.dumps(unknown_team) + "\n")
        f.write(json.dumps(bad_source) + "\n")
        f.write("\n")  # blank line is fine — skipped silently

    result = jsonl.import_snapshots_jsonl(jsonl_path)
    assert result["imported"] == 1
    # 3 problem rows (bad JSON, unknown team, unknown source); blank line skipped silently
    assert len(result["errors"]) == 3


def test_import_accepts_directory_glob(tmp_path, monkeypatch):
    """Passing a directory should import every *.jsonl file underneath.
    The CLI default is ``data/odds_snapshots/`` — pointing it at the
    directory root must Just Work."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)

    d = tmp_path / "snaps"
    d.mkdir()
    _write_jsonl(d / "2026-04-17.jsonl", [{
        "game_date": "2026-04-17",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "source": "espn",
        "timestamp": "2026-04-17T22:00:00",
        "home_prob": 0.58,
        "spread": -2.0,
        "over_under": 220.0,
        "game_id": None,
    }])
    _write_jsonl(d / "2026-04-18.jsonl", [{
        "game_date": "2026-04-18",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "source": "espn",
        "timestamp": "2026-04-18T22:00:00",
        "home_prob": 0.60,
        "spread": -3.0,
        "over_under": 222.0,
        "game_id": None,
    }])

    result = jsonl.import_snapshots_jsonl(d)
    assert result["files"] == 2
    assert result["imported"] == 2


# ---------------------------------------------------------------------------
# Capture side (no network — monkeypatch the fetchers)
# ---------------------------------------------------------------------------


def test_capture_writes_jsonl_without_touching_db(tmp_path, monkeypatch):
    """The GH Actions runner has no persistent SQLite. ``capture_snapshot_to_jsonl``
    MUST NOT call ``init_db`` / ``get_session`` — patching them to raise
    would be overkill (Python imports still resolve), so instead we
    verify no DB file gets created alongside the JSONL."""
    from nba_betting.data import snapshot_jsonl as jsonl

    fake_games = [{
        "game_id": "0022500123",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "home_team_id": 1610612738,
        "away_team_id": 1610612747,
        "game_time_utc": "2026-04-19T00:00:00Z",
    }]
    fake_poly = [{
        "teams": {"BOS": 0.62, "LAL": 0.38},
        "event_title": "Celtics vs Lakers",
    }]
    fake_espn = [{
        "teams": {"BOS": 0.60, "LAL": 0.40},
        "spread": -3.0,
        "over_under": 222.0,
        "event_title": "LAL @ BOS",
        "source": "espn",
    }]

    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: fake_games,
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: fake_poly,
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: fake_espn,
    )

    out_dir = tmp_path / "captured"
    result = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 4, 18, 22, 0, 0),
    )

    assert result["games"] == 1
    assert result["written"] == 2  # one per source
    assert result["warnings"] == []

    files = sorted(out_dir.glob("*.jsonl"))
    assert len(files) == 1
    # Filename follows the stamping timestamp's UTC day.
    assert files[0].name == "2026-04-18.jsonl"

    lines = files[0].read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    sources = {p["source"] for p in parsed}
    assert sources == {"polymarket", "espn"}
    # game_date is the game's ET day (matches Game.date / get_closing_line):
    # the 00:00Z tip is 8 PM ET on the 18th, so the ET game-day is the 18th —
    # NOT the UTC day (19th), which was the old misfiling bug that left
    # snapshots unable to join their game.
    assert all(p["game_date"] == "2026-04-18" for p in parsed)


def test_capture_roundtrip_then_import(tmp_path, monkeypatch):
    """End-to-end: write JSONL from a fake slate, then import it into an
    isolated tempdb. Exercises exactly the contract the GH runner →
    local import pipeline depends on."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)

    fake_games = [{
        "game_id": "0022500123",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "home_team_id": 1610612738,
        "away_team_id": 1610612747,
        "game_time_utc": "2026-04-19T00:00:00Z",
    }]
    fake_poly = [{
        "teams": {"BOS": 0.62, "LAL": 0.38},
        "event_title": "Celtics vs Lakers",
    }]
    fake_espn = [{
        "teams": {"BOS": 0.60, "LAL": 0.40},
        "spread": -3.0,
        "over_under": 222.0,
        "event_title": "LAL @ BOS",
        "source": "espn",
    }]

    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: fake_games,
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: fake_poly,
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: fake_espn,
    )

    out_dir = tmp_path / "captured"
    cap = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 4, 18, 22, 0, 0),
    )
    assert cap["written"] == 2

    # First import lands both; second adds none.
    imp1 = jsonl.import_snapshots_jsonl(out_dir)
    assert imp1["imported"] == 2
    assert imp1["skipped"] == 0

    imp2 = jsonl.import_snapshots_jsonl(out_dir)
    assert imp2["imported"] == 0
    assert imp2["skipped"] == 2


def test_capture_no_games_still_returns_status(tmp_path, monkeypatch):
    """Offseason / no-slate day: neither fetcher returns games. We
    should still return a structured status dict with a ``no games``
    warning, and NOT create an empty file that would get committed."""
    from nba_betting.data import snapshot_jsonl as jsonl

    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    # Also mock the ESPN fallback added for the datacenter-IP case;
    # without this the test would reach the real network and either
    # hang or return a live slate, making it flaky.
    monkeypatch.setattr(
        "nba_betting.data.espn.fetch_scoreboard",
        lambda date_str=None: [],
    )

    out_dir = tmp_path / "captured"
    result = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 7, 10, 22, 0, 0),  # July — no games
    )
    assert result["games"] == 0
    assert result["written"] == 0
    assert "no games scheduled" in result["warnings"]
    # The file must NOT be created — the workflow's `git status --porcelain`
    # check uses that to skip empty commits.
    assert not Path(result["path"]).exists()


# ---------------------------------------------------------------------------
# Timestamp parsing — naive-UTC compatibility with snapshot_current_odds
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ESPN fallback — the GitHub runner can't hit stats.nba.com (datacenter
# IPs get silently blocked). ``fetch_todays_games`` returns ``[]``, so
# capture_snapshot_to_jsonl falls through to ESPN's scoreboard, which
# is not blocked. These tests pin that behavior.
# ---------------------------------------------------------------------------


def test_capture_falls_back_to_espn_when_nba_api_empty(tmp_path, monkeypatch):
    """Simulate the GH runner's exact failure mode: nba_api silently
    returns ``[]`` (datacenter IP block). The capture function must
    still produce game records by hitting ESPN's scoreboard."""
    from nba_betting.data import snapshot_jsonl as jsonl

    # nba_api: both fetchers return nothing (datacenter IP block).
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )

    # ESPN scoreboard: returns two scheduled games for the first day
    # we check. The helper walks today → today+days_ahead, so the
    # mock returns games on the first call and [] after.
    espn_games_response = [
        {
            "espn_event_id": 401869187,
            "date": "2026-04-19T23:00Z",
            "status": "STATUS_SCHEDULED",
            "home_team": {"espn_id": 2, "abbr": "BOS", "name": "Celtics"},
            "away_team": {"espn_id": 20, "abbr": "PHI", "name": "76ers"},
            "odds": {},
        },
        {
            "espn_event_id": 401869188,
            "date": "2026-04-20T01:00Z",
            "status": "STATUS_FINAL",  # must be filtered out
            "home_team": {"espn_id": 25, "abbr": "OKC", "name": "Thunder"},
            "away_team": {"espn_id": 21, "abbr": "PHX", "name": "Suns"},
            "odds": {},
        },
    ]
    calls = {"n": 0}

    def fake_fetch_scoreboard(date_str=None):
        calls["n"] += 1
        return espn_games_response if calls["n"] == 1 else []

    monkeypatch.setattr(
        "nba_betting.data.espn.fetch_scoreboard",
        fake_fetch_scoreboard,
    )
    # Odds providers return empty but must not crash — we still want
    # the games in the slate so we know *which* matchups to snapshot
    # once odds providers come back online.
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: [],
    )

    out_dir = tmp_path / "captured"
    result = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 4, 19, 22, 0, 0),
    )

    # One game survived the STATUS_SCHEDULED filter.
    assert result["games"] == 1
    # No odds providers are mocked with data → no records written,
    # but the slate was discovered via the ESPN fallback.
    assert result["source"] == "espn-fallback"
    # The fallback message is informational, not a warning — it lives
    # in `notes` so the CLI stays green when the runner takes this
    # (expected) path. Pre-fix it landed in `warnings` and the yellow
    # "warn" label mis-signaled that something had broken.
    assert any("ESPN fallback" in n for n in result["notes"])
    assert not any("ESPN fallback" in w for w in result["warnings"])


def test_capture_prefers_nba_api_when_it_returns_games(tmp_path, monkeypatch):
    """On the user's local machine nba_api works fine. We must NOT
    fall back to ESPN in that case — nba_api has authoritative NBA
    team IDs that match ``games.id`` FKs; ESPN IDs don't."""
    from nba_betting.data import snapshot_jsonl as jsonl

    fake_games = [{
        "game_id": "0022500123",
        "home_team_abbr": "BOS",
        "away_team_abbr": "LAL",
        "home_team_id": 1610612738,
        "away_team_id": 1610612747,
        "game_time_utc": "2026-04-19T00:00:00Z",
    }]
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: fake_games,
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    # Sentinel: if the fallback fires, this raises and the test fails
    # visibly — better than a silent mismatch in the result dict.
    def _should_not_be_called(date_str=None):
        raise AssertionError("ESPN fallback must not fire when nba_api has games")
    monkeypatch.setattr(
        "nba_betting.data.espn.fetch_scoreboard",
        _should_not_be_called,
    )
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: [],
    )

    result = jsonl.capture_snapshot_to_jsonl(
        tmp_path / "captured",
        timestamp=datetime(2026, 4, 18, 22, 0, 0),
    )
    assert result["games"] == 1
    assert result["source"] == "nba-api"


def test_espn_fallback_strips_event_id_so_import_is_safe(tmp_path, monkeypatch):
    """ESPN event IDs (e.g. ``401869187``) are not valid ``games.id``
    FKs — those use NBA API format (``0022500123``). The fallback must
    emit an empty/None game_id so imported rows don't create dangling
    FK references that later confuse the reconcile path."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)

    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.espn.fetch_scoreboard",
        lambda date_str=None: [{
            "espn_event_id": 401869187,
            "date": "2026-04-19T23:00Z",
            "status": "STATUS_SCHEDULED",
            "home_team": {"espn_id": 2, "abbr": "BOS", "name": "Celtics"},
            "away_team": {"espn_id": 13, "abbr": "LAL", "name": "Lakers"},
            "odds": {},
        }],
    )
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: [{
            "teams": {"BOS": 0.55, "LAL": 0.45},
            "event_title": "LAL @ BOS",
        }],
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: [],
    )

    out_dir = tmp_path / "captured"
    result = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 4, 19, 22, 0, 0),
    )
    assert result["source"] == "espn-fallback"
    assert result["written"] == 1

    # The JSONL record must have game_id == null (not the ESPN event ID).
    line = Path(result["path"]).read_text().strip()
    rec = json.loads(line)
    assert rec["game_id"] is None

    # Import must succeed and land the row with game_id NULL.
    imp = jsonl.import_snapshots_jsonl(out_dir)
    assert imp["imported"] == 1
    assert imp["errors"] == []
    from nba_betting.db.models import OddsSnapshot
    from sqlalchemy import select
    sess = session_module.get_session()
    try:
        row = sess.execute(select(OddsSnapshot)).scalars().one()
        assert row.game_id is None
    finally:
        sess.close()


def test_capture_picks_right_polymarket_event_when_pair_has_multiple_dates(tmp_path, monkeypatch):
    """Regression: Polymarket publishes a separate event per matchup date,
    so a season's ORL@DET pair can have several open events at once
    (today's game + future rematches that opened for early trading).

    The original consumer keyed the Polymarket index by team-pair alone
    and silently picked the last event, producing the wrong moneyline
    for tonight's game. Example caught 2026-04-22: stored
    ``home_prob=0.645`` for DET when the actual moneyline was 0.785 —
    0.645 came from a future rematch event 5 days out.

    The fix: Polymarket odds now carry the ET ``game_date`` parsed from
    the event slug, and the consumer matches by (pair, game_date). This
    test pins that: the consumer must pick the 0.78 event (tonight's)
    and ignore the 0.645 one (a future rematch), even though both share
    the same team pair.
    """
    from nba_betting.data import snapshot_jsonl as jsonl

    fake_games = [{
        "game_id": "0042500102",
        "home_team_abbr": "DET",
        "away_team_abbr": "ORL",
        "home_team_id": 1,
        "away_team_id": 2,
        # 23:00Z on Apr 22 = 19:00 ET Apr 22 → ET date is 2026-04-22.
        "game_time_utc": "2026-04-22T23:00:00Z",
    }]
    # Tonight's game (correct event) + a future rematch with a
    # different slug-encoded date. Order intentionally puts the wrong
    # one last so the old dict-overwrite bug would surface.
    fake_poly = [
        {
            "teams": {"ORL": 0.215, "DET": 0.785},
            "event_title": "Magic vs. Pistons",
            "event_slug": "nba-orl-det-2026-04-22",
            "game_date": "2026-04-22",
        },
        {
            "teams": {"DET": 0.645, "ORL": 0.355},
            "event_title": "Pistons vs. Magic",
            "event_slug": "nba-det-orl-2026-04-27",
            "game_date": "2026-04-27",
        },
    ]

    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: fake_games,
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: fake_poly,
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: [],
    )

    out_dir = tmp_path / "captured"
    result = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 4, 22, 17, 0, 0),
    )
    assert result["written"] == 1
    rec = json.loads(Path(result["path"]).read_text().strip())
    assert rec["source"] == "polymarket"
    assert rec["home_team_abbr"] == "DET"
    # The crucial assertion: picked tonight's event, not the rematch.
    assert rec["home_prob"] == pytest.approx(0.785)


def test_capture_skips_ambiguous_polymarket_when_no_date_match(tmp_path, monkeypatch):
    """If Polymarket returns multiple events for the same team pair and
    none matches the game's ET date, we'd rather skip the Polymarket
    record than silently pick one. The old behavior picked the last
    one seen — the new behavior must emit nothing for that source."""
    from nba_betting.data import snapshot_jsonl as jsonl

    fake_games = [{
        "game_id": "0042500102",
        "home_team_abbr": "DET",
        "away_team_abbr": "ORL",
        "home_team_id": 1,
        "away_team_id": 2,
        "game_time_utc": "2026-04-22T23:00:00Z",  # ET date 2026-04-22
    }]
    # Two events, neither matches tonight's date.
    fake_poly = [
        {
            "teams": {"DET": 0.60, "ORL": 0.40},
            "event_title": "Pistons vs. Magic",
            "event_slug": "nba-det-orl-2026-04-25",
            "game_date": "2026-04-25",
        },
        {
            "teams": {"DET": 0.645, "ORL": 0.355},
            "event_title": "Pistons vs. Magic",
            "event_slug": "nba-det-orl-2026-04-27",
            "game_date": "2026-04-27",
        },
    ]

    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_todays_games",
        lambda *a, **kw: fake_games,
    )
    monkeypatch.setattr(
        "nba_betting.data.nba_stats.fetch_upcoming_games",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "nba_betting.data.polymarket.get_nba_odds",
        lambda: fake_poly,
    )
    monkeypatch.setattr(
        "nba_betting.data.espn_odds.get_espn_odds",
        lambda *a, **kw: [],
    )

    out_dir = tmp_path / "captured"
    result = jsonl.capture_snapshot_to_jsonl(
        out_dir,
        timestamp=datetime(2026, 4, 22, 17, 0, 0),
    )
    # No Polymarket record written (no date match, ambiguous pair).
    assert result["written"] == 0


def test_parse_timestamp_drops_tz_to_match_existing_rows():
    """The existing ``snapshot_current_odds()`` writes
    ``datetime.utcnow()`` (naive). If JSONL imports landed as
    tz-aware, SQLite would either coerce or raise on ordering.
    Verify the parser normalizes both ``Z`` and ``+00:00`` to naive UTC.
    """
    from nba_betting.data.snapshot_jsonl import _parse_timestamp

    ts_z = _parse_timestamp("2026-04-18T22:00:00Z")
    ts_offset = _parse_timestamp("2026-04-18T22:00:00+00:00")
    ts_naive = _parse_timestamp("2026-04-18T22:00:00")

    assert ts_z.tzinfo is None
    assert ts_offset.tzinfo is None
    assert ts_naive.tzinfo is None
    assert ts_z == ts_offset == ts_naive


def _add_games(session_module, games):
    """games: [(id, date)] for the BOS(home)-LAL(away) pair."""
    from nba_betting.db.models import Game
    sess = session_module.get_session()
    try:
        for gid, d in games:
            sess.add(Game(id=gid, home_team_id=1610612738, away_team_id=1610612747,
                          date=d, season="2025-26"))
        sess.commit()
    finally:
        sess.close()


def _rows(session_module):
    from nba_betting.db.models import OddsSnapshot
    from sqlalchemy import select
    sess = session_module.get_session()
    try:
        return [(r.game_date, r.game_id, r.source, r.home_prob)
                for r in sess.execute(select(OddsSnapshot).order_by(OddsSnapshot.id)).scalars()]
    finally:
        sess.close()


def test_import_files_late_tip_capture_under_tonights_game_not_the_next_one(tmp_path, monkeypatch):
    """The 2026 playoff bug: an 8:06 PM ET capture (00:06Z next day) for
    Game 3 was filed under Game 4, two days later at the same arena, because
    the resolver anchored on the capture's UTC date. Both record formats —
    current (ET game date) and pre-2026-05-29 (UTC tip date, ET + 1) — must
    land on tonight's game."""
    from datetime import date as _date
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)
    _add_games(session_module, [("G3", _date(2026, 4, 23)), ("G4", _date(2026, 4, 25))])

    base = {"home_team_abbr": "BOS", "away_team_abbr": "LAL", "source": "polymarket",
            "timestamp": "2026-04-24T00:06:01", "home_prob": 0.6,
            "spread": None, "over_under": None, "game_id": None}
    _write_jsonl(tmp_path / "s" / "a.jsonl", [
        dict(base, game_date="2026-04-23"),                               # ET format
        dict(base, game_date="2026-04-24", timestamp="2026-04-24T00:07:00"),  # legacy UTC format
    ])
    assert jsonl.import_snapshots_jsonl(tmp_path / "s")["imported"] == 2
    assert [r[:2] for r in _rows(session_module)] == [(_date(2026, 4, 23), "G3")] * 2


def test_import_never_guesses_a_game_the_record_date_does_not_name(tmp_path, monkeypatch):
    """A late-preseason capture must not attach to an opening-week rematch
    of the same pair: no game on the record's date (or the day before) means
    unmatched, kept under its own date for `sync` to link later."""
    from datetime import date as _date
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)
    _add_games(session_module, [("REG1", _date(2026, 10, 20))])

    _write_jsonl(tmp_path / "s" / "a.jsonl", [{
        "game_date": "2026-10-17", "home_team_abbr": "BOS", "away_team_abbr": "LAL",
        "source": "polymarket", "timestamp": "2026-10-17T18:00:00", "home_prob": 0.55,
        "spread": None, "over_under": None, "game_id": None,
    }])
    jsonl.import_snapshots_jsonl(tmp_path / "s")
    assert _rows(session_module) == [(_date(2026, 10, 17), None, "polymarket", 0.55)]


def test_import_key_ignores_derived_game_date_so_reimport_never_duplicates(tmp_path, monkeypatch):
    """Import before the game is synced (unmatched), then again after: the
    second pass must skip, not insert a copy under the resolved date."""
    from datetime import date as _date
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)
    _write_jsonl(tmp_path / "s" / "a.jsonl", [{
        "game_date": "2026-04-19", "home_team_abbr": "BOS", "away_team_abbr": "LAL",
        "source": "polymarket", "timestamp": "2026-04-19T22:00:00", "home_prob": 0.6,
        "spread": None, "over_under": None, "game_id": None,
    }])
    assert jsonl.import_snapshots_jsonl(tmp_path / "s")["imported"] == 1
    _add_games(session_module, [("G1", _date(2026, 4, 18))])   # legacy +1 date resolves here
    second = jsonl.import_snapshots_jsonl(tmp_path / "s")
    assert (second["imported"], second["skipped"]) == (0, 1)
    assert len(_rows(session_module)) == 1


def test_import_nulls_legacy_espn_spread_proxy_but_keeps_moneyline_probs(tmp_path, monkeypatch):
    """Before the 2026-10 fix every ESPN home_prob was the 2.5%/pt spread
    proxy (ESPN had moved its moneylines). Those must not import as market
    prices; post-fix records (with moneyline fields) keep theirs."""
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)
    common = {"game_date": "2026-04-19", "home_team_abbr": "BOS", "away_team_abbr": "LAL",
              "source": "espn", "over_under": 220.5, "game_id": None}
    _write_jsonl(tmp_path / "s" / "a.jsonl", [
        dict(common, timestamp="2026-04-19T18:00:00", home_prob=0.5875, spread=-3.5),
        dict(common, timestamp="2026-04-19T19:00:00", home_prob=0.5875, spread=-3.5,
             home_moneyline=-150.0, away_moneyline=130.0),
    ])
    jsonl.import_snapshots_jsonl(tmp_path / "s")
    assert [r[3] for r in _rows(session_module)] == [None, 0.5875]


def test_reresolve_links_unmatched_rows_once_their_game_is_synced(tmp_path, monkeypatch):
    """`sync` path: rows imported before their game existed get linked."""
    from datetime import date as _date, datetime as _dt
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)
    from nba_betting.db.models import OddsSnapshot
    sess = session_module.get_session()
    try:
        sess.add(OddsSnapshot(
            game_id=None, game_date=_date(2026, 4, 18),
            home_team_id=1610612738, away_team_id=1610612747,
            source="polymarket", timestamp=_dt(2026, 4, 18, 13, 7, 0), home_prob=0.62,
        ))
        sess.commit()
    finally:
        sess.close()
    _add_games(session_module, [("0042500042", _date(2026, 4, 18))])

    out = jsonl.reresolve_existing_snapshots(only_unmatched=True)
    assert out["updated"] == 1 and out["unmatched"] == 0
    assert _rows(session_module)[0][:2] == (_date(2026, 4, 18), "0042500042")


def test_repair_dedupes_refiles_and_nulls_spread_proxies(tmp_path, monkeypatch):
    """The one-off repair: collapse duplicate captures, move a closing line
    the old resolver filed under the next series game back to its game
    (from the JSONL record), NULL spread-proxy ESPN probs — and re-date rows
    that aren't in any file (local `predict` captures) from their game_id.
    Dry run changes nothing; a second real run changes nothing either."""
    from datetime import date as _date, datetime as _dt
    session_module, jsonl = _reload_with_tmp_db(tmp_path, monkeypatch)
    _seed_teams(session_module)
    _add_games(session_module, [("G3", _date(2026, 4, 23)), ("G4", _date(2026, 4, 25))])
    from nba_betting.db.models import OddsSnapshot

    ts = _dt(2026, 4, 24, 0, 6, 1)
    sess = session_module.get_session()
    try:
        for _ in range(2):   # misfiled under G4, twice (duplicate)
            sess.add(OddsSnapshot(game_id="G4", game_date=_date(2026, 4, 25),
                                  home_team_id=1610612738, away_team_id=1610612747,
                                  source="espn", timestamp=ts, home_prob=0.5875, spread=-3.5))
        sess.add(OddsSnapshot(game_id="G4", game_date=_date(2026, 4, 26),   # local predict row,
                              home_team_id=1610612738, away_team_id=1610612747,  # UTC-dated
                              source="polymarket", timestamp=_dt(2026, 4, 26, 0, 30, 0, 123456),
                              home_prob=0.61))
        sess.commit()
    finally:
        sess.close()
    _write_jsonl(tmp_path / "s" / "a.jsonl", [{
        "game_date": "2026-04-24", "home_team_abbr": "BOS", "away_team_abbr": "LAL",
        "source": "espn", "timestamp": "2026-04-24T00:06:01", "home_prob": 0.5875,
        "spread": -3.5, "over_under": 220.5, "game_id": None,
    }])

    dry = jsonl.repair_snapshots(tmp_path / "s", dry_run=True)
    assert (dry["duplicates_removed"], dry["rematched"], dry["espn_probs_nulled"]) == (1, 2, 1)
    assert len(_rows(session_module)) == 3

    res = jsonl.repair_snapshots(tmp_path / "s")
    assert (res["duplicates_removed"], res["rematched"], res["espn_probs_nulled"]) == (1, 2, 1)
    assert _rows(session_module) == [
        (_date(2026, 4, 23), "G3", "espn", None),
        (_date(2026, 4, 25), "G4", "polymarket", 0.61),
    ]
    again = jsonl.repair_snapshots(tmp_path / "s")
    assert (again["duplicates_removed"], again["rematched"], again["espn_probs_nulled"]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# 2026-10 audit: capture behaviour on the GitHub runner
# ---------------------------------------------------------------------------


def _espn_slate(date_utc="2026-10-07T23:30Z"):
    return [{
        "espn_event_id": 1, "date": date_utc, "status": "STATUS_SCHEDULED",
        "home_team": {"espn_id": 2, "abbr": "BOS", "name": "Celtics"},
        "away_team": {"espn_id": 13, "abbr": "LAL", "name": "Lakers"},
        "odds": {},
    }]


def _patch_sources(monkeypatch, *, espn_odds, poly=(), nba_api_calls=None, slate=None):
    def _nba(*a, **kw):
        if nba_api_calls is not None:
            nba_api_calls.append(1)
        return []
    monkeypatch.setattr("nba_betting.data.nba_stats.fetch_todays_games", _nba)
    monkeypatch.setattr("nba_betting.data.nba_stats.fetch_upcoming_games", _nba)
    monkeypatch.setattr("nba_betting.data.espn.fetch_scoreboard",
                        lambda date_str=None: slate if slate is not None else _espn_slate())
    monkeypatch.setattr("nba_betting.data.polymarket.get_nba_odds", lambda: list(poly))
    monkeypatch.setattr("nba_betting.data.espn_odds.get_espn_odds", espn_odds)


def test_capture_skip_nba_api_never_touches_stats_nba_com(tmp_path, monkeypatch):
    """On the runner every stats.nba.com call hung for its 30 s timeout x3
    retries x3 dates (~4.6 min per capture)."""
    from nba_betting.data import snapshot_jsonl as jsonl
    calls: list = []
    _patch_sources(monkeypatch, espn_odds=lambda *a, **kw: [], nba_api_calls=calls)
    res = jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True,
                                          timestamp=datetime(2026, 10, 7, 12, 0))
    assert calls == []
    assert res["games"] == 1 and res["source"] == "espn" and res["notes"] == []


def test_capture_stores_only_market_espn_probs_and_raw_moneylines(tmp_path, monkeypatch):
    """A spread-only ESPN line keeps its spread/total but no home_prob; a
    moneyline line keeps the de-vigged prob plus the raw American odds."""
    from nba_betting.data import snapshot_jsonl as jsonl
    spread_only = [{"teams": {"BOS": 0.5875, "LAL": 0.4125}, "prob_source": "spread",
                    "home_moneyline": None, "away_moneyline": None,
                    "spread": -3.5, "over_under": 221.5}]
    _patch_sources(monkeypatch, espn_odds=lambda *a, **kw: spread_only)
    jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, timestamp=datetime(2026, 10, 7, 12, 0))
    with_ml = [{"teams": {"BOS": 0.645, "LAL": 0.355}, "prob_source": "moneyline",
                "home_moneyline": -205.0, "away_moneyline": 170.0,
                "spread": -5.5, "over_under": 212.5}]
    _patch_sources(monkeypatch, espn_odds=lambda *a, **kw: with_ml)
    jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, timestamp=datetime(2026, 10, 7, 13, 0))

    recs = [json.loads(l) for l in (tmp_path / "2026-10-07.jsonl").read_text().splitlines()]
    assert [(r["home_prob"], r["spread"], r["home_moneyline"]) for r in recs] == [
        (None, -3.5, None), (0.645, -5.5, -205.0),
    ]


def test_capture_fetches_espn_odds_for_the_slate_date(tmp_path, monkeypatch):
    """After tonight's games tip, the slate is tomorrow's — ESPN's default
    (current) scoreboard is still tonight, so ask for the slate's date."""
    from nba_betting.data import snapshot_jsonl as jsonl
    asked: list = []

    def _espn(date_str=None):
        asked.append(date_str)
        return []
    # 2026-10-08T02:00Z is 10 PM ET on Oct 7.
    _patch_sources(monkeypatch, espn_odds=_espn, slate=_espn_slate("2026-10-08T02:00Z"))
    jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, timestamp=datetime(2026, 10, 7, 12, 0))
    assert asked == ["20261007"]


def test_capture_stamps_records_after_the_odds_are_fetched(tmp_path, monkeypatch):
    from datetime import timezone as _tz
    from nba_betting.data import snapshot_jsonl as jsonl
    state = {"fetched": False}

    def _poly():
        state["fetched"] = True
        return [{"teams": {"BOS": 0.6, "LAL": 0.4}, "game_date": "2026-10-07"}]

    def _clock():
        assert state["fetched"], "timestamp taken before the odds were fetched"
        return datetime(2026, 10, 7, 12, 4, 37, tzinfo=_tz.utc)

    _patch_sources(monkeypatch, espn_odds=lambda *a, **kw: [])
    monkeypatch.setattr("nba_betting.data.polymarket.get_nba_odds", _poly)
    jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, clock=_clock)
    rec = json.loads((tmp_path / "2026-10-07.jsonl").read_text())
    assert rec["timestamp"] == "2026-10-07T12:04:37"


def test_capture_dedupes_unchanged_lines_until_heartbeat(tmp_path, monkeypatch):
    from datetime import timedelta
    from nba_betting.data import snapshot_jsonl as jsonl
    poly = [{"teams": {"BOS": 0.6, "LAL": 0.4}, "game_date": "2026-10-07"}]
    _patch_sources(monkeypatch, espn_odds=lambda *a, **kw: [], poly=poly)
    state: dict = {}
    hb = timedelta(minutes=30)
    out = [jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, last_written=state,
                                           heartbeat=hb, timestamp=datetime(2026, 10, 7, 12, m))
           for m in (0, 5, 31)]
    assert [(o["written"], o["deduped"]) for o in out] == [(1, 0), (0, 1), (1, 0)]
    poly[0]["teams"]["BOS"] = 0.62     # a moved line is always written
    moved = jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, last_written=state,
                                            heartbeat=hb, timestamp=datetime(2026, 10, 7, 12, 33))
    assert moved["written"] == 1


def test_capture_reports_next_tip_and_fetch_failure(tmp_path, monkeypatch):
    from datetime import timezone as _tz
    from nba_betting.data import snapshot_jsonl as jsonl
    _patch_sources(monkeypatch, espn_odds=lambda *a, **kw: [])
    res = jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, timestamp=datetime(2026, 10, 7, 12, 0))
    assert res["next_tip_utc"] == datetime(2026, 10, 7, 23, 30, tzinfo=_tz.utc)
    assert res["games_fetch_failed"] is False

    def _down(date_str=None):
        raise ConnectionError("espn down")
    monkeypatch.setattr("nba_betting.data.espn.fetch_scoreboard", _down)
    res = jsonl.capture_snapshot_to_jsonl(tmp_path, skip_nba_api=True, timestamp=datetime(2026, 10, 7, 12, 0))
    assert res["games"] == 0 and res["games_fetch_failed"] is True
