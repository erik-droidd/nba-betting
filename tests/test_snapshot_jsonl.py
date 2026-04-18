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
        lambda: fake_espn,
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
    # Game_date should follow the game's UTC-day, which is the next day
    # (19th) since the tipoff is at 00:00Z on the 19th.
    assert all(p["game_date"] == "2026-04-19" for p in parsed)


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
        lambda: fake_espn,
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
