"""Daily injury snapshots via JSONL (2026-09): DB-free capture on the GH
runner, idempotent per-day import into ``historical_injuries``."""
from __future__ import annotations

import importlib
import json
from datetime import date, datetime, timezone

import pytest

from nba_betting.data.injuries import PlayerInjury


def _inj(name, team, status="Out", impact=7.0, pid="1"):
    return PlayerInjury(player_name=name, team_abbr=team, status=status,
                        reason="knee", impact_rating=impact, player_id=pid)


def _reload_with_tmp_db(tmp_path, monkeypatch):
    from nba_betting import config as _cfg
    monkeypatch.setattr(_cfg, "DB_PATH", str(tmp_path / "t.sqlite"))
    from nba_betting.db import session as _session
    importlib.reload(_session)
    from nba_betting.data import injury_jsonl as _mod
    importlib.reload(_mod)
    return _session, _mod


# ---------------------------------------------------------------- capture

def test_capture_files_under_et_date_and_sorts_lines(tmp_path):
    from nba_betting.data.injury_jsonl import capture_injuries_to_jsonl

    # 01:30Z on the 19th is 9:30 PM ET on the 18th -> file is 2026-04-18.
    ts = datetime(2026, 4, 19, 1, 30, tzinfo=timezone.utc)
    injs = [_inj("Zed Zulu", "LAL", pid="9"), _inj("Amy Adams", "BOS", "Questionable", 4.0, "2")]
    res = capture_injuries_to_jsonl(tmp_path, timestamp=ts, injuries=injs)

    assert res["snapshot_date"] == "2026-04-18"
    assert res["written"] == 2 and res["unchanged"] is False and not res["warnings"]
    path = tmp_path / "2026-04-18.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["team_abbr"] for l in lines] == ["BOS", "LAL"]      # sorted by team, name
    assert lines[0] == {
        "snapshot_date": "2026-04-18", "captured_at": "2026-04-19T01:30:00Z",
        "player_name": "Amy Adams", "player_id": "2", "team_abbr": "BOS",
        "status": "Questionable", "reason": "knee", "impact_rating": 4.0,
    }


def test_capture_skips_rewrite_when_list_unchanged(tmp_path):
    from nba_betting.data.injury_jsonl import capture_injuries_to_jsonl

    injs = [_inj("Amy Adams", "BOS")]
    t1 = datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 18, 18, 0, tzinfo=timezone.utc)
    capture_injuries_to_jsonl(tmp_path, timestamp=t1, injuries=injs)
    before = (tmp_path / "2026-04-18.jsonl").read_text()

    res = capture_injuries_to_jsonl(tmp_path, timestamp=t2, injuries=injs)
    assert res["unchanged"] is True and res["written"] == 0
    assert (tmp_path / "2026-04-18.jsonl").read_text() == before   # captured_at kept

    # A status change IS rewritten, with the new capture time.
    res = capture_injuries_to_jsonl(
        tmp_path, timestamp=t2, injuries=[_inj("Amy Adams", "BOS", "Probable", 1.0)])
    assert res["written"] == 1
    rec = json.loads((tmp_path / "2026-04-18.jsonl").read_text())
    assert rec["status"] == "Probable" and rec["captured_at"] == "2026-04-18T18:00:00Z"


def test_capture_empty_list_leaves_file_and_warns(tmp_path):
    from nba_betting.data.injury_jsonl import capture_injuries_to_jsonl

    ts = datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc)
    capture_injuries_to_jsonl(tmp_path, timestamp=ts, injuries=[_inj("Amy Adams", "BOS")])
    res = capture_injuries_to_jsonl(tmp_path, timestamp=ts, injuries=[])
    assert res["written"] == 0 and res["warnings"]
    assert (tmp_path / "2026-04-18.jsonl").exists()


# ----------------------------------------------------------------- import

def test_import_replaces_day_idempotently(tmp_path, monkeypatch):
    session_module, mod = _reload_with_tmp_db(tmp_path, monkeypatch)
    from sqlalchemy import select
    from nba_betting.db.models import HistoricalInjury, Team

    sess = session_module.get_session()
    sess.add(Team(id=1610612738, abbreviation="BOS", name="Celtics"))
    sess.commit(); sess.close()

    snaps = tmp_path / "snaps"
    ts = datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc)
    mod.capture_injuries_to_jsonl(
        snaps, timestamp=ts,
        injuries=[_inj("Amy Adams", "BOS"), _inj("Bob Brown", "BOS", "Day-to-Day", 2.0, "3")])

    r1 = mod.import_injuries_jsonl(snaps)
    r2 = mod.import_injuries_jsonl(snaps)          # idempotent
    assert (r1["files"], r1["days"], r1["rows"]) == (1, 1, 2)
    assert r2["rows"] == 2 and not r2["errors"]

    sess = session_module.get_session()
    rows = sess.execute(select(HistoricalInjury)).scalars().all()
    assert len(rows) == 2
    assert {r.player_name for r in rows} == {"Amy Adams", "Bob Brown"}
    assert all(r.snapshot_date == date(2026, 4, 18) and r.team_id == 1610612738 for r in rows)
    sess.close()

    # A later capture of the same day (player cleared) supersedes the rows.
    mod.capture_injuries_to_jsonl(
        snaps, timestamp=ts.replace(hour=22), injuries=[_inj("Amy Adams", "BOS", "Questionable", 3.5)])
    mod.import_injuries_jsonl(snaps)
    sess = session_module.get_session()
    rows = sess.execute(select(HistoricalInjury)).scalars().all()
    assert [(r.player_name, r.status, r.impact_rating) for r in rows] == [("Amy Adams", "Questionable", 3.5)]
    sess.close()


def test_import_tolerates_bad_lines_and_uses_stem_date(tmp_path, monkeypatch):
    session_module, mod = _reload_with_tmp_db(tmp_path, monkeypatch)
    from sqlalchemy import select
    from nba_betting.db.models import HistoricalInjury

    snaps = tmp_path / "snaps"; snaps.mkdir()
    (snaps / "2026-04-20.jsonl").write_text(
        "not json\n"
        + json.dumps({"player_name": "No Team", "status": "Out"}) + "\n"
        + json.dumps({"player_name": "Cy Cole", "team_abbr": "lal", "status": "Out",
                      "impact_rating": "bad"}) + "\n"
    )
    res = mod.import_injuries_jsonl(snaps)
    assert len(res["errors"]) == 2 and res["rows"] == 1
    sess = session_module.get_session()
    row = sess.execute(select(HistoricalInjury)).scalars().one()
    assert (row.snapshot_date, row.team_abbr, row.impact_rating) == (date(2026, 4, 20), "LAL", 0.0)
    sess.close()


# ------------------------------------------------- ESPN builder (no file I/O)

def test_build_injury_list_from_espn_estimates_impact_and_keeps_overrides(monkeypatch):
    from nba_betting.data import espn, injuries as inj_mod

    monkeypatch.setattr(espn, "fetch_injuries", lambda: [
        {"player_name": "Star Guy", "player_id": "11", "team_abbr": "BOS",
         "status": "Out", "description": "ankle", "date": "2026-04-18"},
        {"player_name": "Bench Guy", "player_id": "12", "team_abbr": "BOS",
         "status": "Questionable", "description": "", "date": ""},
    ])
    monkeypatch.setattr(espn, "fetch_depth_chart", lambda tid: {
        "PG": [{"espn_id": "11", "rank": 1}, {"espn_id": "12", "rank": 3}],
    })
    override = PlayerInjury(player_name="Manual Man", team_abbr="LAL", status="Out",
                            reason="manual", impact_rating=9.0)

    out = inj_mod.build_injury_list_from_espn({"manual man": override})
    by_name = {i.player_name: i for i in out}
    assert by_name["Star Guy"].impact_rating == 7.0          # starter, Out
    assert by_name["Bench Guy"].impact_rating == 1.0         # bench (2.0) * questionable (0.5)
    assert by_name["Manual Man"] is override                 # override appended untouched

    monkeypatch.setattr(espn, "fetch_injuries", lambda: [])
    assert inj_mod.build_injury_list_from_espn() == []
