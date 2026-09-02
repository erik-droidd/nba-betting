"""ESPN injury sync robustness (2026-09).

- ESPN's injuries endpoint dropped ``athlete.id``; the id is recovered
  from the athlete's links / headshot URL.
- Manual overrides are flagged by ``source="manual"``, not inferred from
  a missing id (which made every ESPN entry immortal once ids vanished:
  258 stale entries in injuries.json).
- Status lookups are case-insensitive (ESPN says ``Day-To-Day``).
"""
from __future__ import annotations

import json

import pytest

from nba_betting.data.injuries import PlayerInjury, _status_multiplier, _estimate_impact_rating


def test_athlete_id_recovered_from_links_or_headshot():
    from nba_betting.data.espn import _athlete_id

    assert _athlete_id({"id": 123}) == "123"
    assert _athlete_id({"links": [{"href": "https://www.espn.com/nba/player/_/id/4712863/mouhamed-gueye"}]}) == "4712863"
    assert _athlete_id({"headshot": {"href": "https://a.espncdn.com/i/headshots/nba/players/full/4712863.png"}}) == "4712863"
    assert _athlete_id({"links": [{"href": "sportscenter://x-callback-url/showClubhouse?uid=s:40~l:46~a:99"}]}) == "99"
    assert _athlete_id({"displayName": "No Id"}) == ""


def test_status_lookup_is_case_insensitive():
    assert _status_multiplier("Day-To-Day") == _status_multiplier("Day-to-Day") == 0.60
    assert _status_multiplier("OUT") == 1.0
    assert _status_multiplier("Out For Season") == 1.0
    assert _status_multiplier("questionable") == 0.5
    assert _status_multiplier("") == 0.5
    assert _estimate_impact_rating("Day-To-Day", 1) == pytest.approx(7.0 * 0.6)
    assert _estimate_impact_rating("out", 2) == 4.0


def test_sync_drops_stale_espn_entries_but_keeps_manual(tmp_path, monkeypatch):
    from nba_betting.data import injuries as inj_mod

    monkeypatch.setattr(inj_mod, "INJURIES_FILE", tmp_path / "injuries.json")
    monkeypatch.setattr(inj_mod, "persist_historical_injuries", lambda *a, **k: 0)

    # Existing file: a legacy ESPN entry with no id and no `source`
    # (the pre-fix shape), a current-format ESPN entry, and a manual one.
    (tmp_path / "injuries.json").write_text(json.dumps([
        {"player_name": "Healed Guy", "team_abbr": "BOS", "status": "Out", "reason": "",
         "impact_rating": 7.0, "date_reported": "2025-11-01", "player_id": ""},
        {"player_name": "Still Hurt", "team_abbr": "BOS", "status": "Out", "reason": "",
         "impact_rating": 7.0, "date_reported": "2026-04-01", "player_id": "5", "source": "espn"},
        {"player_name": "Manual Man", "team_abbr": "LAL", "status": "Out", "reason": "manual",
         "impact_rating": 9.0, "date_reported": "2026-04-10", "player_id": "", "source": "manual"},
    ]))
    monkeypatch.setattr(inj_mod, "build_injury_list_from_espn", lambda overrides: [
        PlayerInjury("Still Hurt", "BOS", "Out", "", 7.0, "2026-04-18", "5", source="espn"),
    ] + list(overrides.values()))

    out = inj_mod.sync_injuries_from_espn()
    names = {i.player_name for i in out}
    assert names == {"Still Hurt", "Manual Man"}          # legacy id-less entry dropped
    saved = json.loads((tmp_path / "injuries.json").read_text())
    assert {s["player_name"]: s["source"] for s in saved} == {"Still Hurt": "espn", "Manual Man": "manual"}
    # Round-trip keeps the field; legacy records default to "espn".
    assert {i.player_name: i.source for i in inj_mod.load_injuries()} == {"Still Hurt": "espn", "Manual Man": "manual"}


def test_add_injury_is_manual(tmp_path, monkeypatch):
    from nba_betting.data import injuries as inj_mod

    monkeypatch.setattr(inj_mod, "INJURIES_FILE", tmp_path / "injuries.json")
    inj = inj_mod.add_injury("Some Star", "lal", impact_rating=8.0)
    assert inj.source == "manual" and inj.team_abbr == "LAL"
    assert inj_mod.load_injuries()[0].source == "manual"
