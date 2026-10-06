"""ESPN odds parsing (2026-10 audit).

ESPN moved moneylines from ``homeTeamOdds.moneyLine`` to
``moneyline.<side>.close.odds``; reading only the old field made every
ESPN probability since 2026-04 the 2.5%/pt spread proxy.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _event(odds: dict) -> dict:
    return {"events": [{
        "id": "1", "date": "2026-10-06T23:00Z",
        "competitions": [{
            "status": {"type": {"name": "STATUS_SCHEDULED"}},
            "competitors": [
                {"homeAway": "home", "team": {"id": "30", "abbreviation": "CHA", "displayName": "Hornets"}},
                {"homeAway": "away", "team": {"id": "17", "abbreviation": "BKN", "displayName": "Nets"}},
            ],
            "odds": [odds],
        }],
    }]}


CURRENT_SHAPE = {   # trimmed from the live 2026-10-06 payload
    "provider": {"name": "DraftKings"}, "spread": -5.5, "overUnder": 212.5,
    "homeTeamOdds": {"favorite": True}, "awayTeamOdds": {"favorite": False},
    "moneyline": {"home": {"close": {"odds": "-205"}, "open": {"odds": "-192"}},
                  "away": {"close": {"odds": "+170"}, "open": {"odds": "+160"}}},
}


def test_scoreboard_reads_moneyline_from_current_espn_shape(monkeypatch):
    from nba_betting.data import espn
    monkeypatch.setattr(espn, "_get", lambda path, params=None: _event(CURRENT_SHAPE))
    odds = espn.fetch_scoreboard("20261006")[0]["odds"]
    assert (odds["home_moneyline"], odds["away_moneyline"]) == (-205.0, 170.0)
    assert (odds["spread"], odds["over_under"]) == (-5.5, 212.5)


def test_scoreboard_still_reads_legacy_moneyline_field(monkeypatch):
    from nba_betting.data import espn
    legacy = {"spread": -2.0, "overUnder": 220.0,
              "homeTeamOdds": {"moneyLine": -130}, "awayTeamOdds": {"moneyLine": 110}}
    monkeypatch.setattr(espn, "_get", lambda path, params=None: _event(legacy))
    odds = espn.fetch_scoreboard()[0]["odds"]
    assert (odds["home_moneyline"], odds["away_moneyline"]) == (-130.0, 110.0)


@pytest.mark.parametrize("raw,expected", [
    ("-205", -205.0), ("+170", 170.0), ("EVEN", 100.0), (" -110 ", -110.0),
    (-150, -150.0), ("OFF", None), ("", None), (None, None), ("0", None),
])
def test_parse_american(raw, expected):
    from nba_betting.data.espn import _parse_american
    assert _parse_american(raw) == expected


def test_get_espn_odds_devigs_moneyline_and_flags_spread_proxy(monkeypatch):
    from nba_betting.data import espn_odds
    games = [
        {"home_team": {"abbr": "CHA"}, "away_team": {"abbr": "BKN"},
         "odds": {"home_moneyline": -205.0, "away_moneyline": 170.0, "spread": -5.5, "over_under": 212.5}},
        {"home_team": {"abbr": "OKC"}, "away_team": {"abbr": "NOP"},
         "odds": {"home_moneyline": None, "away_moneyline": None, "spread": -3.5, "over_under": 221.5}},
    ]
    monkeypatch.setattr(espn_odds, "fetch_scoreboard", lambda date_str=None: games)
    ml, proxy = espn_odds.get_espn_odds()

    assert ml["prob_source"] == "moneyline"
    assert ml["teams"]["CHA"] == pytest.approx(0.6447, abs=1e-4)
    assert espn_odds.market_home_prob(ml, "CHA") == ml["teams"]["CHA"]
    assert (ml["home_moneyline"], ml["away_moneyline"]) == (-205.0, 170.0)

    assert proxy["prob_source"] == "spread"
    assert proxy["teams"]["OKC"] == pytest.approx(0.5875)
    assert espn_odds.market_home_prob(proxy, "OKC") is None   # never stored as a price


def test_is_spread_heuristic():
    from nba_betting.data.espn_odds import is_spread_heuristic
    assert is_spread_heuristic(0.5875, -3.5)
    assert is_spread_heuristic(0.3375, 6.5)
    assert not is_spread_heuristic(0.6447, -5.5)      # a real de-vigged price
    assert not is_spread_heuristic(0.5, 0.0)          # pick'em: proxy == market
    assert not is_spread_heuristic(None, -3.5)
    assert not is_spread_heuristic(0.6, None)


def test_prob_movement_stays_within_one_source():
    """Both sources share each capture's timestamp, so first/last over all
    rows compared the first Polymarket price with the last ESPN price."""
    from nba_betting.data.odds_tracker import _prob_movement
    s = lambda src, p: SimpleNamespace(source=src, home_prob=p)
    snaps = [s("polymarket", 0.60), s("espn", 0.55), s("polymarket", 0.63), s("espn", 0.56)]
    assert _prob_movement(snaps) == pytest.approx(0.03)
    espn_only = [s("polymarket", 0.60), s("espn", 0.55), s("espn", 0.52)]
    assert _prob_movement(espn_only) == pytest.approx(-0.03)
    assert _prob_movement([s("polymarket", 0.6), s("espn", None)]) == 0.0


def test_recommendations_never_price_a_bet_off_the_spread_proxy(monkeypatch):
    """No Polymarket market + a spread-only ESPN line = no market price:
    the 2.5%/pt proxy would otherwise produce edges and stakes."""
    from nba_betting.betting import recommendations as rec_mod
    monkeypatch.setattr(rec_mod, "get_recent_roi", lambda lookback=10: (0.0, 0))
    game = {"home_team_abbr": "BOS", "away_team_abbr": "LAL", "home_team_id": 1, "away_team_id": 2,
            "game_time_utc": "2026-01-10T00:00:00Z"}
    espn = lambda source, p: [{"teams": {"BOS": p, "LAL": 1 - p}, "prob_source": source,
                               "spread": -6.5, "over_under": 224.5}]

    proxy = rec_mod.generate_recommendations([game], {1: 1500.0, 2: 1500.0}, [], 1000.0,
                                             predict_fn=lambda h, a: 0.75,
                                             espn_odds=espn("spread", 0.6625))[0]
    assert proxy.market_home_prob == 0.0 and proxy.bet_size == 0
    assert proxy.spread == -6.5                     # the spread itself is still shown

    real = rec_mod.generate_recommendations([game], {1: 1500.0, 2: 1500.0}, [], 1000.0,
                                            predict_fn=lambda h, a: 0.75,
                                            espn_odds=espn("moneyline", 0.70))[0]
    assert real.market_home_prob == pytest.approx(0.70)
