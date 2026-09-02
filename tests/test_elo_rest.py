"""Back-to-back adjustment in the Elo model (2026-09).

Pins:
- the scalar / vectorized adjustment helpers agree and follow the
  ``rest_days <= 1`` b2b definition shared with ``features/rest_days.py``;
- ``predict_home_win_prob`` and ``update_elo`` apply the same adjusted
  expectation (schedule effects stay out of the rating);
- ``compute_all_elos`` threads real schedule rest into the update, so a
  stored rating replay matches an in-memory replay that uses rest;
- ``build_prediction_features`` computes ``elo_home_prob`` from the FINAL
  rest values (after the ``extra_features`` override the engine injects).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_betting.config import ELO_B2B_PENALTY, ELO_HOME_ADVANTAGE


def test_rest_adjustment_scalar_cases():
    from nba_betting.models.elo import rest_elo_adjustment as adj

    assert adj(2, 2) == 0.0
    assert adj(1, 3) == -ELO_B2B_PENALTY          # home on a b2b
    assert adj(3, 1) == +ELO_B2B_PENALTY          # away on a b2b
    assert adj(1, 1) == 0.0                       # both on a b2b: cancels
    assert adj(0, 2) == -ELO_B2B_PENALTY          # rest 0 still counts as b2b
    assert adj(None, 2) == 0.0                    # unknown -> no adjustment
    assert adj(float("nan"), 2) == 0.0
    assert adj("x", 2) == 0.0


def test_rest_adjustment_vec_matches_scalar():
    from nba_betting.models.elo import rest_elo_adjustment, rest_elo_adjustment_vec

    h = pd.Series([1, 3, 1, 2, np.nan, 0, 7])
    a = pd.Series([3, 1, 1, 2, 1, np.nan, 1])
    vec = rest_elo_adjustment_vec(h, a)
    for i in range(len(h)):
        hs = None if np.isnan(h[i]) else h[i]
        as_ = None if np.isnan(a[i]) else a[i]
        assert vec[i] == rest_elo_adjustment(hs, as_)


def test_predict_home_win_prob_b2b_direction_and_symmetry():
    from nba_betting.models.elo import predict_home_win_prob, expected_score

    p_rested = predict_home_win_prob(1500, 1500, 2, 2)
    p_home_b2b = predict_home_win_prob(1500, 1500, 1, 2)
    p_away_b2b = predict_home_win_prob(1500, 1500, 2, 1)
    assert p_home_b2b < p_rested < p_away_b2b
    # Exact formula: the penalty enters as Elo points on the home side.
    assert p_home_b2b == pytest.approx(
        expected_score(1500 + ELO_HOME_ADVANTAGE - ELO_B2B_PENALTY, 1500)
    )
    # No schedule context -> the legacy formula, unchanged.
    assert predict_home_win_prob(1500, 1500) == pytest.approx(
        expected_score(1500 + ELO_HOME_ADVANTAGE, 1500)
    )
    # Magnitude sanity: ~3-4pp at parity for the default penalty.
    assert 0.02 < p_rested - p_home_b2b < 0.05


def test_update_elo_uses_adjusted_expectation():
    from nba_betting.models.elo import update_elo

    # A home team on a b2b that LOSES was expected to lose more often, so it
    # should lose less rating than a rested home team dropping the same game.
    h_rested, _ = update_elo(1500, 1500, 100, 110, home_rest_days=2, away_rest_days=2)
    h_b2b, a_b2b = update_elo(1500, 1500, 100, 110, home_rest_days=1, away_rest_days=2)
    assert h_b2b > h_rested
    # Zero-sum is preserved.
    assert h_b2b + a_b2b == pytest.approx(3000.0)
    # Default (no rest passed) is the legacy update.
    assert update_elo(1500, 1500, 100, 110) == update_elo(
        1500, 1500, 100, 110, home_rest_days=None, away_rest_days=None
    )


def _replay_with_rest(games, k=20.0):
    """Reference in-memory replay: (date, home, away, hs, as) tuples."""
    from nba_betting.config import INITIAL_ELO
    from nba_betting.models.elo import update_elo

    elos: dict[int, float] = {}
    last: dict[int, date] = {}
    for d, h, a, hs, as_ in games:
        he = elos.get(h, INITIAL_ELO)
        ae = elos.get(a, INITIAL_ELO)
        hr = min((d - last[h]).days, 7) if h in last else 3
        ar = min((d - last[a]).days, 7) if a in last else 3
        elos[h], elos[a] = update_elo(he, ae, hs, as_, k, hr, ar)
        last[h] = d
        last[a] = d
    return elos


def test_compute_all_elos_threads_schedule_rest(tmp_path, monkeypatch):
    """The DB replay must match the reference replay that uses real rest."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from nba_betting.db.models import Base, Team, Game
    from nba_betting.models import elo as elo_mod

    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(elo_mod, "get_session", lambda: Session())

    s = Session()
    for tid in (1, 2, 3):
        s.add(Team(id=tid, abbreviation=f"T{tid}", name=f"Team {tid}"))
    d0 = date(2025, 10, 20)
    # Team 1 plays on consecutive days (b2b on day 1 and day 2).
    schedule = [
        (d0, 1, 2, 110, 100),
        (d0 + timedelta(days=1), 3, 1, 105, 108),   # team 1 on a b2b, away
        (d0 + timedelta(days=2), 1, 3, 99, 104),    # team 1 on a b2b, home
        (d0 + timedelta(days=5), 2, 3, 120, 90),
        (d0 + timedelta(days=6), 2, 1, 100, 101),   # team 2 on a b2b, home
    ]
    for i, (d, h, a, hs, as_) in enumerate(schedule):
        s.add(Game(id=f"g{i}", date=d, season="2025-26", home_team_id=h,
                   away_team_id=a, home_score=hs, away_score=as_, home_win=hs > as_))
    s.commit()
    s.close()

    got = elo_mod.compute_all_elos()
    want = _replay_with_rest(schedule)
    for tid in (1, 2, 3):
        assert got[tid] == pytest.approx(want[tid], abs=1e-9)

    # And it is NOT what a rest-blind replay produces (the b2b games above
    # change the expectation used in the update).
    blind = _replay_with_rest([(d + timedelta(days=10 * i), h, a, hs, as_)
                               for i, (d, h, a, hs, as_) in enumerate(schedule)])
    assert any(abs(got[t] - blind[t]) > 1e-6 for t in (1, 2, 3))


def _full_rolling_row(team_id: int, game_id: str, rest: int = 3) -> dict:
    """A rolling-frame row with every column build_prediction_features
    reads, so the >30%-NaN guard doesn't reject the fixture."""
    from nba_betting.features import builder as b

    row = {"game_id": game_id, "date": pd.Timestamp("2025-11-01"), "team_id": team_id,
           "rest_days": rest, "is_back_to_back": int(rest <= 1),
           "games_last_7": 2, "games_last_14": 5}
    for w in b._WINDOWS:
        for stat in b._ROLLING_STATS + ["efg_pct", "tov_pct", "orb_pct", "ft_rate"]:
            row[f"{stat}_roll_{w}"] = 1.0
        row[f"pts_roll_{w}"] = 110.0
        row[f"pts_against_roll_{w}"] = 108.0
    for w in (10, 20):
        for st in ("plus_minus", "net_rtg_game"):
            row[f"{st}_home_split_roll_{w}"] = 1.0
            row[f"{st}_away_split_roll_{w}"] = 1.0
    for w in (5, 10):
        row[f"fg3_pct_b2b_roll_{w}"] = 0.35
    for stat in b._SOS_PACE_ROLLING_STATS:
        for w in b._SOS_PACE_WINDOWS:
            row[f"{stat}_{w}"] = 100.0 if stat == "poss_roll" else 1.0
    for stat in b._EWM_STATS:
        row[stat] = 1.0
    return row


def test_build_prediction_features_elo_prob_uses_final_rest():
    """elo_home_prob must reflect the rest override from extra_features."""
    from nba_betting.features.builder import build_prediction_features
    from nba_betting.models.elo import predict_home_win_prob

    rdf = pd.DataFrame([_full_rolling_row(1, "a"), _full_rolling_row(2, "b")])
    kw = dict(feature_means={}, injury_impacts={})

    stale = build_prediction_features(1, 2, rdf, 1550, 1500, **kw)
    fresh = build_prediction_features(
        1, 2, rdf, 1550, 1500,
        extra_features={"home_rest_days": 1, "home_is_back_to_back": 1,
                        "away_rest_days": 3, "rest_diff": -2},
        **kw,
    )
    assert stale is not None and fresh is not None
    assert stale["elo_home_prob"].iloc[0] == pytest.approx(
        predict_home_win_prob(1550, 1500, 3, 3))
    assert fresh["elo_home_prob"].iloc[0] == pytest.approx(
        predict_home_win_prob(1550, 1500, 1, 3))
    assert fresh["elo_home_prob"].iloc[0] < stale["elo_home_prob"].iloc[0]
    # The stale row's rest is the rolling frame's (last completed game);
    # the override replaced it for the upcoming game.
    assert fresh["home_rest_days"].iloc[0] == 1 and fresh["rest_diff"].iloc[0] == -2
