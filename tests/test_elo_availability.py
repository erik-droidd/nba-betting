"""Player-availability term in the Elo model (2026-09)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_betting.config import ELO_AVAILABILITY_SCALE, ELO_HOME_ADVANTAGE


def test_availability_adjustment_scalar_and_vec():
    from nba_betting.models.elo import (
        availability_elo_adjustment as adj, availability_elo_adjustment_vec as vec,
    )
    assert adj(0.0, 0.0) == 0.0
    assert adj(0.2, 0.0) == pytest.approx(-0.2 * ELO_AVAILABILITY_SCALE)   # home missing 20%
    assert adj(0.0, 0.2) == pytest.approx(+0.2 * ELO_AVAILABILITY_SCALE)
    assert adj(0.3, 0.3) == 0.0
    assert adj(None, 0.2) == 0.0 and adj(float("nan"), 0.2) == 0.0 and adj("x", 0) == 0.0
    v = vec(pd.Series([0.2, np.nan, 0.0]), pd.Series([0.0, 0.1, 0.5]))
    assert v.tolist() == pytest.approx([-0.2 * ELO_AVAILABILITY_SCALE, 0.0, 0.5 * ELO_AVAILABILITY_SCALE])


def test_predict_and_update_use_availability():
    from nba_betting.models.elo import predict_home_win_prob, update_elo, expected_score

    p_full = predict_home_win_prob(1500, 1500, 2, 2, 0.0, 0.0)
    p_home_short = predict_home_win_prob(1500, 1500, 2, 2, 0.15, 0.0)
    assert p_home_short < p_full
    assert p_home_short == pytest.approx(
        expected_score(1500 + ELO_HOME_ADVANTAGE - 0.15 * ELO_AVAILABILITY_SCALE, 1500))
    # ~15% of minutes (a star) ≈ 3pp at parity.
    assert 0.02 < p_full - p_home_short < 0.05
    # Legacy calls (no context) unchanged.
    assert predict_home_win_prob(1500, 1500) == pytest.approx(expected_score(1540, 1500))
    # A short-handed home loss costs less rating than a full-strength loss.
    h_full, _ = update_elo(1500, 1500, 100, 110, home_missing_pct=0.0, away_missing_pct=0.0)
    h_short, a_short = update_elo(1500, 1500, 100, 110, home_missing_pct=0.3, away_missing_pct=0.0)
    assert h_short > h_full and h_short + a_short == pytest.approx(3000.0)


def test_compute_all_elos_threads_availability(tmp_path, monkeypatch):
    """With player logs present, the stored replay must equal a reference
    replay that feeds each game's missing-minutes share into update_elo."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from nba_betting.db.models import Base, Team, Game, PlayerGameStat
    from nba_betting.models import elo as elo_mod
    from nba_betting.features import availability as av_mod
    from nba_betting.config import INITIAL_ELO

    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(elo_mod, "get_session", lambda: Session())
    monkeypatch.setattr(av_mod, "get_session", lambda: Session())

    s = Session()
    for tid in (1, 2):
        s.add(Team(id=tid, abbreviation=f"T{tid}", name=f"Team {tid}"))
    d0 = date(2025, 10, 20)
    games = [(d0 + timedelta(days=3 * i), 1 if i % 2 == 0 else 2, 2 if i % 2 == 0 else 1,
              105 + i, 100) for i in range(6)]
    roster = {1: [(11, 36), (12, 30), (13, 20), (14, 14)], 2: [(21, 34), (22, 30), (23, 22), (24, 15)]}
    for i, (d, h, a, hs, as_) in enumerate(games):
        gid = f"g{i}"
        s.add(Game(id=gid, date=d, season="2025-26", home_team_id=h, away_team_id=a,
                   home_score=hs, away_score=as_, home_win=hs > as_))
        for tid in (h, a):
            for pid, mins in roster[tid]:
                if i == 3 and pid == 11:      # team 1's star sits game 3
                    continue
                s.add(PlayerGameStat(game_id=gid, team_id=tid, player_id=pid,
                                     player_name=f"P{pid}", minutes=mins, pts=10, ast=2, reb=3))
    s.commit(); s.close()

    got = elo_mod.compute_all_elos()

    av = av_mod.compute_availability_features(av_mod.load_player_game_df())
    miss = {(r.game_id, int(r.team_id)): r.missing_minutes_pct for r in av.itertuples()}
    assert miss[("g3", 1)] > 0.3            # star's 36 of 100 regular minutes
    elos = {1: INITIAL_ELO, 2: INITIAL_ELO}
    last = {}
    for i, (d, h, a, hs, as_) in enumerate(games):
        hr = min((d - last[h]).days, 7) if h in last else 3
        ar = min((d - last[a]).days, 7) if a in last else 3
        elos[h], elos[a] = elo_mod.update_elo(
            elos[h], elos[a], hs, as_, home_rest_days=hr, away_rest_days=ar,
            home_missing_pct=miss.get((f"g{i}", h)), away_missing_pct=miss.get((f"g{i}", a)))
        last[h] = d; last[a] = d
    assert got[1] == pytest.approx(elos[1], abs=1e-9) and got[2] == pytest.approx(elos[2], abs=1e-9)


def test_build_prediction_features_uses_availability_from_extra():
    from nba_betting.features.builder import build_prediction_features
    from nba_betting.models.elo import predict_home_win_prob
    from tests.test_elo_rest import _full_rolling_row

    rdf = pd.DataFrame([_full_rolling_row(1, "a"), _full_rolling_row(2, "b")])
    kw = dict(feature_means={}, injury_impacts={})
    base = build_prediction_features(1, 2, rdf, 1550, 1500, **kw)
    short = build_prediction_features(
        1, 2, rdf, 1550, 1500, **kw,
        extra_features={"home_missing_minutes_pct": 0.2, "away_missing_minutes_pct": 0.0,
                        "home_star_out": 1.0, "away_star_out": 0.0,
                        "diff_missing_minutes_pct": 0.2, "diff_available_talent": -0.15})
    assert base["elo_home_prob"].iloc[0] == pytest.approx(predict_home_win_prob(1550, 1500, 3, 3))
    assert short["elo_home_prob"].iloc[0] == pytest.approx(predict_home_win_prob(1550, 1500, 3, 3, 0.2, 0.0))
    assert short["elo_home_prob"].iloc[0] < base["elo_home_prob"].iloc[0]


def test_post_hoc_injury_adjustment_is_display_only(monkeypatch):
    """generate_recommendations must not shift the probability by the
    heuristic injury estimate (it is modelled inside the prediction now)."""
    from nba_betting.betting import recommendations as rec_mod
    from nba_betting.data import injuries as inj_mod

    # generate_recommendations imports the adjustment lazily from the
    # injuries module, so patch it there.
    monkeypatch.setattr(inj_mod, "get_team_injury_adjustment", lambda abbr: -0.10 if abbr == "BOS" else 0.0)
    monkeypatch.setattr(rec_mod, "get_recent_roi", lambda lookback=10: (0.0, 0))
    game = {"home_team_abbr": "BOS", "away_team_abbr": "LAL", "home_team_id": 1, "away_team_id": 2,
            "game_time_utc": "2026-01-10T00:00:00Z"}
    recs = rec_mod.generate_recommendations([game], {1: 1500.0, 2: 1500.0}, [], 1000.0,
                                            predict_fn=lambda h, a: 0.60)
    assert recs[0].model_home_prob == pytest.approx(0.60)
    assert recs[0].home_injury_adj == pytest.approx(-0.10)      # still reported
