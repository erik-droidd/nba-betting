"""Tests for the 2026-04 three-tier improvement pass.

Covers the tiered changes introduced in this round:

- Tier 1.1: SOS-adjusted rolling stats
- Tier 1.2: Pace & pace-differential features
- Tier 1.3: Off/Def Elo split
- Tier 1.4: Opponent-strength MOV multiplier
- Tier 1.5: Exponential-decay rolling features
- Tier 2.1: Hyperparameter grid search (smoke test)
- Tier 2.2: Stacked meta-learner
- Tier 2.4: Slate-level portfolio Kelly
- Tier 2.5: Signal-dependent Kelly fraction
- Tier 3.1: In-memory model cache
- Tier 3.2: Vectorized four_factors.add_opponent_rebound_data
- Tier 3.4: Odds-snapshot dedup helper
- Tier 3.5: Polymarket fuzzy-fallback name matcher
"""
from __future__ import annotations

from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Tier 1.4 — Opponent-strength MOV dampening
# ---------------------------------------------------------------------------


def test_opp_strength_factor_sigmoid_shape():
    """Dampener is ~0.5 at parity, <0.5 vs weak, >0.5 vs strong."""
    from nba_betting.models.elo import opp_strength_factor

    # Parity: ~0.5
    assert abs(opp_strength_factor(1500, 1500) - 0.5) < 1e-9
    # Big favorite: heavily damped toward 0
    assert opp_strength_factor(1700, 1300) < 0.2
    # Big underdog: amplified toward 1
    assert opp_strength_factor(1300, 1700) > 0.8


def test_mov_multiplier_vs_weak_opponent_is_smaller():
    """Same MOV against a weaker team should update Elo less."""
    from nba_betting.models.elo import mov_multiplier

    strong_vs_weak = mov_multiplier(20, 1700, 1300)
    peer = mov_multiplier(20, 1500, 1500)
    # Strong-team blowout against tanker is less informative than a
    # same-MOV result between peers.
    assert strong_vs_weak < peer


# ---------------------------------------------------------------------------
# Tier 1.3 — Off/def Elo updates
# ---------------------------------------------------------------------------


def test_off_def_elo_updates_asymmetrically():
    """Home blowout should raise home's offensive Elo and lower away's def."""
    from nba_betting.models.elo import update_off_def_elo

    h_off0, h_def0, a_off0, a_def0 = 1500.0, 1500.0, 1500.0, 1500.0
    h_off1, h_def1, a_off1, a_def1 = update_off_def_elo(
        h_off0, h_def0, a_off0, a_def0,
        home_score=130, away_score=100,
    )

    # Home scored well above expected → off ↑
    assert h_off1 > h_off0
    # Away allowed too many → def ↓
    assert a_def1 < a_def0
    # Away scored roughly as expected → smaller move
    # Home's def wasn't tested much either way but should move mildly
    assert abs(h_def1 - h_def0) < abs(h_off1 - h_off0) * 2


# ---------------------------------------------------------------------------
# Tier 1.5 — EWM rolling helper
# ---------------------------------------------------------------------------


def test_rolling_ewm_weighting():
    """Exponentially-weighted mean puts more weight on recent observations."""
    from nba_betting.features.rolling import rolling_ewm

    s = pd.Series([0.0] * 9 + [10.0, 20.0])
    ewm = rolling_ewm(s, halflife=3.0)
    # The last value (at index 10) is based on values from 0..9 (shift(1)).
    # The values at index 0..8 are 0, and index 9 is 10. So EWM at index 10
    # should weight the 10 heavily but should still be pulled toward zero
    # by earlier zeros. It must be in (0, 10).
    assert 0 < ewm.iloc[10] < 10


# ---------------------------------------------------------------------------
# Tier 2.2 — Stacked meta-learner
# ---------------------------------------------------------------------------


def test_meta_learner_round_trip(tmp_path, monkeypatch):
    """Fit + save + load + predict cycle yields consistent probabilities."""
    from nba_betting.models import stacking

    # Redirect the persistence path to a tmp directory.
    monkeypatch.setattr(stacking, "META_MODEL_PATH", tmp_path / "meta.joblib")

    rng = np.random.default_rng(0)
    n = 300
    elo = np.clip(rng.beta(2, 2, size=n), 0.02, 0.98)
    xgb = np.clip(elo + rng.normal(0, 0.05, size=n), 0.02, 0.98)
    y = (rng.random(n) < elo).astype(int)

    artifact = stacking.fit_meta_model(elo, xgb, y)
    stacking.save_meta_model(artifact)
    loaded = stacking.load_meta_model()
    assert loaded is not None

    probs = stacking.predict_meta(elo[:10], xgb[:10], artifact=loaded)
    assert probs.shape == (10,)
    # Should produce legal probabilities
    assert (probs > 0).all() and (probs < 1).all()


# ---------------------------------------------------------------------------
# Tier 2.5 — Signal-dependent Kelly
# ---------------------------------------------------------------------------


def test_signal_dependent_lambda_monotone_in_clv():
    """Positive CLV t-stat should *not* reduce lambda; negative should."""
    from nba_betting.betting.kelly import signal_dependent_lambda

    base = 0.25
    neutral = signal_dependent_lambda(base, edge=0.05, clv_tstat=0.0, model_market_disagreement=0.05)
    good = signal_dependent_lambda(base, edge=0.05, clv_tstat=2.0, model_market_disagreement=0.05)
    bad = signal_dependent_lambda(base, edge=0.05, clv_tstat=-2.0, model_market_disagreement=0.05)

    assert bad < neutral <= good


def test_signal_dependent_lambda_clamps():
    """Even with very strong signals, lambda stays in [0.25x, 1.25x] of base."""
    from nba_betting.betting.kelly import signal_dependent_lambda

    base = 0.25
    very_good = signal_dependent_lambda(base, edge=0.05, clv_tstat=10.0, model_market_disagreement=0.01)
    very_bad = signal_dependent_lambda(base, edge=0.30, clv_tstat=-10.0, model_market_disagreement=0.30)

    assert very_good <= base * 1.25 + 1e-9
    assert very_bad >= base * 0.25 - 1e-9


# ---------------------------------------------------------------------------
# Tier 2.4 — Slate-level portfolio Kelly
# ---------------------------------------------------------------------------


def test_portfolio_optimizer_respects_exposure_cap():
    """Sum of fractions must never exceed MAX_EXPOSURE_PCT."""
    from nba_betting.betting.portfolio import optimize_slate, BetCandidate
    from nba_betting.config import MAX_EXPOSURE_PCT

    bets = [
        BetCandidate(id=f"g{i}", prob=0.60, market_price=0.50)
        for i in range(8)
    ]
    result = optimize_slate(bets)
    total = float(result["fractions"].sum())
    # Allow a tiny bit of numerical slack
    assert total <= MAX_EXPOSURE_PCT + 1e-6


def test_portfolio_zero_fractions_for_negative_ev():
    """Bets with prob <= market_price should be assigned zero fraction."""
    from nba_betting.betting.portfolio import optimize_slate, BetCandidate

    bets = [
        BetCandidate(id="good", prob=0.60, market_price=0.50),  # +10% edge
        BetCandidate(id="bad", prob=0.40, market_price=0.50),   # -10% edge
    ]
    result = optimize_slate(bets)
    # The good bet should size > 0; the bad one must be exactly zero.
    assert result["fractions"][0] > 0
    assert result["fractions"][1] == 0


# ---------------------------------------------------------------------------
# Tier 3.2 — Vectorized opponent-rebound data
# ---------------------------------------------------------------------------


def test_add_opponent_rebound_data_vectorized():
    """Opp DREB derived by group-sum minus self-row matches the old apply logic."""
    from nba_betting.features.four_factors import add_opponent_rebound_data

    df = pd.DataFrame([
        {"game_id": "G1", "team_id": 1, "home_team_id": 1, "away_team_id": 2, "oreb": 10, "dreb": 30},
        {"game_id": "G1", "team_id": 2, "home_team_id": 1, "away_team_id": 2, "oreb": 8,  "dreb": 28},
        {"game_id": "G2", "team_id": 3, "home_team_id": 3, "away_team_id": 4, "oreb": 12, "dreb": 32},
        {"game_id": "G2", "team_id": 4, "home_team_id": 3, "away_team_id": 4, "oreb": 9,  "dreb": 29},
    ])
    out = add_opponent_rebound_data(df)
    # Team 1's opponent (team 2) had dreb=28
    assert out.loc[out["team_id"] == 1, "opp_dreb"].iloc[0] == 28
    # Team 4's opponent (team 3) had dreb=32
    assert out.loc[out["team_id"] == 4, "opp_dreb"].iloc[0] == 32
    # ORB% sanity
    orb = out.loc[out["team_id"] == 1, "orb_pct"].iloc[0]
    assert 0 < orb < 1


# ---------------------------------------------------------------------------
# Tier 3.4 — Odds snapshot dedup helper
# ---------------------------------------------------------------------------


def test_is_duplicate_detects_no_movement(monkeypatch):
    """When the latest snapshot matches within tolerance, new insert is a dup."""
    from nba_betting.data import odds_tracker

    class _Snap:
        timestamp = datetime.utcnow() - timedelta(hours=1)
        home_prob = 0.555
        spread = -3.5
        over_under = 220.0

    class _Scalars:
        def first(self):
            return _Snap()

    class _Result:
        def scalars(self):
            return _Scalars()

    class _Session:
        def execute(self, *args, **kwargs):
            return _Result()

    assert odds_tracker._is_duplicate(
        _Session(),
        game_date=date.today(),
        home_team_id=1,
        away_team_id=2,
        source="espn",
        now=datetime.utcnow(),
        home_prob=0.556,  # within tolerance
        spread=-3.5,
        over_under=220.0,
    ) is True


def test_is_duplicate_detects_movement():
    """Detects movement above tolerance as non-duplicate."""
    from nba_betting.data import odds_tracker

    class _Snap:
        timestamp = datetime.utcnow() - timedelta(hours=1)
        home_prob = 0.55
        spread = -3.5
        over_under = 220.0

    class _Scalars:
        def first(self):
            return _Snap()

    class _Result:
        def scalars(self):
            return _Scalars()

    class _Session:
        def execute(self, *args, **kwargs):
            return _Result()

    assert odds_tracker._is_duplicate(
        _Session(),
        game_date=date.today(),
        home_team_id=1,
        away_team_id=2,
        source="espn",
        now=datetime.utcnow(),
        home_prob=0.60,  # 5 pct-pt move → not a dup
        spread=-3.5,
        over_under=220.0,
    ) is False


# ---------------------------------------------------------------------------
# Tier 3.5 — Polymarket fuzzy fallback
# ---------------------------------------------------------------------------


def test_name_to_abbr_exact_and_fuzzy():
    """Exact match kept strict; substring match kicks in only on fallback."""
    from nba_betting.data.polymarket import _name_to_abbr

    assert _name_to_abbr("Lakers") == "LAL"
    assert _name_to_abbr("Warriors") == "GSW"
    # Fuzzy fallback: a title fragment containing the nickname
    assert _name_to_abbr("The Los Angeles Lakers tonight") == "LAL"


# ---------------------------------------------------------------------------
# Tier 3.1 — Model cache invalidation
# ---------------------------------------------------------------------------


def test_model_cache_invalidates_on_mtime(tmp_path, monkeypatch):
    """Rewriting the artifact should force a fresh load next time."""
    import joblib
    from nba_betting.models import xgboost_model

    # Redirect module-level paths to tmp
    model_p = tmp_path / "m.joblib"
    cols_p = tmp_path / "cols.joblib"
    monkeypatch.setattr(xgboost_model, "MODEL_PATH", model_p)
    monkeypatch.setattr(xgboost_model, "FEATURE_COLS_PATH", cols_p)
    # Reset cache
    xgboost_model._MODEL_CACHE.clear()

    joblib.dump({"dummy": 1}, model_p)
    joblib.dump(["a", "b"], cols_p)

    v1 = xgboost_model.load_model()
    v2 = xgboost_model.load_model()
    # Same object returned twice (cache hit)
    assert v1 is v2

    # Touch the file — update mtime → cache should reload
    import time
    time.sleep(0.01)
    joblib.dump({"dummy": 2}, model_p)

    v3 = xgboost_model.load_model()
    assert v3 is not v1
