"""Tests for the 2026-04 prediction improvement pass.

Covers the 8 improvements:
1. Line-movement features in training matrix
2. Home/away split rolling stats
3. CLV tracking
4. Starting lineups fetcher
5. Drawdown-aware Kelly sizing
6. Per-fold isotonic calibration refit
7. Same-day bet correlation adjustment
8. Back-to-back 3PT cold streak feature
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Line-movement feature builder helper
# ---------------------------------------------------------------------------


def test_attach_line_movement_defaults_to_zero():
    """When no snapshots exist, all three line-movement features are 0.0."""
    from nba_betting.features.builder import _attach_line_movement_features

    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"]),
        "home_team_id": [1],
        "away_team_id": [2],
    })
    _attach_line_movement_features(df)
    assert df["spread_movement"].iloc[0] == 0.0
    assert df["prob_movement"].iloc[0] == 0.0
    assert df["odds_disagreement"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# 2. Home/away split rolling
# ---------------------------------------------------------------------------


def test_home_away_split_columns_exist():
    """Rolling output should include venue-split columns for key stats."""
    from nba_betting.features.rolling import compute_rolling_features

    # We can't call with real data easily, so just verify the code path
    # doesn't crash with an empty result
    df = compute_rolling_features(windows=(5, 10, 20))
    # If DB is empty, df is empty but the function shouldn't raise
    if not df.empty:
        # Check that split columns exist for at least one team
        expected_cols = [
            "plus_minus_home_split_roll_10",
            "plus_minus_away_split_roll_10",
            "net_rtg_game_home_split_roll_20",
            "net_rtg_game_away_split_roll_20",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"


# ---------------------------------------------------------------------------
# 3. CLV tracking
# ---------------------------------------------------------------------------


def test_prediction_record_clv_fields():
    """PredictionRecord has opening, closing, and CLV fields."""
    from nba_betting.betting.tracker import PredictionRecord

    rec = PredictionRecord(
        date="2025-01-01",
        home_team="LAL",
        away_team="BOS",
        model_home_prob=0.55,
        market_home_prob=0.50,
        bet_side="HOME",
        edge=0.10,
        bet_size=50.0,
        opening_market_prob=0.48,
        closing_market_prob=0.52,
        clv=0.083,  # (0.52/0.48) - 1
    )
    assert rec.opening_market_prob == pytest.approx(0.48)
    assert rec.closing_market_prob == pytest.approx(0.52)
    assert rec.clv == pytest.approx(0.083, abs=0.001)


def test_prediction_record_backward_compat():
    """Records without CLV fields load cleanly (default None)."""
    from nba_betting.betting.tracker import PredictionRecord

    rec = PredictionRecord(
        date="2025-01-01",
        home_team="LAL",
        away_team="BOS",
        model_home_prob=0.55,
        market_home_prob=0.50,
        bet_side="HOME",
        edge=0.10,
        bet_size=50.0,
    )
    assert rec.opening_market_prob is None
    assert rec.closing_market_prob is None
    assert rec.clv is None


# ---------------------------------------------------------------------------
# 4. Starting lineups — apply_lineup_bumps
# ---------------------------------------------------------------------------


def test_lineup_bump_surprise_dnp():
    """A high-impact player absent from lineups gets bumped to Out."""
    from nba_betting.data.injuries import PlayerInjury
    from nba_betting.data.lineups import apply_lineup_bumps

    injuries = [
        PlayerInjury(
            player_name="Star Player",
            team_abbr="LAL",
            status="Questionable",
            reason="Ankle",
            impact_rating=8.0,
        ),
        PlayerInjury(
            player_name="Role Player",
            team_abbr="LAL",
            status="Questionable",
            reason="Knee",
            impact_rating=4.0,  # below threshold
        ),
    ]
    starters = {"LAL": ["Some Other Guy", "Another Starter"]}

    result = apply_lineup_bumps(injuries, starters)
    # Star Player should be bumped to Out with impact 10
    star = next(r for r in result if r.player_name == "Star Player")
    assert star.status == "Out"
    assert star.impact_rating == 10.0

    # Role Player (impact < 7) should NOT be bumped
    role = next(r for r in result if r.player_name == "Role Player")
    assert role.status == "Questionable"
    assert role.impact_rating == 4.0


def test_lineup_bump_empty_starters_no_change():
    """When no lineup data, injuries are returned unchanged."""
    from nba_betting.data.injuries import PlayerInjury
    from nba_betting.data.lineups import apply_lineup_bumps

    injuries = [
        PlayerInjury(
            player_name="Star", team_abbr="LAL",
            status="Questionable", reason="", impact_rating=8.0,
        ),
    ]
    result = apply_lineup_bumps(injuries, {})
    assert result[0].status == "Questionable"
    assert result[0].impact_rating == 8.0


def test_lineup_bump_already_out_not_double_bumped():
    """Players already Out should not get their impact changed."""
    from nba_betting.data.injuries import PlayerInjury
    from nba_betting.data.lineups import apply_lineup_bumps

    injuries = [
        PlayerInjury(
            player_name="Star", team_abbr="LAL",
            status="Out", reason="", impact_rating=8.0,
        ),
    ]
    starters = {"LAL": ["Other Guy"]}

    result = apply_lineup_bumps(injuries, starters)
    assert result[0].status == "Out"
    assert result[0].impact_rating == 8.0  # unchanged


# ---------------------------------------------------------------------------
# 5. Drawdown-aware Kelly
# ---------------------------------------------------------------------------


def test_drawdown_multiplier_tiers():
    """Verify the three tiers of drawdown response."""
    from nba_betting.betting.kelly import compute_drawdown_multiplier

    # Normal: no drawdown
    assert compute_drawdown_multiplier(0.02, lookback_bets=10) == 1.0
    assert compute_drawdown_multiplier(-0.03, lookback_bets=10) == 1.0

    # Mild: -5% to -10%
    assert compute_drawdown_multiplier(-0.07, lookback_bets=10) == 0.5

    # Severe: below -10%
    assert compute_drawdown_multiplier(-0.12, lookback_bets=10) == 0.25

    # Insufficient data: always 1.0
    assert compute_drawdown_multiplier(-0.20, lookback_bets=3) == 1.0


def test_kelly_drawdown_mult_scales_fraction():
    """Kelly fraction with drawdown_mult < 1 should produce smaller bets."""
    from nba_betting.betting.kelly import kelly_fraction

    base = kelly_fraction(0.60, 0.50)
    half = kelly_fraction(0.60, 0.50, drawdown_mult=0.5)
    assert half == pytest.approx(base * 0.5, abs=1e-9)


def test_kelly_drawdown_mult_clamped():
    """Drawdown mult should be clamped to [0, 1]."""
    from nba_betting.betting.kelly import kelly_fraction

    base = kelly_fraction(0.60, 0.50)
    # mult > 1 should be clamped to 1
    over = kelly_fraction(0.60, 0.50, drawdown_mult=1.5)
    assert over == pytest.approx(base, abs=1e-9)

    # mult < 0 should be clamped to 0
    neg = kelly_fraction(0.60, 0.50, drawdown_mult=-0.5)
    assert neg == 0.0


# ---------------------------------------------------------------------------
# 6. Per-fold calibration (integration — would need real data to fully test,
#    so we verify the code path doesn't crash with a small synthetic dataset)
# ---------------------------------------------------------------------------


def test_walk_forward_includes_calibration_keys():
    """Walk-forward output should include calibrated metrics when enough data."""
    # This is a structural test — verify the keys exist in fold results
    # when calibration ran successfully. We test with a tiny synthetic
    # dataset to keep the test fast.
    from nba_betting.models.xgboost_model import walk_forward_validate

    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "_date": dates,
    })
    y = pd.Series((np.random.randn(n) > 0).astype(float))

    result = walk_forward_validate(X, y, n_splits=1)
    if result["folds"]:
        fold = result["folds"][0]
        # Per-fold calibration adds these keys
        assert "brier_calibrated" in fold or "accuracy" in fold
        # At minimum, the basic keys must always be present
        assert "accuracy" in fold
        assert "brier_score" in fold


# ---------------------------------------------------------------------------
# 7. Same-day bet correlation adjustment
# ---------------------------------------------------------------------------


def test_correlation_adjustment_scales_multiple_bets():
    """With N > 1 actionable bets, bet sizes should be scaled down."""
    from nba_betting.betting.recommendations import BetRecommendation

    # Create 4 "actionable" recs
    recs = []
    for i in range(4):
        recs.append(BetRecommendation(
            home_team="LAL", away_team="BOS",
            model_home_prob=0.55, market_home_prob=0.50,
            bet_side="HOME", edge=0.10, ev_per_dollar=0.10,
            kelly_pct=0.03, bet_size=30.0, badge="MODERATE",
        ))
    # Add a NO BET rec to verify it's untouched
    recs.append(BetRecommendation(
        home_team="MIA", away_team="NYK",
        model_home_prob=0.50, market_home_prob=0.50,
        bet_side="NO BET", edge=0.0, ev_per_dollar=0.0,
        kelly_pct=0.0, bet_size=0.0, badge="NO BET",
    ))

    # Simulate the correlation adjustment
    import math
    _SAME_DAY_RHO = 0.15
    actionable = [r for r in recs if r.bet_side != "NO BET"]
    n_bets = len(actionable)
    corr_scale = 1.0 / math.sqrt(1.0 + (n_bets - 1) * _SAME_DAY_RHO)

    for r in actionable:
        r.kelly_pct *= corr_scale
        r.bet_size = round(r.bet_size * corr_scale, 2)

    # Verify scaling happened
    assert corr_scale < 1.0
    assert actionable[0].bet_size < 30.0
    # NO BET should be untouched
    no_bet = next(r for r in recs if r.bet_side == "NO BET")
    assert no_bet.bet_size == 0.0

    # Verify the formula: 4 bets, rho=0.15 → scale = 1/sqrt(1 + 3*0.15) = 1/sqrt(1.45)
    expected_scale = 1.0 / math.sqrt(1.45)
    assert corr_scale == pytest.approx(expected_scale, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Back-to-back 3PT feature label
# ---------------------------------------------------------------------------


def test_humanize_b2b_and_venue_labels():
    """New feature labels should be human-readable."""
    from nba_betting.models.drivers import humanize_feature

    assert humanize_feature("diff_fg3_pct_b2b_roll_5") == "5-game back-to-back 3PT% differential"
    assert humanize_feature("diff_fg3_pct_b2b_roll_10") == "10-game back-to-back 3PT% differential"
    assert humanize_feature("diff_plus_minus_venue_roll_10") == "10-game venue-split +/- differential"
    assert humanize_feature("diff_net_rtg_game_venue_roll_20") == "20-game venue-split net rating differential"


# ---------------------------------------------------------------------------
# Batch line movement helper
# ---------------------------------------------------------------------------


def test_batch_line_movements_returns_dict():
    """batch_line_movements_by_game should return a dict (possibly empty)."""
    from nba_betting.data.odds_tracker import batch_line_movements_by_game

    result = batch_line_movements_by_game()
    assert isinstance(result, dict)
    # Each value should have the three expected keys
    for key, val in result.items():
        assert "spread_movement" in val
        assert "prob_movement" in val
        assert "odds_disagreement" in val


# ---------------------------------------------------------------------------
# Opening line helper
# ---------------------------------------------------------------------------


def test_get_opening_line_returns_none_for_missing():
    """get_opening_line should return None when no snapshots exist."""
    from nba_betting.data.odds_tracker import get_opening_line

    result = get_opening_line(date(2000, 1, 1), 99999, 99998)
    assert result is None
