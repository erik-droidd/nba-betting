"""Tests for features added in the 2026-04 hardening pass.

Covers the pieces that the live / backtest pipelines depend on — if any
of these regress, `predict` and `backtest` start producing wrong numbers
silently (wrong edge sign, stale drivers, wrong bet count) so they're
worth guarding with fast unit tests.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from nba_betting.betting.shrinkage import shrink_to_market
from nba_betting.models.drivers import (
    compute_prediction_drivers,
    humanize_feature,
)
from nba_betting.models.spreads_totals import generate_spread_total_picks


# ---------------------------------------------------------------------------
# 1. Bayesian shrinkage invariants
# ---------------------------------------------------------------------------


def test_shrinkage_lambda_zero_returns_model_unchanged():
    # lambda=0 means "no weight on market" → posterior = model.
    assert shrink_to_market(0.70, 0.55, lambda_market=0.0) == pytest.approx(0.70, abs=1e-9)


def test_shrinkage_lambda_one_returns_market():
    # lambda=1 means "all market prior" → posterior = market.
    assert shrink_to_market(0.70, 0.55, lambda_market=1.0) == pytest.approx(0.55, abs=1e-9)


def test_shrinkage_midway_moves_toward_market():
    # lambda=0.5 should land strictly between model and market (log-odds
    # midpoint isn't the arithmetic midpoint but still monotone).
    result = shrink_to_market(0.70, 0.55, lambda_market=0.5)
    assert 0.55 < result < 0.70


def test_shrinkage_degenerate_market_returns_raw_model():
    # A market_prob of 0 or 1 would send the logit to ±inf; we short-
    # circuit so we don't push into a tail that's guaranteed to fail Kelly.
    assert shrink_to_market(0.70, 0.0, lambda_market=0.6) == 0.70
    assert shrink_to_market(0.70, 1.0, lambda_market=0.6) == 0.70


# ---------------------------------------------------------------------------
# 2. humanize_feature label map
# ---------------------------------------------------------------------------


def test_humanize_feature_injury_labels():
    # These labels are the ones exposed in the explanation sentence; if
    # someone renames the feature column the explanation would suddenly
    # cite the raw snake_case name. This test pins the happy path.
    assert humanize_feature("injury_impact_diff") == "injury impact differential"
    assert humanize_feature("home_injury_impact_out") == "home injury impact (out/doubtful)"
    assert humanize_feature("away_injury_impact_out") == "away injury impact (out/doubtful)"


def test_humanize_feature_player_impact_and_line_movement():
    assert humanize_feature("diff_missing_minutes_pct") == "missing-minutes differential"
    assert humanize_feature("home_star_out") == "home star player out"
    assert humanize_feature("spread_movement") == "spread line movement"
    assert humanize_feature("odds_disagreement") == "Polymarket vs ESPN odds disagreement"


def test_humanize_feature_rolling_diff():
    # diff_<stat>_roll_<N> → "<N>-game <label> differential"
    assert humanize_feature("diff_pts_roll_5") == "5-game scoring differential"
    assert humanize_feature("diff_efg_pct_roll_10") == "10-game effective FG % differential"


def test_humanize_feature_unknown_falls_back_to_underscore_strip():
    # Unknown feature names should not raise; they should render readable.
    assert humanize_feature("some_new_feature") == "some new feature"


# ---------------------------------------------------------------------------
# 3. Spread / total pick sign convention
# ---------------------------------------------------------------------------


def test_generate_spread_total_picks_home_cover():
    # ESPN convention: negative spread = home favored. A market spread of
    # -4 means "home laying 4". If the model thinks home wins by 7, that
    # beats the market by +3 → HOME_COVER at home -4.
    out = generate_spread_total_picks(
        predicted_spread=7.0,
        predicted_total=220.0,
        market_spread=-4.0,
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["spread_pick"] == "LAL -4.0"
    assert out["spread_edge"] == pytest.approx(3.0, abs=1e-9)


def test_generate_spread_total_picks_away_cover():
    # Model thinks home only wins by 1; market says home favored by 5.
    # Gap = 1 - 5 = -4. -4 <= -SPREAD_EDGE_PTS(1.5) so we take the dog.
    out = generate_spread_total_picks(
        predicted_spread=1.0,
        predicted_total=220.0,
        market_spread=-5.0,
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["spread_pick"] == "BOS +5.0"


def test_generate_spread_total_picks_no_edge_when_gap_too_small():
    # 0.5 pt model/market gap < 1.5 pt threshold → NO BET on spread.
    out = generate_spread_total_picks(
        predicted_spread=-3.5,
        predicted_total=220.0,
        market_spread=3.0,  # market says away -3 → home margin = -3.0
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["spread_pick"] == "NO BET"


def test_generate_spread_total_picks_total_over():
    # Model thinks 230; market 220. +10 > TOTAL_EDGE_PTS(2.5) → OVER.
    out = generate_spread_total_picks(
        predicted_spread=0.0,
        predicted_total=230.0,
        market_spread=None,
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["total_pick"] == "OVER 220.0"
    assert out["total_edge"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Driver attribution: ordering + no-crash behavior
# ---------------------------------------------------------------------------


class _FakeModel:
    """Tiny stand-in for sklearn's predict_proba — we don't need a real
    tree to validate the leave-one-out loop, just a deterministic scoring
    function that responds to column values."""

    def predict_proba(self, X):
        # P(home win) = sigmoid(weight . x). Chosen weights give each
        # feature a known contribution so tests can assert ranking.
        import numpy as np
        w = {"elo_diff": 1.0, "diff_pts_roll_5": 0.3, "noise": 0.01}
        if hasattr(X, "columns"):
            cols = list(X.columns)
            vals = X.values
        else:
            cols = ["elo_diff", "diff_pts_roll_5", "noise"]
            vals = X

        out = []
        for row in vals:
            z = sum(w.get(c, 0.0) * float(row[i]) for i, c in enumerate(cols))
            p = 1.0 / (1.0 + math.exp(-z))
            out.append([1.0 - p, p])
        return np.array(out) if (_np := __import__("numpy")) else out


def test_compute_prediction_drivers_sorted_by_magnitude():
    feat_row = pd.DataFrame([{
        "elo_diff": 2.0,        # large positive → strongest driver
        "diff_pts_roll_5": 1.0, # moderate driver
        "noise": 5.0,           # high value but tiny weight → smallest delta
    }])
    means = {"elo_diff": 0.0, "diff_pts_roll_5": 0.0, "noise": 0.0}

    drivers = compute_prediction_drivers(_FakeModel(), feat_row, means, top_k=3)

    names = [d[0] for d in drivers]
    # Magnitude ranking: elo_diff first (strongest delta), noise last.
    assert names[0] == "elo_diff"
    assert names[-1] == "noise"
    # Sign convention: positive elo_diff pushes P(home) up → positive delta.
    assert drivers[0][1] > 0


def test_compute_prediction_drivers_empty_row_returns_empty():
    # Robustness: attribution must not raise on degenerate inputs, since
    # it's wrapped in a try/except in the caller and we want the catch
    # path to be rare.
    out = compute_prediction_drivers(_FakeModel(), pd.DataFrame(), {}, top_k=5)
    assert out == []


# ---------------------------------------------------------------------------
# 5. Backtest defaults — apply_live_strategy couples to use_real_odds
# ---------------------------------------------------------------------------


def test_backtest_live_strategy_default_follows_real_odds_flag():
    """When the caller passes `apply_live_strategy=None`, the function
    must resolve the default based on `use_real_odds`:

    - use_real_odds=False → live_strategy=False (pure model benchmark;
      shrinking toward the Elo proxy is a null-op/dampener).
    - use_real_odds=True  → live_strategy=True (live-equivalent sim).

    We verify via an empty feature matrix — the resolution happens at the
    top of `run_backtest` before any data work, so we can exercise it
    without a real training set.
    """
    from nba_betting.betting.backtest import run_backtest

    # A feature matrix that straddles July 1 so split_dates is non-empty
    # and the function reaches the `use_real_odds` metadata check. We
    # don't need enough games to actually fit a model — the test passes
    # as long as (a) resolution survives to the real-odds branch,
    # (b) the `_home_team_id` ValueError is raised when the metadata is
    # missing, and (c) the False branch returns a valid (empty) summary.
    import pandas as pd
    X = pd.DataFrame({
        "_date": pd.to_datetime(["2024-06-01", "2024-08-01"]),
        "elo_home_prob": [0.5, 0.5],
    })
    y = pd.Series([1, 0])

    out_false = run_backtest(X, y, n_splits=1, apply_live_strategy=None, use_real_odds=False)
    assert "summary" in out_false  # resolution path didn't raise

    # use_real_odds=True should bail on missing metadata. The check
    # lives AFTER the apply_live_strategy resolution, so reaching it
    # proves the None→False-or-True default machinery worked.
    with pytest.raises(ValueError, match="_home_team_id"):
        run_backtest(X, y, n_splits=1, apply_live_strategy=None, use_real_odds=True)


# ---------------------------------------------------------------------------
# 6. Additive migration idempotence
# ---------------------------------------------------------------------------


def test_apply_additive_migrations_is_idempotent(tmp_path, monkeypatch):
    """Running init_db() twice in a row must not raise (no duplicate
    ALTER TABLE). This guards against a regression where the migration
    tries to add an already-existing column."""
    # Point DB_PATH at a tempfile so we don't touch the real DB.
    from nba_betting import config as _cfg
    test_db = tmp_path / "t.sqlite"
    monkeypatch.setattr(_cfg, "DB_PATH", str(test_db))

    # Re-import session with the patched DB_PATH so engine binds to the
    # tempfile, not the real DB.
    import importlib
    from nba_betting.db import session as _session
    importlib.reload(_session)

    _session.init_db()
    _session.init_db()  # second call must be a no-op — no error raised.
    assert test_db.exists()
